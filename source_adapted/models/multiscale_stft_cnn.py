# -*- coding: utf-8 -*-
"""source_adapted/models/multiscale_stft_cnn.py - time-reversal-symmetric
multi-scale STFT CNN encoder.

SCIENCE (V4 module 1): a SHARED per-electrode CNN learns LOCAL spectro-temporal
patterns. The same weights are applied independently to each of the 18 EEG nodes
(axis = B*18); there is NO channel-index convolution -- nodes are never mixed by the
CNN, only time x frequency are mixed locally.

TIME-REVERSAL EQUIVARIANCE (needed by the T7 reversal test)
    Re-encoding a time-reversed spectrogram must yield the time-reversed encoding.
    Every local temporal kernel is forced to be centre-symmetric over the TIME axis
    at every forward call (w_sym = 0.5*(w + flip(w, time))). Combined with SAME/odd
    kernel padding this makes the whole encoding equivariant to within-window time
    reversal -- the property the direction-adaptive bidirectional wrapper requires.

INPUT  x   : (B,53,18,70) -> (B*18,1,53,70)
STAGES
  1. Multi-scale first layer: Branch A (3,3), Branch B (3,7), Branch C (5,5), each
     1->32, conv -> GroupNorm -> GELU, concatenated -> 96 channels -> 1x1 fuse.
  2. Residual local block: time-symmetric depthwise (3,3,groups=96) + pointwise +
     GroupNorm -> residual add. T=53 preserved (no downsampling).
  3. Frequency compression: adaptive pooling of the FREQUENCY axis to one bin keeps
     time (53) and nodes intact; a linear projection gives 96 dims.
  4. Raw skip: H_raw = Linear(70,96)(x);  H = LayerNorm(H_cnn + H_raw).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_STEPS = 53
NUM_NODES = 18
FEATURE_DIM = 70


def _sym_w(w):
    """Centre-symmetrise a conv2d weight over its TIME axis (kernel index 2)."""
    return 0.5 * (w + torch.flip(w, dims=[2]))


def _gn(c, groups=None):
    while True:
        g = groups if groups else max(1, c // 4)
        if g > 1 and c % g != 0:
            groups = g - 1 if groups else None
            g = groups
        g = max(1, g or 1)
        if c % g == 0:
            return nn.GroupNorm(min(g, c), c)
        if groups is not None:
            groups -= 1
        else:
            return nn.GroupNorm(1, c)


class _Branch(nn.Module):
    """conv(kh,kw) -> GroupNorm -> GELU on a 2D spectrogram map."""

    def __init__(self, cin, cout, kh, kw):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=(kh, kw))
        self.kh = kh
        self.kw = kw
        self.gn = nn.GroupNorm(min(cout, 32), cout)

    def forward(self, x):
        w = _sym_w(self.conv.weight)
        out = F.conv2d(x, w, bias=self.conv.bias, padding=(self.kh // 2, self.kw // 2))
        return F.gelu(self.gn(out))


class _ResidualBlock(nn.Module):
    def __init__(self, c=96):
        super().__init__()
        self.dw = nn.Conv2d(c, c, kernel_size=(3, 3), groups=c)
        self.pw = nn.Conv2d(c, c, kernel_size=1)
        self.gn = nn.GroupNorm(min(c, 32), c)

    def forward(self, x):
        w = _sym_w(self.dw.weight)
        d = F.conv2d(x, w, bias=self.dw.bias, padding=(1, 1), groups=self.dw.groups)
        p = F.conv2d(d, self.pw.weight, bias=self.pw.bias, padding=0)
        return x + F.gelu(self.gn(p))


class MultiScaleSTFTCNN(nn.Module):
    def __init__(self, num_nodes=NUM_NODES, feature_dim=FEATURE_DIM,
                 model_steps=MODEL_STEPS, hidden=96):
        super().__init__()
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.model_steps = model_steps
        self.hidden = hidden
        self.branch_a = _Branch(1, 32, 3, 3)
        self.branch_b = _Branch(1, 32, 3, 7)
        self.branch_c = _Branch(1, 32, 5, 5)
        self.fuse1 = nn.Conv2d(96, 96, kernel_size=1)
        self.fuse1_gn = nn.GroupNorm(min(96, 32), 96)
        self.block = _ResidualBlock(96)
        self.freq_pool = nn.AdaptiveAvgPool2d((model_steps, 1))
        self.freq_proj = nn.Linear(96, hidden)
        self.raw_linear = nn.Linear(feature_dim, hidden)
        self.h_ln = nn.LayerNorm(hidden)

    def forward(self, x):
        B, T, Nn, _ = x.shape  # F = _
        y = x.reshape(B * Nn, 1, T, x.shape[3])
        ha = self.branch_a(y)
        hb = self.branch_b(y)
        hc = self.branch_c(y)
        h = torch.cat([ha, hb, hc], dim=1)                # (B*18,96,T,F)
        h = F.gelu(self.fuse1_gn(F.conv2d(h, self.fuse1.weight, bias=self.fuse1.bias)))
        h = self.block(h)
        h = self.freq_pool(h).squeeze(-1).permute(0, 2, 1)   # (B*18,T,96)
        H_cnn = self.freq_proj(h).reshape(B, Nn, T, self.hidden).permute(0, 2, 1, 3)
        H_raw = self.raw_linear(x)
        return self.h_ln(H_cnn + H_raw)
