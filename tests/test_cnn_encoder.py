# -*- coding: utf-8 -*-
"""T1, T2 - CNN encoder shapes (no time/node loss)."""
import torch
from models.multiscale_stft_cnn import MultiScaleSTFTCNN

def _enc():
    torch.manual_seed(0)
    return MultiScaleSTFTCNN().eval()

def test_cnn_shape_output_53x18x96():
    x = torch.randn(2, 53, 18, 70)
    out = _enc()(x)
    assert tuple(out.shape) == (2, 53, 18, 96), out.shape

def test_cnn_no_time_or_node_loss():
    x = torch.randn(3, 53, 18, 70)
    out = _enc()(x)
    assert out.shape[1] == 53, "time dim lost"
    assert out.shape[2] == 18, "node dim lost"

def test_cnn_batch_forward_reproducible_eval():
    enc = _enc()
    x = torch.randn(2, 53, 18, 70)
    a = enc(x); b = enc(x)
    assert torch.allclose(a, b, atol=1e-6)
