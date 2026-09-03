# -*- coding: utf-8 -*-
"""source_adapted/models/g3_core_v4.py

V4 connectivity-conditioned selective SSM core.

FASTFIX changes
---------------
1. Backward scan reverses BOTH derived graph-feature sequences:
   g_sc[t] = SC @ H[t] and g_fc[t] = FC[t] @ H[t].
   The static SC adjacency itself is never reversed.
2. Expensive gate quantile/.item() diagnostics are computed only when
   collect_stats=True. Training and normal DEV-AUC evaluation use collect_stats=False.
3. lambda_value stays as a Tensor on the fast path, avoiding per-forward GPU sync.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

STATE_DIM = 128
CONTEXT_RANK = 32
NUM_NODES = 18
MODEL_STEPS = 53
FUSION_DIM = 128
CONTEXT_FEAT = 96


class ConnectivityContext(nn.Module):
    def __init__(self, fusion_dim=FUSION_DIM, context_feat=CONTEXT_FEAT,
                 context_rank=CONTEXT_RANK, state_dim=STATE_DIM,
                 context_dropout=0.10):
        super().__init__()
        self.rank = context_rank
        self.state_dim = state_dim
        self.W_sc = nn.Linear(context_feat, context_rank, bias=False)
        self.W_fc = nn.Linear(context_feat, context_rank, bias=False)
        self.W_gate = nn.Linear(fusion_dim + context_rank, 1)
        nn.init.zeros_(self.W_gate.weight)
        nn.init.zeros_(self.W_gate.bias)
        self.R_delta = nn.Linear(context_rank, state_dim, bias=False)
        self.R_B = nn.Linear(context_rank, state_dim, bias=False)
        self.R_C = nn.Linear(context_rank, state_dim, bias=False)
        for w in (self.R_delta, self.R_B, self.R_C):
            nn.init.zeros_(w.weight)
        self.context_dropout = (
            nn.Dropout(context_dropout) if context_dropout > 0 else None
        )

    def q_from_context(self, g_sc, g_fc):
        q_sc = F.silu(self.W_sc(g_sc))
        q_fc = F.silu(self.W_fc(g_fc))
        q = torch.tanh(q_sc + q_fc)
        if self.context_dropout is not None and self.training:
            q = self.context_dropout(q)
        return q

    def residual_corrections(self, q):
        return (
            torch.tanh(self.R_delta(q)),
            torch.tanh(self.R_B(q)),
            torch.tanh(self.R_C(q)),
        )

    def node_gate(self, e_t, q_t):
        return torch.sigmoid(self.W_gate(torch.cat([e_t, q_t], dim=-1)))


class G3CellV4(nn.Module):
    def __init__(self, state_dim=STATE_DIM, num_nodes=NUM_NODES):
        super().__init__()
        self.state_dim = state_dim
        self.num_nodes = num_nodes
        self.W_in = nn.Linear(state_dim, state_dim)
        self.W_delta = nn.Linear(state_dim, state_dim)
        self.W_B = nn.Linear(state_dim, state_dim)
        self.W_C = nn.Linear(state_dim, state_dim)
        self.W_out = nn.Linear(state_dim, state_dim)
        self.A_log = nn.Parameter(torch.randn(state_dim) * 0.1)
        for w in (self.W_in, self.W_delta, self.W_B, self.W_C):
            nn.init.xavier_uniform_(w.weight)

    def step(self, e_t, scale, r_delta, r_B, r_C, h):
        u = self.W_in(e_t)
        delta_base = self.W_delta(e_t)
        B_base = self.W_B(e_t)
        C_base = self.W_C(e_t)

        delta_logits = delta_base + scale * r_delta
        delta = F.softplus(delta_logits).clamp(max=20.0)
        B = B_base + scale * r_B
        C = C_base + scale * r_C

        # autocast keeps numerically sensitive exp operations in a safe dtype
        # on supported PyTorch/CUDA versions; parameters remain FP32.
        A = torch.exp(self.A_log)
        Abar = torch.exp(-delta * A)
        h_new = Abar * h + delta * B * u
        y = C * h_new
        out_y = self.W_out(y)
        return out_y, h_new, delta, Abar


class G3CoreV4(nn.Module):
    def __init__(self, fusion_dim=FUSION_DIM, state_dim=STATE_DIM,
                 num_nodes=NUM_NODES, model_steps=MODEL_STEPS,
                 context_feat=CONTEXT_FEAT, layers=2,
                 context_rank=CONTEXT_RANK, context_dropout=0.10,
                 lambda_init=0.10, inter_layer_rms=True,
                 enable_direction_lora=False, lora_rank=8):
        super().__init__()
        self.state_dim = state_dim
        self.num_nodes = num_nodes
        self.model_steps = model_steps
        self.layers = layers
        self.fusion_dim = fusion_dim
        self.enable_direction_lora = bool(enable_direction_lora)
        self.lora_rank = int(lora_rank)

        self.ctx = ConnectivityContext(
            fusion_dim=fusion_dim,
            context_feat=context_feat,
            context_rank=context_rank,
            state_dim=state_dim,
            context_dropout=context_dropout,
        )
        self.lambda_logit = nn.Parameter(
            torch.tensor(
                math.log(lambda_init / (1.0 - lambda_init)),
                dtype=torch.float32,
            )
        )

        self.cells = nn.ModuleList(
            [G3CellV4(state_dim, num_nodes) for _ in range(layers)]
        )
        if inter_layer_rms:
            self.inter_layer_rms = nn.ModuleList(
                [nn.LayerNorm(state_dim) for _ in range(max(0, layers - 1))]
            )
        else:
            self.inter_layer_rms = None
        self.final_ln = nn.LayerNorm(state_dim)

        # Kept as a scaffold for future V5 work. It is disabled in V4.
        if self.enable_direction_lora:
            for cell in self.cells:
                cell.lora_back = nn.ModuleList(
                    [_LoRALinear(state_dim, state_dim, lora_rank) for _ in range(3)]
                )

    @property
    def lambda_value(self):
        return torch.sigmoid(self.lambda_logit)

    def _scan(self, x_seq, g_sc, g_fc, reverse=False, collect_stats=False):
        B, T, N, _ = x_seq.shape
        lam = self.lambda_value

        x = x_seq
        g_sc_use = g_sc
        g_fc_use = g_fc

        if reverse:
            x = torch.flip(x, dims=[1])
            # SC adjacency is static, but g_sc[t] is a TIME-VARYING derived
            # feature sequence and therefore must remain aligned with x[t].
            g_sc_use = torch.flip(g_sc_use, dims=[1])
            g_fc_use = torch.flip(g_fc_use, dims=[1])

        q = self.ctx.q_from_context(g_sc_use, g_fc_use)
        r_delta, r_B, r_C = self.ctx.residual_corrections(q)

        gate_vals = [] if collect_stats else None
        last_alpha = None
        final_h = None
        y_seq_last = None

        for li, cell in enumerate(self.cells):
            h = x.new_zeros(B, N, self.state_dim)
            outs = []
            alphas = []

            for t in range(T):
                e_t = x[:, t]
                q_t = q[:, t]
                alpha_t = self.ctx.node_gate(e_t, q_t)
                scale_t = lam * alpha_t

                out_y, h, _dt, _ab = cell.step(
                    e_t,
                    scale_t,
                    r_delta[:, t],
                    r_B[:, t],
                    r_C[:, t],
                    h,
                )
                outs.append(out_y)
                alphas.append(alpha_t)
                if collect_stats:
                    gate_vals.append(alpha_t)

            x = torch.stack(outs, dim=1)
            if li == self.layers - 1:
                y_seq_last = x
            final_h = h
            last_alpha = torch.stack(alphas, dim=1)

            if li < self.layers - 1 and self.inter_layer_rms is not None:
                x = self.inter_layer_rms[li](x)

        hT = self.final_ln(final_h)

        gate_stats = {}
        if collect_stats and gate_vals:
            # This path is intentionally diagnostics-only because quantile/item
            # synchronizes the GPU with the CPU.
            g = torch.cat([v.reshape(-1) for v in gate_vals]).detach().float()
            gate_stats["gate_mean"] = g.mean().item()
            gate_stats["gate_p95"] = g.quantile(0.95).item()

        return {
            "y_seq": y_seq_last,
            "hT": hT,
            "q": q,
            "alpha": last_alpha,
            "gate_stats": gate_stats,
            "lambda_value": lam,
        }

    def forward(self, E, g_sc, g_fc, reverse=False, collect_stats=False):
        if E.dim() != 4:
            raise ValueError("E must be (B,T,N,state)")
        return self._scan(
            E,
            g_sc,
            g_fc,
            reverse=reverse,
            collect_stats=collect_stats,
        )


class _LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, rank=8, scale=1.0):
        super().__init__()
        self.A = nn.Parameter(torch.randn(in_f, rank) * 0.02)
        self.B = nn.Parameter(torch.zeros(rank, out_f))
        self.scale = scale

    def forward(self, x):
        return x @ (self.A @ self.B) * self.scale
