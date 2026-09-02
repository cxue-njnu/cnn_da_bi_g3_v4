# -*- coding: utf-8 -*-
"""source_adapted/models/cnn_da_bi_g3_v4.py - CNN_DA_BI_G3_V4 composite model.

DESIGN
    1. CNN (shared per-electrode multi-scale) encodes LOCAL spectro-temporal patterns
       -> H (B,53,18,96); raw STFT skip -> H = LayerNorm(H_cnn + H_raw).
    2. SC/FC aggregation on the CNN latent: g_sc, g_fc (each B,53,18,96).
    3. Shared Fusion: Z_t = concat(H,g_sc,g_fc) 288 -> 192 -> 128 = E_t (per direction).
    4. G3 connectivity-conditioned selective SSM run forward AND backward.
    5. Selective Readout uses y_t = W_out(C_t*h_t) (NOT just the final hidden state)
       with temporal-node softmax attention and node attention on h_T.
    6. Direction-Adaptive Gate: shared psi() scores each direction -> softmax weights.
    7. Classifier on r_bi ; Direction Auxiliary loss on shared head over r_f / r_b.

FORWARD / BACKWARD : The CNN encoder + fusion run ONCE. Only the G3 core (and its
readout) is run in both time orders. Forward 0->52. Backward 52->0 (the G3 flips E
and the FC-derived context; static SC passes through).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .multiscale_stft_cnn import MultiScaleSTFTCNN, MODEL_STEPS, NUM_NODES, FEATURE_DIM
from .g3_core_v4 import G3CoreV4, STATE_DIM, CONTEXT_RANK, FUSION_DIM, CONTEXT_FEAT


def normalize_adj(adj):
    row_sum = adj.abs().sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return adj / row_sum


def _distribution_entropy(p):
    """p: (B, K) row-stochastic -> entropy H = -sum p*log p per row."""
    return -(p * torch.log(p.clamp(min=1e-12))).sum(dim=-1)


class SharedFusion(nn.Module):
    """SC/FC latent aggregation over the CNN latent + shared fusion MLP -> E_t."""

    def __init__(self, context_feat=CONTEXT_FEAT, fusion_dim=FUSION_DIM,
                 num_nodes=NUM_NODES, dropout=0.10):
        super().__init__()
        self.num_nodes = num_nodes
        self.context_feat = context_feat
        cat = context_feat * 3
        self.fc1 = nn.Linear(cat, 192)
        self.ln1 = nn.LayerNorm(192)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(192, fusion_dim)
        self.ln2 = nn.LayerNorm(fusion_dim)

    def aggregate(self, H, sc, fc):
        B, T, N, _ = H.shape
        sc_n = normalize_adj(sc if sc.dim() == 2 else sc.mean(dim=0))
        g_sc = torch.einsum('nk,btnf->btnf', sc_n, H)
        g_fc = torch.einsum('btnk,btkf->btnf', normalize_adj(fc), H)
        return g_sc, g_fc

    def forward(self, H, sc, fc):
        g_sc, g_fc = self.aggregate(H, sc, fc)
        Z = torch.cat([H, g_sc, g_fc], dim=-1)   # (B,T,N,288)
        e = self.gelu(self.ln1(self.fc1(Z)))
        if self.training:
            e = self.drop(e)
        e = self.ln2(self.fc2(e))
        return e, g_sc, g_fc


class SelectiveReadout(nn.Module):
    def __init__(self, fusion_dim=FUSION_DIM, context_rank=CONTEXT_RANK):
        super().__init__()
        in_feat = fusion_dim + context_rank + 1  # 128+32+1 = 161
        self.scorer = nn.Sequential(nn.Linear(in_feat, 64), nn.GELU(), nn.Linear(64, 1))
        self.node_attn = nn.Linear(fusion_dim, 1)
        self.proj = nn.Linear(2 * fusion_dim, fusion_dim)
        self.ln = nn.LayerNorm(fusion_dim)

    def forward(self, y_seq, hT, q, alpha):
        B, T, N, _ = y_seq.shape
        cat = torch.cat([y_seq, q, alpha], dim=-1)
        score = self.scorer(cat).squeeze(-1).reshape(B, T * N)
        attn = F.softmax(score, dim=1)                    # (B, T*N)
        y_flat = y_seq.reshape(B, T * N, -1)
        r_seq = torch.einsum('bk,bkf->bf', attn, y_flat)  # (B,128)
        a_node = F.softmax(self.node_attn(hT).squeeze(-1), dim=1)  # (B,N)
        r_last = torch.einsum('bn,bnf->bf', a_node, hT)   # (B,128)
        r_dir = self.ln(self.proj(torch.cat([r_last, r_seq], dim=-1)))
        return (r_dir, r_seq, r_last, attn,
                _distribution_entropy(attn), attn.max(dim=-1).values)


class DirectionModule(nn.Module):
    def __init__(self, fusion_dim=FUSION_DIM, context_rank=CONTEXT_RANK):
        super().__init__()
        self.psi = nn.Sequential(nn.Linear(fusion_dim + context_rank, 64), nn.GELU(),
                                 nn.Linear(64, 1))
        self.head_z = nn.Linear(fusion_dim, 1)

    def score(self, r, q_summary):
        return self.psi(torch.cat([r, q_summary], dim=-1))

    def aux(self, r):
        return self.head_z(r).squeeze(-1)


class FinalClassifier(nn.Module):
    def __init__(self, fusion_dim=FUSION_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(fusion_dim, 64), nn.GELU(),
                                 nn.Dropout(0.20), nn.Linear(64, 1))

    def forward(self, r_bi):
        return self.net(r_bi).squeeze(-1)


class CNN_DA_BI_G3_V4(nn.Module):
    def __init__(self, num_nodes=NUM_NODES, feature_dim=FEATURE_DIM, model_steps=MODEL_STEPS,
                 fusion_dim=FUSION_DIM, state_dim=STATE_DIM, context_rank=CONTEXT_RANK,
                 context_feat=CONTEXT_FEAT, layers=2, context_dropout=0.10,
                 lambda_init=0.10, direction_dropout=0.10,
                 enable_direction_lora=False, lora_rank=8,
                 encoder_hidden=96, inter_layer_rms=True):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.state_dim = state_dim
        self.num_nodes = num_nodes
        self.model_steps = model_steps
        self.direction_dropout = float(direction_dropout)
        self.direction_config = {
            "weight_sharing": True, "fusion": "direction_adaptive_gate",
            "within_window_noncausal": True, "uses_future_beyond_clip": False,
            "directions": 2, "direction_lora": bool(enable_direction_lora),
        }
        self.encoder = MultiScaleSTFTCNN(
            num_nodes=num_nodes, feature_dim=feature_dim, model_steps=model_steps,
            hidden=encoder_hidden)
        self.fusion = SharedFusion(context_feat=context_feat, fusion_dim=fusion_dim,
                                   num_nodes=num_nodes, dropout=0.10)
        self.g3 = G3CoreV4(
            fusion_dim=fusion_dim, state_dim=state_dim, num_nodes=num_nodes,
            model_steps=model_steps, context_feat=context_feat, layers=layers,
            context_rank=context_rank, context_dropout=context_dropout,
            lambda_init=lambda_init, inter_layer_rms=inter_layer_rms,
            enable_direction_lora=enable_direction_lora, lora_rank=lora_rank)
        self.readout = SelectiveReadout(fusion_dim=fusion_dim, context_rank=context_rank)
        self.dir = DirectionModule(fusion_dim=fusion_dim, context_rank=context_rank)
        self.classifier = FinalClassifier(fusion_dim=fusion_dim)

    def _direction(self, E, g_sc, g_fc, reverse):
        scan = self.g3(E, g_sc, g_fc, reverse=reverse)
        y_seq = scan["y_seq"]; hT = scan["hT"]; q = scan["q"]; alpha = scan["alpha"]
        r, r_seq, r_last, attn, entropy, peak = self.readout(y_seq, hT, q, alpha)
        return {
            "r": r, "r_seq": r_seq, "r_last": r_last, "attn": attn,
            "entropy": entropy, "peak": peak, "qsum": q.mean(dim=(1, 2)),
            "gate_stats": scan["gate_stats"], "lambda_value": scan["lambda_value"],
        }

    def forward(self, stft, sc=None, fc=None, return_arrays=False):
        if sc is None or fc is None:
            raise ValueError("CNN_DA_BI_G3_V4 requires sc and fc")
        B = stft.shape[0]
        H = self.encoder(stft)
        E, g_sc, g_fc = self.fusion(H, sc, fc)

        fw = self._direction(E, g_sc, g_fc, reverse=False)
        bw = self._direction(E, g_sc, g_fc, reverse=True)

        s_f = self.dir.score(fw["r"], fw["qsum"])             # (B,1)
        s_b = self.dir.score(bw["r"], bw["qsum"])
        w2 = torch.softmax(torch.cat([s_f, s_b], dim=-1), dim=-1)  # (B,2)
        w_f = w2[:, 0:1]; w_b = w2[:, 1:2]

        # --- direction dropout (TRAIN only, per-sample) ---
        if self.training and self.direction_dropout > 0:
            u = torch.rand(B, device=stft.device)
            fwd_only = u < self.direction_dropout
            bwd_only = (u >= self.direction_dropout) & (u < 2 * self.direction_dropout)
            mask_adapt = ((~fwd_only) & (~bwd_only)).float().unsqueeze(-1)  # (B,1)
            w_f = mask_adapt * w_f + fwd_only.float().unsqueeze(-1)
            w_b = mask_adapt * w_b + bwd_only.float().unsqueeze(-1)

        r_bi = w_f * fw["r"] + w_b * bw["r"]                  # (B,128)
        z_final = self.classifier(r_bi)                        # (B,)
        z_f = self.dir.aux(fw["r"])                            # (B,)
        z_b = self.dir.aux(bw["r"])

        if not return_arrays:
            return z_final
        return {
            "logit": z_final,
            "forward_raw_logit": z_f,
            "backward_raw_logit": z_b,
            "direction_weight_forward": w_f.squeeze(-1),
            "direction_weight_backward": w_b.squeeze(-1),
            "readout_entropy": fw["entropy"],
            "readout_peak_weight": fw["peak"],
            "gate_mean": fw["gate_stats"].get("gate_mean"),
            "gate_p95": fw["gate_stats"].get("gate_p95"),
            "lambda_value": fw["lambda_value"],
            "r_bi": r_bi, "z_f": z_f, "z_b": z_b,
        }


def build_cnn_da_bi_g3_v4(model_config=None):
    cfg = model_config or {}
    return CNN_DA_BI_G3_V4(
        num_nodes=int(cfg.get("nodes", NUM_NODES)),
        feature_dim=int(cfg.get("feature_dim", FEATURE_DIM)),
        model_steps=int(cfg.get("model_steps", MODEL_STEPS)),
        fusion_dim=int(cfg.get("fusion_dim", FUSION_DIM)),
        state_dim=int(cfg.get("state_dim", STATE_DIM)),
        context_rank=int(cfg.get("context_rank", CONTEXT_RANK)),
        context_feat=int(cfg.get("context_feat", CONTEXT_FEAT)),
        layers=int(cfg.get("layers", 2)),
        context_dropout=float(cfg.get("context_dropout", 0.10)),
        lambda_init=float(cfg.get("lambda_init", 0.10)),
        direction_dropout=float(cfg.get("direction_dropout", 0.10)),
        enable_direction_lora=bool(cfg.get("enable_direction_lora", False)),
        lora_rank=int(cfg.get("direction_lora_rank", 8)),
        encoder_hidden=int(cfg.get("encoder_hidden", 96)),
        inter_layer_rms=bool(cfg.get("inter_layer_rms", True)),
    )


RESIDUAL_TOKENS = {"W_sc", "W_fc", "W_gate", "R_delta", "R_B", "R_C", "lambda_logit"}


def _g3_group_class(name):
    leaf = set(name.split("."))
    return 3 if (leaf & RESIDUAL_TOKENS) else 2


def build_optimizer_v4(model, lr_map=None, wd_map=None, enable_direction_lora=False):
    lr_map = lr_map or {"g1": 3e-4, "g2": 1e-3, "g3": 3e-4, "g4": 5e-4}
    wd_map = wd_map or {"g1": 5e-4, "g2": 5e-4, "g3": 1e-3, "g4": 5e-4}
    groups = {1: [], 2: [], 3: [], 4: []}
    names = {1: [], 2: [], 3: [], 4: []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder.") or name.startswith("fusion."):
            g = 1
        elif name.startswith("g3."):
            g = _g3_group_class(name[len("g3."):])
        else:
            g = 4
        groups[g].append(p)
        names[g].append(name)

    order = [1, 2, 3, 4]
    configs = []
    for g in order:
        if not groups[g]:
            continue
        configs.append({"params": groups[g], "lr": lr_map["g%d" % g],
                        "weight_decay": wd_map["g%d" % g]})

    n_grouped = sum(len(v) for v in groups.values())
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    if n_grouped != n_trainable:
        raise RuntimeError("optimizer coverage mismatch: grouped=%d trainable=%d"
                           % (n_grouped, n_trainable))

    optimizer = torch.optim.AdamW(configs)
    idx_of = {}
    seen = 0
    for g in order:
        if not groups[g]:
            continue
        idx_of[g] = seen
        seen += 1
    optimizer.v4_group_index = {"g2": idx_of.get(2), "g3": idx_of.get(3)}
    optimizer.v4_full_lr = [g["lr"] for g in optimizer.param_groups]

    optimizer_config = {
        "optimizer": "AdamW",
        "groups": {
            "group1_cnn_fusion": {"lr": lr_map["g1"], "weight_decay": wd_map["g1"], "n_params": len(groups[1])},
            "group2_g3_base": {"lr": lr_map["g2"], "weight_decay": wd_map["g2"], "n_params": len(groups[2])},
            "group3_connectivity_residual": {"lr": lr_map["g3"], "weight_decay": wd_map["g3"], "n_params": len(groups[3])},
            "group4_readout_direction_classifier": {"lr": lr_map["g4"], "weight_decay": wd_map["g4"], "n_params": len(groups[4])},
        },
        "total_trainable_params": n_trainable,
        "coverage_ok": n_grouped == n_trainable,
    }
    return optimizer, optimizer_config
