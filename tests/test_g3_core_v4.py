# -*- coding: utf-8 -*-
"""T4 + G3 core scans: forward/backward representations, connectivity context."""
import torch
from models.cnn_da_bi_g3_v4 import build_cnn_da_bi_g3_v4

def _ready(dev="cpu"):
    torch.manual_seed(1)
    m = build_cnn_da_bi_g3_v4().to(dev).eval()
    B = 2
    stft = torch.randn(B, 53, 18, 70)
    fc = torch.randn(B, 53, 18, 18).abs() + 0.1
    sc = torch.rand(18, 18) + 0.1
    H = m.encoder(stft)
    E, g_sc, g_fc = m.fusion(H, sc, fc)
    return m, E, g_sc, g_fc

def test_g3_state_dim_config():
    m = build_cnn_da_bi_g3_v4()
    assert m.g3.state_dim == 128
    assert m.g3.layers == 2
    assert m.g3.ctx.rank == 32
    v = float(torch.sigmoid(m.g3.lambda_logit))
    assert 0.0 <= v <= 1.0

def test_forward_backward_representations_shape():
    m, E, g_sc, g_fc = _ready()
    fw = m._direction(E, g_sc, g_fc, reverse=False)
    bw = m._direction(E, g_sc, g_fc, reverse=True)
    assert tuple(fw["r"].shape) == (2, 128)
    assert tuple(bw["r"].shape) == (2, 128)
    assert tuple(fw["qsum"].shape) == (2, 32)
    assert torch.isfinite(fw["r"]).all() and torch.isfinite(bw["r"]).all()

def test_selective_readout_y_produced_from_scan():
    m, E, g_sc, g_fc = _ready()
    scan = m.g3(E, g_sc, g_fc, reverse=False)
    assert tuple(scan["y_seq"].shape) == (2, 53, 18, 128)
    assert tuple(scan["hT"].shape) == (2, 18, 128)
    assert tuple(scan["q"].shape) == (2, 53, 18, 32)
    assert tuple(scan["alpha"].shape) == (2, 53, 18, 1)

def test_optimizer_four_groups_and_residual_tokens():
    from models.cnn_da_bi_g3_v4 import build_optimizer_v4, RESIDUAL_TOKENS
    m = build_cnn_da_bi_g3_v4()
    opt, oc = build_optimizer_v4(m)
    assert len(opt.param_groups) == 4
    assert oc["coverage_ok"] is True
    assert {"W_sc", "lambda_logit"} <= RESIDUAL_TOKENS
