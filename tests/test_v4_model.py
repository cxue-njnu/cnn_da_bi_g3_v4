# -*- coding: utf-8 -*-
"""T3 (SC/FC latent agg), T5 (selective readout), T8 (gradient), mechanisms."""
import torch
import torch.nn as nn
from models.cnn_da_bi_g3_v4 import build_cnn_da_bi_g3_v4, SelectiveReadout
from models.cnn_da_bi_g3_v4 import build_optimizer_v4

def _inputs(B=2, dev="cpu"):
    torch.manual_seed(3)
    stft = torch.randn(B, 53, 18, 70, device=dev)
    fc = torch.randn(B, 53, 18, 18, device=dev).abs() + 0.1
    sc = torch.rand(18, 18, device=dev) + 0.1
    return stft, sc, fc

def test_sc_fc_latent_aggregation_shape():
    m = build_cnn_da_bi_g3_v4().eval()
    stft, sc, fc = _inputs()
    H = m.encoder(stft)
    g_sc, g_fc = m.fusion.aggregate(H, sc, fc)
    assert tuple(g_sc.shape) == (2, 53, 18, 96)
    assert tuple(g_fc.shape) == (2, 53, 18, 96)
    assert torch.isfinite(g_sc).all() and torch.isfinite(g_fc).all()

def test_selective_readout_attention_weights():
    m = build_cnn_da_bi_g3_v4().eval()
    B = 2
    y_seq = torch.randn(B, 53, 18, 128)
    hT = torch.randn(B, 18, 128)
    q = torch.randn(B, 53, 18, 32)
    alpha = torch.rand(B, 53, 18, 1).sigmoid()
    ro = SelectiveReadout()
    r, r_seq, r_last, attn, ent, peak = ro(y_seq, hT, q, alpha)
    s = attn.sum(dim=1)
    assert torch.allclose(s, torch.ones(B), atol=1e-3)
    assert torch.isfinite(attn).all()
    assert tuple(r.shape) == (B, 128)
    assert (ent >= 0).all()
    assert (peak >= 0).all() and (peak <= 1).all()

def test_gradient_flows_finite_everywhere():
    torch.manual_seed(5)
    m = build_cnn_da_bi_g3_v4().train()
    stft, sc, fc = _inputs()
    y = torch.rand(2)
    arr = m(stft, sc=sc, fc=fc, return_arrays=True)
    pw = torch.tensor([1.0])
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    loss = crit(arr["logit"], y) + 0.15 * 0.5 * (crit(arr["z_f"], y) + crit(arr["z_b"], y))
    loss.backward()
    bad = [n for n, p in m.named_parameters()
           if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    assert bad == [], "non-finite/missing grads: %s" % bad

def test_optimizer_each_trainable_param_exactly_one_group():
    m = build_cnn_da_bi_g3_v4()
    opt, oc = build_optimizer_v4(m)
    assert len(opt.param_groups) == 4
    from collections import Counter
    seen = Counter()
    for grp in opt.param_groups:
        for p in grp["params"]:
            seen[id(p)] += 1
    n_trainable = sum(1 for p in m.parameters() if p.requires_grad)
    assert sum(1 for c in seen.values() if c != 1) == 0, dict(seen)
    assert len(seen) == n_trainable
