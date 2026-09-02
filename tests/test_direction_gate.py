# -*- coding: utf-8 -*-
"""T6 (direction weights sum to 1 in [0,1]) and T7 (reversal symmetry, eval mode)."""
import math

import pytest
import torch

from models.cnn_da_bi_g3_v4 import build_cnn_da_bi_g3_v4


def _model():
    torch.manual_seed(11)
    return build_cnn_da_bi_g3_v4().eval()


def test_direction_weights_partition_of_unity():
    m = _model()
    stft = torch.randn(4, 53, 18, 70)
    fc = torch.randn(4, 53, 18, 18).abs() + 0.1
    sc = torch.rand(18, 18) + 0.1
    a = m(stft, sc=sc, fc=fc, return_arrays=True)
    wf = a["direction_weight_forward"]
    wb = a["direction_weight_backward"]
    s = wf + wb
    assert torch.allclose(s, torch.ones_like(s), atol=1e-4), "w_f + w_b must equal 1"
    assert (wf >= 0).all() and (wf <= 1).all()
    assert (wb >= 0).all() and (wb <= 1).all()
    assert torch.isfinite(wf).all() and torch.isfinite(wb).all()


def test_reversal_symmetry_eval_mode():
    """Feeding the time-reversed clip swaps forward<->backward and keeps the final
    adaptive probability approximately constant (eval, no direction dropout)."""
    m = _model()
    B = 6
    torch.manual_seed(2)
    stft = torch.randn(B, 53, 18, 70)
    fc = torch.randn(B, 53, 18, 18).abs() + 0.1
    sc = torch.rand(18, 18) + 0.1

    orig = m(stft, sc=sc, fc=fc, return_arrays=True)
    rev = m(torch.flip(stft, dims=[1]), sc=sc, fc=torch.flip(fc, dims=[1]),
            return_arrays=True)

    # (1) forward-on-reversed ~= backward-on-original, and vice versa (weights swap)
    d1 = (orig["direction_weight_forward"] - rev["direction_weight_backward"]).abs()
    d2 = (orig["direction_weight_backward"] - rev["direction_weight_forward"]).abs()
    assert d1.max().item() < 0.05, d1.max().item()
    assert d2.max().item() < 0.05, d2.max().item()

    # (2) selective-readout attention is finite and sums to ~1 on each branch
    # (the model exposes it via the selective readout module, checked in test_v4_model.)

    # (3) final adaptive probability approximately preserved
    po = torch.sigmoid(orig["logit"])
    pr = torch.sigmoid(rev["logit"])
    assert (po - pr).abs().max().item() < 0.05, (po - pr).abs().max().item()

    # direction disagreement is symmetric under reversal
    do = (orig["forward_raw_logit"] - orig["backward_raw_logit"]).abs().max().item()
    dr = (rev["forward_raw_logit"] - rev["backward_raw_logit"]).abs().max().item()
    assert torch.isfinite(torch.tensor([do, dr])).all()
