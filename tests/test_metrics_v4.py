# -*- coding: utf-8 -*-
"""T9 (TRAIN-only normalizer) and T12 (AUC / AP / PR-AUC validity)."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- T9 normalizer -----------------------------------------------------------
from normalization import ChannelFrequencyNormalizer  # noqa: E402


def test_train_only_normalizer_shape_and_stats():
    rng = np.random.RandomState(0)
    train = rng.randn(5, 53, 18, 70).astype(np.float32)
    acc = ChannelFrequencyNormalizer()
    acc.update(train)
    mean, std = acc.finalize()
    assert mean.shape == (18, 70)
    assert std.shape == (18, 70)
    # independent manual check of the exact same formula
    a = train.astype(np.float64)
    exp_mean = a.sum(axis=(0, 1)) / (5 * 53)
    exp_var = (a * a).sum(axis=(0, 1)) / (5 * 53) - exp_mean ** 2
    exp_std = np.sqrt(np.clip(exp_var, 0, None)) + 1e-8
    assert np.allclose(mean, exp_mean, atol=1e-6)
    assert np.allclose(std, exp_std, atol=1e-6)


def test_normalizer_never_sees_dev_test():
    # DEV/TEST tensors are not passed to update/finalize by construction; emulate a
    # leak: if a dev row were included the stats would change.
    rng = np.random.RandomState(1)
    train = rng.randn(3, 53, 18, 70).astype(np.float32)
    dev = (rng.randn(3, 53, 18, 70) * 100 + 500).astype(np.float32)
    a = ChannelFrequencyNormalizer(); a.update(train)
    m1, s1 = a.finalize()
    b = ChannelFrequencyNormalizer(); b.update(train); b.update(dev)
    m2, s2 = b.finalize()
    assert not np.allclose(m1, m2), "normalizer must be TRAIN-only"
    assert not np.allclose(s1, s2)


# --- T12 metrics validity ----------------------------------------------------
from compute_v4_metrics import safe_auc, safe_ap, safe_pr_auc, safe_log_loss,     balanced_accuracy, sens_spec_confusion  # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score,     precision_recall_curve, auc as sk_auc, log_loss  # noqa: E402


def test_auc_ap_pr_match_sklearn():
    y = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
    s = np.array([0.11, 0.92, 0.55, 0.08, 0.73, 0.20, 0.05, 0.88, 0.61, 0.30])
    assert safe_auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-9)
    assert safe_ap(y, s) == pytest.approx(average_precision_score(y, s), abs=1e-9)
    p, r, _ = precision_recall_curve(y, s)
    assert safe_pr_auc(y, s) == pytest.approx(sk_auc(r, p), abs=1e-9)
    assert safe_log_loss(y, s) == pytest.approx(log_loss(y, s), abs=1e-9)


def test_metrics_ideal_and_degenerate():
    # perfect separation
    y = np.array([1, 1, 0, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.1, 0.2, 0.99, 0.15])
    assert safe_auc(y, s) == pytest.approx(1.0, abs=1e-6)
    assert safe_ap(y, s) == pytest.approx(1.0, abs=1e-6)
    # single-class returns nan (guarded), never crashes
    assert np.isnan(safe_auc(np.array([1, 1, 1]), np.array([0.1, 0.5, 0.9])))


def test_confusion_balanced_accuracy():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.9, 0.6, 0.4])
    sens, spec, tp, fn, tn, fp = sens_spec_confusion(y, s, 0.5)
    assert sens == 0.5 and spec == 0.5
    bal = balanced_accuracy(y, s, 0.5)
    assert 0.0 <= bal <= 1.0
    assert tp == 1 and fn == 1 and tn == 1 and fp == 1
