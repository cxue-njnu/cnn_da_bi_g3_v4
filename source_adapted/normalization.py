# -*- coding: utf-8 -*-
"""source_adapted/normalization.py - TRAIN-only per-fold (18,70) normalization.

PROTOCOL (identical to the G3Prime2 experiment / V2.2 protocol):
    statistics are aggregated over TRAIN clips x 53 time steps, and the
    18 channels x 70 frequencies structure is PRESERVED:

        mean.shape == (18, 70)
        std.shape  == (18, 70)

        sum_cf   += X.sum(axis=(0,1))
        sumsq_cf += (X*X).sum(axis=(0,1))
        count    += num_clips * 53

DEV and TEST clips NEVER contribute to the statistics. Accumulation is
incremental / chunked -- a huge float64 array is never materialised.
"""
import numpy as np

NUM_NODES = 18
FEATURE_DIM = 70
MODEL_STEPS = 53
STD_EPS = 1e-8


class ChannelFrequencyNormalizer(object):
    """Incremental TRAIN-only (18,70) mean/std accumulator using sum/sumsq/count."""

    def __init__(self, num_nodes=NUM_NODES, feature_dim=FEATURE_DIM):
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.sum_cf = np.zeros((num_nodes, feature_dim), dtype=np.float64)
        self.sumsq_cf = np.zeros((num_nodes, feature_dim), dtype=np.float64)
        self.count = 0

    def update(self, X):
        """Accumulate a chunk X of shape (K, 53, 18, 70) or (53, 18, 70)."""
        a = np.asarray(X, dtype=np.float64)
        if a.ndim == 3:
            a = a[None, ...]
        if a.ndim != 4 or a.shape[2:] != (self.num_nodes, self.feature_dim):
            raise ValueError("expected (K,T,%d,%d), got %s"
                             % (self.num_nodes, self.feature_dim, a.shape))
        # sum over clips (axis 0) and time (axis 1) -> keeps (channel, frequency)
        self.sum_cf += a.sum(axis=(0, 1), dtype=np.float64)
        self.sumsq_cf += (a * a).sum(axis=(0, 1), dtype=np.float64)
        self.count += int(a.shape[0] * a.shape[1])
        return self

    def finalize(self):
        if self.count == 0:
            raise ValueError("cannot fit a normalizer on an empty TRAIN split")
        mean = self.sum_cf / self.count
        var = np.clip(self.sumsq_cf / self.count - mean * mean, 0.0, None)
        std = np.sqrt(var) + STD_EPS
        return mean.astype(np.float32), std.astype(np.float32)


def apply_normalize(stft, mean, std):
    """Elementwise (x - mean)/std broadcasting (18,70) over (...,53,18,70)."""
    return ((np.asarray(stft, dtype=np.float32) - mean) / std).astype(np.float32)


def normalizer_payload(mean, std, fold, n_train_clips):
    """JSON-serialisable provenance record for the fitted normalizer."""
    return {
        "source": "TRAIN_ONLY",
        "scheme": "TRAIN_ONLY_channel_frequency_18x70",
        "fold": fold,
        "n_train_clips": int(n_train_clips),
        "mean_shape": list(mean.shape),
        "std_shape": list(std.shape),
    }
