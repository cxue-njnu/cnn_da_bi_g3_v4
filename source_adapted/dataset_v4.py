# -*- coding: utf-8 -*-
"""source_adapted/dataset_v4.py

V2.2 feature access for CNN_DA_BI_G3_V4.

FASTFIX:
- preserves TRAIN-only (18,70) normalization;
- keeps HDF5 files cached;
- preloads each split grouped by feature store and in bounded sorted chunks instead
  of one HDF5 read per clip;
- preserves exact fold order and metadata alignment.
"""

import csv
import os

import numpy as np
import torch

from normalization import (
    ChannelFrequencyNormalizer,
    MODEL_STEPS,
    NUM_NODES,
    FEATURE_DIM,
)

V2_1_FEATURE_SUBDIR = ("features",)
V2_2_FEATURE_SUBDIR = (
    "v2_2_literature_aligned",
    "incremental_features",
)
FEATURE_INDEX_SUBPATH = (
    "v2_2_literature_aligned",
    "feature_index",
    "feature_index_v2_2.csv",
)


def clip_basename(edf_file):
    return os.path.splitext(os.path.basename(str(edf_file)))[0]


def feature_index_path(root):
    return os.path.join(root, *FEATURE_INDEX_SUBPATH)


def candidate_store_paths(root, case_id, edf_file, source=None):
    base = clip_basename(edf_file)
    v21 = os.path.join(
        root, *V2_1_FEATURE_SUBDIR, str(case_id), base + ".h5"
    )
    v22 = os.path.join(
        root, *V2_2_FEATURE_SUBDIR, str(case_id), base + ".h5"
    )
    if str(source).upper() == "V2_2_INCREMENTAL":
        return [v22, v21]
    return [v21, v22]


def build_feature_lookup(root, want_clip_ids, rows_by_id=None):
    want = set(str(c) for c in want_clip_ids)
    lookup = {}
    idx_path = feature_index_path(root)

    if os.path.exists(idx_path):
        with open(idx_path, "r", newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                cid = str(row["clip_id"])
                if cid not in want or cid in lookup:
                    continue
                src = row.get("feature_source", "V2_1_REUSED")
                for cand in candidate_store_paths(
                    root, row["case_id"], row["edf_file"], src
                ):
                    if os.path.exists(cand):
                        lookup[cid] = (
                            cand,
                            int(float(row["feature_row"])),
                        )
                        break

    missing = want - set(lookup)
    if missing and rows_by_id:
        for cid in sorted(missing):
            r = rows_by_id.get(cid)
            if not r:
                continue
            for cand in candidate_store_paths(
                root, r["case_id"], r["edf_file"]
            ):
                if os.path.exists(cand):
                    ri = _store_index(cand).get(cid)
                    if ri is not None:
                        lookup[cid] = (cand, int(ri))
                        break
    return lookup


_INDEX_CACHE = {}


def _store_index(store_path):
    import h5py

    if store_path not in _INDEX_CACHE:
        with h5py.File(store_path, "r") as f:
            _INDEX_CACHE[store_path] = {
                (x.decode() if isinstance(x, bytes) else str(x)): i
                for i, x in enumerate(f["clip_id"][:])
            }
    return _INDEX_CACHE[store_path]


def _open_cached(cache, path):
    import h5py

    if path not in cache:
        cache[path] = h5py.File(path, "r")
    return cache[path]


def close_stores(cache):
    for f in list(cache.values()):
        try:
            f.close()
        except Exception:
            pass
    cache.clear()


def _resolve(lookup, record):
    cid = str(record["clip_id"])
    sp, row = lookup.get(cid, (None, None))
    if sp is None or not os.path.exists(sp):
        raise FileNotFoundError("feature store missing for clip %s" % cid)
    return cid, sp, int(row)


def fit_train_normalizer(train_rows, lookup, stores=None, chunk_size=64):
    """TRAIN-ONLY (18,70) normalizer accumulated in bounded HDF5 chunks."""
    cache = stores if stores is not None else {}
    acc = ChannelFrequencyNormalizer()

    by_store = {}
    for r in train_rows:
        _cid, sp, ri = _resolve(lookup, r)
        by_store.setdefault(sp, []).append(ri)

    for sp, rows in by_store.items():
        f = _open_cached(cache, sp)
        ordered = sorted(set(rows))
        for start in range(0, len(ordered), int(chunk_size)):
            batch = ordered[start:start + int(chunk_size)]
            X = f["stft"][batch]
            if X.shape[1:] != (
                MODEL_STEPS,
                NUM_NODES,
                FEATURE_DIM,
            ):
                raise RuntimeError(
                    "unexpected STFT shape %s in %s" % (X.shape, sp)
                )
            acc.update(X)

    mean, std = acc.finalize()
    return mean, std, cache


def preload_split(records, lookup, mean, std, stores=None, chunk_size=64):
    """Preload one split with store-grouped bounded HDF5 reads.

    Output order remains exactly the original `records` order.
    """
    cache = stores if stores is not None else {}
    n = len(records)
    if n == 0:
        raise ValueError("cannot preload an empty split")

    stft = np.empty(
        (n, MODEL_STEPS, NUM_NODES, FEATURE_DIM),
        dtype=np.float32,
    )
    fc = np.empty(
        (n, MODEL_STEPS, NUM_NODES, NUM_NODES),
        dtype=np.float32,
    )
    labels = np.empty(n, dtype=np.float32)

    clip_ids = [None] * n
    case_ids = [None] * n
    edf_files = [None] * n

    # Resolve once and group by HDF5 store.
    by_store = {}
    for out_i, r in enumerate(records):
        cid, sp, ri = _resolve(lookup, r)
        by_store.setdefault(sp, []).append((out_i, ri, cid, r))

    for sp, items in by_store.items():
        f = _open_cached(cache, sp)

        # Multiple records should not normally reference the same row, but this
        # mapping safely supports it while satisfying h5py's sorted-index rule.
        row_to_items = {}
        for item in items:
            row_to_items.setdefault(item[1], []).append(item)

        unique_rows = sorted(row_to_items)

        for start in range(0, len(unique_rows), int(chunk_size)):
            idxs = unique_rows[start:start + int(chunk_size)]

            s_batch = np.asarray(
                f["stft"][idxs],
                dtype=np.float32,
            )
            c_batch = np.asarray(
                f["fc"][idxs],
                dtype=np.float32,
            )
            y_batch = np.asarray(
                f["label"][idxs],
                dtype=np.int64,
            ).reshape(-1)

            if s_batch.shape[1:] != (
                MODEL_STEPS,
                NUM_NODES,
                FEATURE_DIM,
            ):
                raise RuntimeError(
                    "unexpected STFT shape %s in %s"
                    % (s_batch.shape, sp)
                )
            if c_batch.shape[1:] != (
                MODEL_STEPS,
                NUM_NODES,
                NUM_NODES,
            ):
                raise RuntimeError(
                    "unexpected FC shape %s in %s"
                    % (c_batch.shape, sp)
                )

            for local_i, ri in enumerate(idxs):
                s = s_batch[local_i]
                c = c_batch[local_i]
                store_label = int(y_batch[local_i])

                for out_i, _ri, cid, r in row_to_items[ri]:
                    fold_label = int(r["label"])
                    if store_label != fold_label:
                        raise RuntimeError(
                            "label mismatch for clip %s: fold=%d store=%d"
                            % (cid, fold_label, store_label)
                        )

                    stft[out_i] = (s - mean) / std
                    fc[out_i] = c
                    labels[out_i] = float(store_label)
                    clip_ids[out_i] = cid
                    case_ids[out_i] = str(r["case_id"])
                    edf_files[out_i] = str(r["edf_file"])

    if any(x is None for x in clip_ids):
        raise RuntimeError("preload_split failed to fill all output rows")

    meta = {
        "clip_ids": clip_ids,
        "case_ids": case_ids,
        "edf_files": edf_files,
    }

    if not (
        n
        == len(clip_ids)
        == len(case_ids)
        == len(edf_files)
    ):
        raise RuntimeError("metadata length mismatch after preload")

    return (
        torch.from_numpy(stft),
        torch.from_numpy(fc),
        torch.from_numpy(labels),
        meta,
    )


CANONICAL_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]
SC_DISTANCE_THRESHOLD = 0.9


def load_static_sc(root):
    """Load / derive static SC (18,18), read-only."""
    npz = os.path.join(
        root,
        "data",
        "static_graph",
        "SC_V2_SPATIAL_AFFINITY.npz",
    )
    if os.path.exists(npz):
        z = np.load(npz, allow_pickle=True)
        return z["affinity_matrix"].astype(np.float32)

    D = _load_distance_matrix(root)
    sigma = float(np.std(D[np.isfinite(D)]))
    SC = np.exp(-(D / sigma) ** 2)
    SC[D > SC_DISTANCE_THRESHOLD] = 0.0
    np.fill_diagonal(SC, 1.0)
    return SC.astype(np.float32)


def _load_distance_matrix(root):
    import pickle

    pkl = os.path.join(
        root,
        "reference",
        "legacy",
        "GLOBAL_MAT.pkl",
    )
    csv_path = os.path.join(
        root,
        "reference",
        "legacy",
        "my_distances_3d.csv",
    )

    if os.path.exists(pkl):
        with open(pkl, "rb") as fh:
            _chlist, _smap, M = pickle.load(fh)
        return np.asarray(M, dtype=np.float64)

    if os.path.exists(csv_path):
        idx = {ch: i for i, ch in enumerate(CANONICAL_CHANNELS)}
        D = np.zeros((18, 18), dtype=np.float64)
        with open(csv_path, "r", newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                a, b = row["from"], row["to"]
                if a in idx and b in idx:
                    D[idx[a], idx[b]] = float(row["distance"])
        return D + D.T - np.diag(np.diag(D))

    raise FileNotFoundError("no SC distance source found under " + root)
