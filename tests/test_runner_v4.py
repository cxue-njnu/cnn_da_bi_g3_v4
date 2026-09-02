# -*- coding: utf-8 -*-
"""T10 (end-to-end synthetic fold through real dataset+train+serialize pipeline) and
T11 (DONE-marker resume / partial rerun).

NOTE on the DSH file sandbox: writing files is only permitted under workspace
directories that already exist. The tests therefore build synthetic V2.2 roots under
PRE-AUTHORIZED <workspace>/runtest_temp/slot{1,2} directories instead of pytest's
system tmp_path.
"""
import csv
import json
import os
import shutil
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(HERE)
SRC = os.path.join(WORKSPACE, "source_adapted")

SLOT1 = os.path.join(WORKSPACE, "runtest_temp", "slot1")
SLOT2 = os.path.join(WORKSPACE, "runtest_temp", "slot2")


def _make_store_h5(path):
    rng = np.random.RandomState(0)
    stft = rng.randn(1, 53, 18, 70).astype(np.float32)
    fc = (np.abs(rng.randn(1, 53, 18, 18)) + 0.1).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("stft", data=stft)
        f.create_dataset("fc", data=fc)


def _wipe(d):
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)


def build_synth_root(slot):
    """Build a tiny self-contained V2.2-compatible tree in an authorized slot dir."""
    _wipe(slot)
    records = []
    for case in ("chb01", "chb02"):
        for i in range(14):
            cid = "%s_syn%02d" % (case, i)
            edf = "%s_syn%02d.edf" % (case, i)
            label = 1 if (i % 2 == 0) else 0
            records.append(dict(clip_id=cid, case_id=case, edf_file=edf, label=label))
    for rec in records:
        base = os.path.splitext(os.path.basename(rec["edf_file"]))[0]
        store_dir = os.path.join(slot, "features", rec["case_id"])
        os.makedirs(store_dir, exist_ok=True)
        p = os.path.join(store_dir, base + ".h5")
        _make_store_h5(p)
        with h5py.File(p, "r+") as f:
            f.create_dataset("clip_id", data=np.array([rec["clip_id"]], dtype=object))
            f.create_dataset("label", data=np.array([rec["label"]], dtype=np.int64))
    idx_dir = os.path.join(slot, "v2_2_literature_aligned", "feature_index")
    os.makedirs(idx_dir, exist_ok=True)
    with open(os.path.join(idx_dir, "feature_index_v2_2.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip_id", "case_id", "edf_file", "feature_row", "feature_source"])
        for rec in records:
            w.writerow([rec["clip_id"], rec["case_id"], rec["edf_file"], 0, "V2_1_REUSED"])
    sc_dir = os.path.join(slot, "data", "static_graph")
    os.makedirs(sc_dir, exist_ok=True)
    sc = (np.abs(np.random.RandomState(3).randn(18, 18)).astype(np.float32) + 0.2)
    np.savez(os.path.join(sc_dir, "SC_V2_SPATIAL_AFFINITY.npz"), affinity_matrix=sc)
    by_id = {r["clip_id"]: r for r in records}
    tr_ids = [r["clip_id"] for r in records
              if r["clip_id"] not in ("chb01_syn00", "chb01_syn01", "chb02_syn01")]
    dev_ids = ["chb01_syn01", "chb02_syn01"]
    te_ids = ["chb01_syn00"]
    fold = "chb01_event03"
    fdir = os.path.join(slot, "v2_2_literature_aligned", "folds",
                        "patient_event_loso", fold)
    os.makedirs(fdir, exist_ok=True)
    for name, ids in (("train", tr_ids), ("dev", dev_ids), ("test", te_ids)):
        with open(os.path.join(fdir, name + ".csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["clip_id", "case_id", "edf_file", "label"])
            for cid in ids:
                r = by_id[cid]
                w.writerow([r["clip_id"], r["case_id"], r["edf_file"], r["label"]])
    return slot, fold, len(te_ids)


def _config(root, seed=42):
    return {"seed": seed, "device": "cpu", "batch": 8, "parallel": 1, "root": root,
            "out": os.path.join(root, "runs"), "epochs": 1, "early_stop_patience": 3,
            "cnn_warmup_epochs": 0, "lr_group1": 3e-4, "lr_group2": 1e-3,
            "lr_group3": 3e-4, "lr_group4": 5e-4, "wd_group1": 5e-4, "wd_group2": 5e-4,
            "wd_group3": 1e-3, "wd_group4": 5e-4, "gamma_direction": 0.15,
            "direction_dropout": 0.10, "scheduler_patience": 2, "scheduler_factor": 0.5,
            "max_grad_norm": 5.0}


def test_end_to_end_synthetic_fold():
    for p in (SLOT1, SRC, WORKSPACE):
        if p not in sys.path:
            sys.path.insert(0, p)
    from run_cnn_da_bi_g3_v4 import run_one, result_paths, is_done
    root, fold, n_test = build_synth_root(SLOT1)
    cfg = _config(root)
    res = run_one(cfg, fold)
    assert res["n_test"] == n_test
    assert is_done(cfg["out"], 42, fold) is True
    paths = result_paths(os.path.join(root, "runs") if False else cfg["out"], 42, fold)
    assert paths is not None
    for key in ("result", "predictions", "done", "checkpoint"):
        assert os.path.exists(paths[key]) and os.path.getsize(paths[key]) > 0, key
    with open(paths["result"], "r", encoding="utf-8") as fh:
        data = json.load(fh)
    fields = ("best_epoch", "best_dev", "test_auc", "test_logits", "test_raw_logits",
              "test_labels", "test_clip_ids", "test_case_ids", "test_edf_files",
              "test_forward_raw_logits", "test_backward_raw_logits",
              "direction_weight_forward", "direction_weight_backward",
              "direction_disagreement", "selective_readout_entropy",
              "selective_readout_peak_weight", "lambda_value", "gate_mean",
              "gate_p95", "checkpoint", "optimizer_config", "model_config")
    for field in fields:
        assert field in data, field
    assert len(data["test_labels"]) == n_test
    # DONE marker
    with open(paths["done"], "r", encoding="utf-8") as fh:
        marker = json.load(fh)
    assert marker.get("status") == "DONE"


def test_resume_skip_vs_partial():
    for p in (SLOT2, SRC, WORKSPACE):
        if p not in sys.path:
            sys.path.insert(0, p)
    from run_cnn_da_bi_g3_v4 import run_one, result_paths, is_done
    root, fold, n_test = build_synth_root(SLOT2)
    cfg = _config(root)
    run_one(cfg, fold)
    paths = result_paths(cfg["out"], 42, fold)
    assert is_done(cfg["out"], 42, fold) is True
    os.remove(paths["predictions"])
    assert is_done(cfg["out"], 42, fold) is False   # partial => rerun
