# -*- coding: utf-8 -*-
"""source_adapted/run_cnn_da_bi_g3_v4.py - supervisor + CLI runner for CNN_DA_BI_G3_V4.

CLI (identical style to the CNN baseline):
    python source_adapted\run_cnn_da_bi_g3_v4.py --seed 42 --device cuda --batch 32 --parallel 1
    python source_adapted\run_cnn_da_bi_g3_v4.py --seed 42 --device cuda --batch 32 --parallel 1 \
        --fold chb01_event03
Args: --seed --device --batch --parallel --fold --root --out
Defaults: seed=42 device=cuda batch=32 parallel=1

Outputs (seed-aware): <out>/seed<seed>/{results,checkpoints,worker_logs}
    results/CNN_DA_BI_G3_V4_<fold>.json
    results/CNN_DA_BI_G3_V4_<fold>_predictions.csv
    results/CNN_DA_BI_G3_V4_<fold>.done.json
    checkpoints/CNN_DA_BI_G3_V4_<fold>.pt

RESUME: a fold is skipped only when checkpoint / result JSON / predictions CSV /
DONE JSON (status==DONE and matching model+fold) are all present, readable and
non-empty. Any missing or unreadable piece => partial => rerun.
"""
import argparse
import csv
import json
import os
import subprocess
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

DEFAULT_ROOT = r"E:\seizure pred\V2_2_FULL_PACKAGE"
DEFAULT_OUT = os.path.join(os.path.dirname(BASE), "runs")
MODEL_NAME = "CNN_DA_BI_G3_V4"
TAG = "[CNN_DA_BI_G3]"

MODEL_CONFIG = {
    "model_steps": 53,
    "nodes": 18,
    "feature_dim": 70,
    "encoder_hidden": 96,
    "fusion_dim": 128,
    "state_dim": 128,
    "context_rank": 32,
    "context_feat": 96,
    "layers": 2,
    "inter_layer_rms": True,
    "context_dropout": 0.10,
    "lambda_init": 0.10,
    "direction_dropout": 0.10,
    "enable_direction_lora": False,
    "direction_lora_rank": 8,
    "directions": 2,
    "weight_sharing": True,
    "fusion": "direction_adaptive_gate",
    "selective_readout": True,
    "direction_adaptive_gate": True,
    "direction_auxiliary_loss": True,
}

TRAIN_PARAMS = {
    "epochs": 80,
    "early_stop_patience": 10,
    "lr_group1": 3e-4,
    "lr_group2": 1e-3,
    "lr_group3": 3e-4,
    "lr_group4": 5e-4,
    "wd_group1": 5e-4,
    "wd_group2": 5e-4,
    "wd_group3": 1e-3,
    "wd_group4": 5e-4,
    "gamma_direction": 0.15,
    "direction_dropout": 0.10,
    "cnn_warmup_epochs": 3,
    "scheduler_patience": 4,
    "scheduler_factor": 0.5,
    "max_grad_norm": 5.0,
}


def result_paths(out_dir, seed, fold):
    res_dir = os.path.join(out_dir, "seed%s" % seed, "results")
    ck_dir = os.path.join(out_dir, "seed%s" % seed, "checkpoints")
    return {
        "results_dir": res_dir,
        "checkpoints_dir": ck_dir,
        "result": os.path.join(res_dir, "%s_%s.json" % (MODEL_NAME, fold)),
        "predictions": os.path.join(res_dir, "%s_%s_predictions.csv" % (MODEL_NAME, fold)),
        "done": os.path.join(res_dir, "%s_%s.done.json" % (MODEL_NAME, fold)),
        "checkpoint": os.path.join(ck_dir, "%s_%s.pt" % (MODEL_NAME, fold)),
    }


def _readable_nonempty(path):
    try:
        if not os.path.isfile(path) or os.path.getsize(path) <= 2:
            return False
        with open(path, "rb") as fh:
            return bool(fh.read(1))
    except Exception:
        return False


def is_done(out_dir, seed, fold):
    p = result_paths(out_dir, seed, fold)
    for key in ("checkpoint", "result", "predictions", "done"):
        if not _readable_nonempty(p[key]):
            return False
    try:
        with open(p["done"], "r", encoding="utf-8") as fh:
            marker = json.load(fh)
    except Exception:
        return False
    return (marker.get("status") == "DONE"
            and marker.get("model") == MODEL_NAME
            and str(marker.get("fold")) == str(fold))


def run_one(config, fold):
    import numpy as np
    from folds_v4 import load_fold, assert_fold_disjoint
    from dataset_v4 import (build_feature_lookup, fit_train_normalizer, preload_split,
                            load_static_sc, close_stores)
    from train_v4 import run_fold as _run_fold, seed_everything

    seed = int(config["seed"]); root = config["root"]
    out = config["out"]; device = config["device"]

    seed_everything(seed)
    train_rows, dev_rows, test_rows = load_fold(root, fold)
    if assert_fold_disjoint(train_rows, dev_rows, test_rows) != 0:
        raise RuntimeError("fold %s has TRAIN/DEV/TEST clip overlap" % fold)

    all_rows = train_rows + dev_rows + test_rows
    rows_by_id = {r["clip_id"]: r for r in all_rows}
    lookup = build_feature_lookup(root, set(rows_by_id), rows_by_id)

    stores = {}
    try:
        mean, std, stores = fit_train_normalizer(train_rows, lookup, stores=stores)
        data = {
            "train": preload_split(train_rows, lookup, mean, std, stores=stores),
            "dev": preload_split(dev_rows, lookup, mean, std, stores=stores),
            "test": preload_split(test_rows, lookup, mean, std, stores=stores),
            "normalizer": {"mean": mean, "std": std},
        }
        sc = load_static_sc(root)
        cfg = dict(config)
        cfg["model"] = MODEL_NAME
        cfg["model_config"] = MODEL_CONFIG
        res = _run_fold(cfg, fold, data, sc, device, out)
    finally:
        close_stores(stores)

    paths = result_paths(out, seed, fold)
    os.makedirs(paths["results_dir"], exist_ok=True)

    result = {
        "model": MODEL_NAME,
        "fold": fold,
        "seed": seed,
        "best_epoch": res["best_epoch"],
        "best_dev": res["best_dev"],
        "dev_auc_bidirectional": res["dev_auc_bidirectional"],
        "dev_auc_forward": res["dev_auc_forward"],
        "dev_auc_backward": res["dev_auc_backward"],
        "test_auc": res["test_auc"],
        "test_auc_forward": res["test_auc_forward"],
        "test_auc_backward": res["test_auc_backward"],
        "test_logits": res["test_probs"],       # historical name == final probs
        "test_raw_logits": res["test_raw_logits"],
        "test_forward_raw_logits": res["test_forward_raw_logits"],
        "test_backward_raw_logits": res["test_backward_raw_logits"],
        "test_forward_probs": res["test_forward_probs"],
        "test_backward_probs": res["test_backward_probs"],
        "direction_weight_forward": res["test_direction_weight_forward"],
        "direction_weight_backward": res["test_direction_weight_backward"],
        "direction_disagreement": res["test_direction_disagreement"],
        "selective_readout_entropy": res["test_selective_readout_entropy"],
        "selective_readout_peak_weight": res["test_selective_readout_peak"],
        "lambda_value": res["lambda_value"],
        "gate_mean": res["gate_mean"],
        "gate_p95": res["gate_p95"],
        "test_labels": res["test_labels"],
        "test_clip_ids": res["test_clip_ids"],
        "test_case_ids": res["test_case_ids"],
        "test_edf_files": res["test_edf_files"],
        "checkpoint": res["checkpoint"],
        "model_config": MODEL_CONFIG,
        "optimizer_config": res["optimizer_config"],
        "train_config": res["train_config"],
        "direction_config": res["direction_config"],
        "pos_weight": res["pos_weight"],
        "n_train": res["n_train"], "n_dev": res["n_dev"], "n_test": res["n_test"],
        "epochs_run": res["epochs_run"],
        "history": res["history"],
    }

    lengths = {len(result[k]) for k in (
        "test_logits", "test_raw_logits", "test_forward_probs", "test_backward_probs",
        "test_forward_raw_logits", "test_backward_raw_logits", "direction_disagreement",
        "direction_weight_forward", "direction_weight_backward",
        "selective_readout_entropy", "selective_readout_peak_weight",
        "test_labels", "test_clip_ids", "test_case_ids", "test_edf_files")}
    if len(lengths) != 1:
        raise RuntimeError("prediction/identity arrays have inconsistent lengths")

    with open(paths["result"], "w", encoding="utf-8") as fh:
        json.dump(result, fh)

    with open(paths["predictions"], "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["clip_id", "case_id", "edf_file", "y_true", "y_score", "raw_logit",
                    "y_score_forward", "y_score_backward",
                    "direction_weight_forward", "direction_weight_backward",
                    "direction_disagreement"])
        for i in range(result["n_test"]):
            w.writerow([
                result["test_clip_ids"][i], result["test_case_ids"][i],
                result["test_edf_files"][i], result["test_labels"][i],
                result["test_logits"][i], result["test_raw_logits"][i],
                result["test_forward_probs"][i], result["test_backward_probs"][i],
                result["direction_weight_forward"][i],
                result["direction_weight_backward"][i],
                result["direction_disagreement"][i],
            ])

    done = {"status": "DONE", "model": MODEL_NAME, "fold": fold, "seed": seed}
    if not all(_readable_nonempty(paths[k]) for k in ("checkpoint", "result", "predictions")):
        raise RuntimeError("fold %s produced incomplete artefacts" % fold)
    with open(paths["done"], "w", encoding="utf-8") as fh:
        json.dump(done, fh)

    os.makedirs(os.path.join(out, "seed%s" % seed, "worker_logs"), exist_ok=True)
    with open(os.path.join(out, "seed%s" % seed, "worker_logs", "%s_%s.log" % (MODEL_NAME, fold)),
              "w", encoding="utf-8") as fh:
        fh.write("[DONE] fold=%s test_auc=%s forward_auc=%s backward_auc=%s "
                 "best_dev=%s best_epoch=%s epochs_run=%s\n"
                 % (fold, result["test_auc"], result["test_auc_forward"],
                    result["test_auc_backward"], result["best_dev"],
                    result["best_epoch"], result["epochs_run"]))
    return result


def _fmt(x):
    return "%.4f" % x if isinstance(x, float) else str(x)


def worker_entry(config, fold):
    try:
        r = run_one(config, fold)
        print("%s %s DONE test_auc=%s forward_auc=%s backward_auc=%s best_dev=%s"
              % (TAG, fold, _fmt(r["test_auc"]), _fmt(r["test_auc_forward"]),
                 _fmt(r["test_auc_backward"]), _fmt(r["best_dev"])), flush=True)
        return 0
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("%s %s ERROR %s" % (TAG, fold, e), flush=True)
        return 1


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="run_cnn_da_bi_g3_v4.py",
        description="CNN_DA_BI_G3_V4: CNN-encoded direction-adaptive bidirectional "
                    "connectivity-conditioned selective state model over the 117-fold "
                    "V2.2 patient-specific event-LOSO protocol.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda", help="cuda|cpu")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument("--fold", default=None)
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--cnn-warmup-epochs", type=int, default=None,
                   help="override the two-stage CNN warm-up epochs (default 3)")
    args = p.parse_args(argv)

    from folds_v4 import enumerate_folds
    config = {
        "seed": args.seed, "device": args.device, "batch": args.batch,
        "parallel": args.parallel, "root": args.root, "out": args.out,
        "model": MODEL_NAME, "model_config": MODEL_CONFIG,
    }
    for k, v in TRAIN_PARAMS.items():
        config[k] = v
    if args.cnn_warmup_epochs is not None:
        config["cnn_warmup_epochs"] = args.cnn_warmup_epochs

    if args.fold:
        folds = [args.fold]
    else:
        folds = enumerate_folds(args.root)
    total = len(folds)
    pending = [f for f in folds if not is_done(args.out, args.seed, f)]
    print("%s pending folds=%d/%d" % (TAG, len(pending), total))

    workers = max(1, int(args.parallel))
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    procs = []
    for fold in pending:
        if len(procs) >= workers:
            procs[0].join()
            procs.pop(0)
        proc = ctx.Process(target=worker_entry, args=(config, fold))
        proc.start()
        procs.append(proc)
        print("%s fold %s started" % (TAG, fold))
    for proc in procs:
        proc.join()
    print("%s finished pending=%d/%d" % (TAG, len(pending), total))


if __name__ == "__main__":
    main()
