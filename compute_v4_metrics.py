# -*- coding: utf-8 -*-
"""compute_v4_metrics.py - fold/patient-level metrics for CNN_DA_BI_G3_V4.

Usage:
    python compute_v4_metrics.py --seed 42
    (optionally) --out <runs dir> --root <metrics dir>

Reads the seed-aware result JSONs written by run_cnn_da_bi_g3_v4.py and produces:
    metrics/seed<seed>/metrics_v4_folds.csv
    metrics/seed<seed>/metrics_v4_cases.csv
    metrics/seed<seed>/metrics_v4_models.csv
    metrics/seed<seed>/metrics_v4_directionality.csv
    metrics/seed<seed>/metrics_v4_summary.json

In V4 there is a single adaptive-bidirectional final score ("test_logits" = the final
sigmoid probability); forward/backward scores are the shared direction auxillary head
outputs ("test_forward_probs" / "test_backward_probs").
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, auc as sk_auc,
                             log_loss, confusion_matrix)

MODEL_NAME = "CNN_DA_BI_G3_V4"
THRESHOLD = 0.5


def safe_auc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def safe_ap(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    if len(np.unique(y)) < 2 or len(y) == 0:
        return float("nan")
    try:
        return float(average_precision_score(y, s))
    except Exception:
        return float("nan")


def safe_pr_auc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    if len(np.unique(y)) < 2 or len(y) == 0:
        return float("nan")
    try:
        p, r, _ = precision_recall_curve(y, s)
        return float(sk_auc(r, p))
    except Exception:
        return float("nan")


def safe_log_loss(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64).clip(1e-12, 1 - 1e-12)
    return float(log_loss(y, s, labels=[0, 1])) if len(np.unique(y)) == 2 else float("nan")


def sens_spec_confusion(y, s, t=THRESHOLD):
    y = np.asarray(y); pred = (np.asarray(s, dtype=float) >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / float(tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / float(tn + fp) if (tn + fp) > 0 else float("nan")
    return sens, spec, int(tp), int(fn), int(tn), int(fp)


def balanced_accuracy(y, s, t=THRESHOLD):
    sens, spec, *_ = sens_spec_confusion(y, s, t)
    if math.isnan(sens) or math.isnan(spec):
        return float("nan")
    return 0.5 * (sens + spec)


def fin(x):
    if x is None:
        return float("nan")
    if isinstance(x, float) and math.isnan(x):
        return float("nan")
    return x


def fold_row(result):
    y = result["test_labels"]
    s = result["test_logits"]
    sf = result["test_forward_probs"]
    sb = result["test_backward_probs"]
    wf = result.get("direction_weight_forward", [])
    wb = result.get("direction_weight_backward", [])
    disc = result.get("direction_disagreement", [])
    ent = result.get("selective_readout_entropy", [])
    n_pos = int(sum(1 for x in y if x == 1)); n_neg = int(len(y) - n_pos)
    row = {
        "seed": result.get("seed"), "model": MODEL_NAME, "fold": result["fold"],
        "case": str(result["test_case_ids"][0]) if result["test_case_ids"] else "",
        "n_test": len(y), "n_pos": n_pos, "n_neg": n_neg, "threshold": THRESHOLD,
        "roc_auc": safe_auc(y, s), "average_precision": safe_ap(y, s),
        "pr_auc": safe_pr_auc(y, s), "log_loss": safe_log_loss(y, s),
        "forward_auc": safe_auc(y, sf), "backward_auc": safe_auc(y, sb),
        "final_auc": safe_auc(y, s),
    }
    sens, spec, tp, fn_, tn, fp = sens_spec_confusion(y, s)
    row.update({
        "sensitivity": sens, "specificity": spec,
        "balanced_accuracy": balanced_accuracy(y, s),
        "tp": tp, "fn": fn_, "tn": tn, "fp": fp,
        "direction_weight_forward_mean": float(np.mean(wf)) if len(wf) else float("nan"),
        "direction_weight_forward_std": float(np.std(wf)) if len(wf) else float("nan"),
        "direction_weight_backward_mean": float(np.mean(wb)) if len(wb) else float("nan"),
        "direction_weight_backward_std": float(np.std(wb)) if len(wb) else float("nan"),
        "mean_direction_disagreement": float(np.mean(disc)) if len(disc) else float("nan"),
        "mean_readout_entropy": float(np.mean(ent)) if len(ent) else float("nan"),
        "best_dev": fin(result.get("best_dev")), "best_epoch": result.get("best_epoch"),
    })
    return row


def event_weighted_score(case_df, metric):
    total = float(case_df["n_test"].sum())
    return float((case_df[metric] * case_df["n_test"]).sum() / total) if total else float("nan")


def main():
    p = argparse.ArgumentParser(description="CNN_DA_BI_G3_V4 metrics")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="runs output root")
    p.add_argument("--root", default=None, help="metrics output dir")
    args = p.parse_args()
    seed = str(args.seed)

    workspace = os.path.dirname(os.path.abspath(__file__))
    if args.out:
        results_dir = os.path.join(args.out, "seed%s" % seed, "results")
    else:
        results_dir = os.path.join(workspace, "runs", "seed%s" % seed, "results")
    metrics_out = args.root or os.path.join(workspace, "metrics", "seed%s" % seed)
    os.makedirs(metrics_out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(results_dir, "%s_*.json" % MODEL_NAME)))
    files = [f for f in files if not f.endswith(".done.json")]
    done_map = {}
    for f in glob.glob(os.path.join(results_dir, "*.done.json")):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            if d.get("status") == "DONE":
                done_map[d.get("fold")] = True
        except Exception:
            pass

    folds = []; skipped = []
    for f in files:
        fold = os.path.basename(f)
        fold = fold.replace("%s_" % MODEL_NAME, "").replace(".json", "")
        if fold not in done_map:
            skipped.append(fold); continue
        with open(f, "r", encoding="utf-8") as fh:
            result = json.load(fh)
        if result.get("model") != MODEL_NAME:
            skipped.append(fold); continue
        folds.append(fold_row(result))

    if not folds:
        print("NO completed folds found under %s" % results_dir)
        raise SystemExit(1)

    df = pd.DataFrame(folds)

    case_rows = []
    for case, g in df.groupby("case"):
        case_rows.append({
            "seed": int(seed), "model": MODEL_NAME, "case": case,
            "n_folds": int(len(g)), "n_test": int(g["n_test"].sum()),
            "case_roc_auc": float(g["roc_auc"].mean()),
            "case_average_precision": float(g["average_precision"].mean()),
            "case_pr_auc": float(g["pr_auc"].mean()),
            "case_forward_auc": float(g["forward_auc"].mean()),
            "case_backward_auc": float(g["backward_auc"].mean()),
            "event_weighted_roc_auc": event_weighted_score(g, "roc_auc"),
            "case_mean_direction_disagreement": float(g["mean_direction_disagreement"].mean()),
            "case_mean_readout_entropy": float(g["mean_readout_entropy"].mean()),
        })
    cases = pd.DataFrame(case_rows)

    n_folds = int(len(df)); n_cases = int(len(cases))
    macro_auc = float(df.groupby("case")["roc_auc"].mean().mean())
    macro_fwd = float(df.groupby("case")["forward_auc"].mean().mean())
    macro_bwd = float(df.groupby("case")["backward_auc"].mean().mean())
    macro_ap = float(df.groupby("case")["average_precision"].mean().mean())
    macro_pr = float(df.groupby("case")["pr_auc"].mean().mean())
    macro_ba = float(df.groupby("case")["balanced_accuracy"].mean().mean())
    ev_auc = event_weighted_score(df, "roc_auc")

    mean_disc = float(df["mean_direction_disagreement"].mean())
    wf_mean = float(df["direction_weight_forward_mean"].mean())
    wb_mean = float(df["direction_weight_backward_mean"].mean())
    ent_mean = float(df["mean_readout_entropy"].mean())

    model_summary = {
        "seed": int(seed), "model": MODEL_NAME, "n_folds": n_folds, "n_cases": n_cases,
        "macro_patient_roc_auc": macro_auc,
        "macro_patient_forward_auc": macro_fwd,
        "macro_patient_backward_auc": macro_bwd,
        "macro_patient_bi_minus_forward_delta": float(macro_auc - macro_fwd),
        "macro_patient_bi_minus_backward_delta": float(macro_auc - macro_bwd),
        "macro_case_average_precision": macro_ap,
        "macro_case_pr_auc": macro_pr,
        "macro_case_balanced_accuracy": macro_ba,
        "event_weighted_roc_auc": ev_auc,
        "event_weighted_forward_roc_auc": event_weighted_score(df, "forward_auc"),
        "event_weighted_backward_roc_auc": event_weighted_score(df, "backward_auc"),
        "event_weighted_average_precision": event_weighted_score(df, "average_precision"),
        "event_weighted_pr_auc": event_weighted_score(df, "pr_auc"),
        "mean_direction_disagreement": mean_disc,
        "mean_direction_weight_forward": wf_mean,
        "mean_direction_weight_backward": wb_mean,
        "mean_selective_readout_entropy": ent_mean,
    }
    directionality_row = {
        "seed": int(seed), "model": MODEL_NAME, "n_folds": n_folds,
        "mean_delta_bi_minus_forward_auc": float(macro_auc - macro_fwd),
        "mean_delta_bi_minus_backward_auc": float(macro_auc - macro_bwd),
        "mean_delta_backward_minus_forward_auc": float(macro_bwd - macro_fwd),
        "mean_direction_disagreement": mean_disc,
        "direction_weight_forward_mean": wf_mean,
        "direction_weight_backward_mean": wb_mean,
        "folds_bi_beats_forward": int((df["roc_auc"] > df["forward_auc"]).sum()),
        "folds_bi_below_forward": int((df["roc_auc"] < df["forward_auc"]).sum()),
        "mean_selective_readout_entropy": ent_mean,
    }

    df.sort_values("fold").to_csv(os.path.join(metrics_out, "metrics_v4_folds.csv"), index=False)
    cases.sort_values("case").to_csv(os.path.join(metrics_out, "metrics_v4_cases.csv"), index=False)
    pd.DataFrame([model_summary]).to_csv(os.path.join(metrics_out, "metrics_v4_models.csv"), index=False)
    pd.DataFrame([directionality_row]).to_csv(
        os.path.join(metrics_out, "metrics_v4_directionality.csv"), index=False)

    summary = {
        "seed": int(seed), "results_dir": results_dir,
        "completed_result_files": len(folds), "completed_folds": n_folds,
        "skipped_files": skipped, "threshold": THRESHOLD,
        "FINAL_METRIC_STATUS": "COMPLETE" if not skipped else "PARTIAL",
        "models": {MODEL_NAME: model_summary},
        "directionality": directionality_row,
        "notes": {
            "metric_policy": "FAILED/skipped folds are EXCLUDED. Formal metric valid only at 117/117.",
            "test_logits": "historical name; stores final sigmoid probabilities",
        },
    }
    with open(os.path.join(metrics_out, "metrics_v4_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print("metrics written under %s : folds=%d cases=%d (skipped=%d)"
          % (metrics_out, n_folds, n_cases, len(skipped)))
    print("macro_patient_roc_auc=%.4f event_weighted_roc_auc=%.4f" % (macro_auc, ev_auc))


if __name__ == "__main__":
    main()
