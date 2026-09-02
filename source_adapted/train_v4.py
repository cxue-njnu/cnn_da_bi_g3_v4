# -*- coding: utf-8 -*-
"""source_adapted/train_v4.py - per-fold training / evaluation for CNN_DA_BI_G3_V4.

PROTOCOL
  optimizer         AdamW, four parameter groups (see models/cnn_da_bi_g3_v4.py)
  scheduler         ReduceLROnPlateau(mode=max, patience=4, factor=0.5) on DEV AUC
  epochs            80       early_stop_patience    10   (monitor = DEV ROC-AUC)
  max_grad_norm     5.0
  loss              L = L_main + gamma_direction * L_dir
                    L_main = BCEWithLogitsLoss(logit_final, y, pos_weight)
                    L_dir  = 0.5 * (BCE(z_f,y) + BCE(z_b,y))
                    gamma_direction default 0.15
  class balance     pos_weight ONLY (N_train_neg / N_train_pos); NO sampler
  TRAIN loader      shuffle=True, num_workers=0
  normalization     TRAIN-only (18,70)
  checkpoint        selected on DEV AUC ONLY; TEST is run ONCE at the end
  TEST is never used for early stopping / thresholds / LR / arch / checkpoint.

DIRECTION-DROPOUT (section 17)
  During TRAIN: with P=0.10 forward-only, P=0.10 backward-only, P=0.80 normal
  adaptive bidirectional. During DEV/TEST the model is in eval() so BOTH directions
  + the adaptive direction gate always run (no randomness).
"""
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.cnn_da_bi_g3_v4 import build_cnn_da_bi_g3_v4, build_optimizer_v4

DEFAULT_TRAIN_CONFIG = {
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
    "scheduler": "ReduceLROnPlateau",
    "scheduler_patience": 4,
    "scheduler_factor": 0.5,
    "max_grad_norm": 5.0,
    "monitor": "dev_auc",
    "loss": "BCEWithLogitsLoss_with_directional_aux",
    "class_balance": "TRAIN_ONLY_pos_weight",
}


def seed_everything(seed):
    import random
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def roc_auc(labels, scores):
    y = np.asarray(labels).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    u = ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def make_train_loader(stft, fc, labels, batch_size, seed):
    g = torch.Generator()
    g.manual_seed(int(seed))
    return DataLoader(TensorDataset(stft, fc, labels), batch_size=int(batch_size),
                      shuffle=True, num_workers=0, generator=g, drop_last=False)


def make_eval_loader(stft, fc, labels, batch_size):
    return DataLoader(TensorDataset(stft, fc, labels), batch_size=int(batch_size),
                      shuffle=False, num_workers=0, drop_last=False)


def compute_pos_weight(labels):
    y = np.asarray(labels).reshape(-1)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return (float(n_neg) / float(max(n_pos, 1))) if n_pos > 0 else 1.0


def apply_warmup(optimizer, epoch, warmup_epochs, scale=0.3):
    """Two-stage CNN warm-up (section 22). For epoch <= warmup_epochs, keep the G3
    base + connectivity residual groups at a LOWER learning rate; the CNN/fusion and
    readout/direction groups run at full LR. Afterwards restore full LR."""
    idx = getattr(optimizer, "v4_group_index", {})
    full = getattr(optimizer, "v4_full_lr", None)
    if full is None:
        return
    low = set([idx.get("g2"), idx.get("g3")])
    in_warmup = warmup_epochs > 0 and epoch <= int(warmup_epochs)
    for gi, group in enumerate(optimizer.param_groups):
        if gi in low and gi is not None:
            group["lr"] = full[gi] * scale if in_warmup else full[gi]


@torch.inference_mode()
def predict_full(model, loader, sc, device):
    """Evaluate a split in eval() mode (both directions + adaptive gate).

    Returns dict of numpy arrays aligned with loader order (per test clip):
        probs / raw_logits, forward_probs / backward_probs,
        forward_raw_logits / backward_raw_logits,
        direction_weight_forward / direction_weight_backward,
        direction_disagreement, readout_entropy, readout_peak_weight, labels
    plus model-global scalars lambda_value / gate_mean / gate_p95 (batch means).
    """
    model.eval()
    keys = ("prob", "raw", "f_prob", "b_prob", "f_raw", "b_raw",
            "w_f", "w_b", "entropy", "peak", "y")
    out = {k: [] for k in keys}
    scalars = []
    for stft, fc, labels in loader:
        arr = model(stft.to(device), sc=sc, fc=fc.to(device), return_arrays=True)
        out["raw"].append(arr["logit"].detach().float().cpu())
        out["f_raw"].append(arr["forward_raw_logit"].detach().float().cpu())
        out["b_raw"].append(arr["backward_raw_logit"].detach().float().cpu())
        out["prob"].append(torch.sigmoid(arr["logit"]).detach().float().cpu())
        out["f_prob"].append(torch.sigmoid(arr["forward_raw_logit"]).detach().float().cpu())
        out["b_prob"].append(torch.sigmoid(arr["backward_raw_logit"]).detach().float().cpu())
        out["w_f"].append(arr["direction_weight_forward"].detach().float().cpu())
        out["w_b"].append(arr["direction_weight_backward"].detach().float().cpu())
        out["entropy"].append(arr["readout_entropy"].detach().float().cpu())
        out["peak"].append(arr["readout_peak_weight"].detach().float().cpu())
        out["y"].append(labels.detach().float().cpu())
        if arr.get("gate_mean") is not None:
            scalars.append([float(arr["lambda_value"]),
                            float(arr["gate_mean"]), float(arr["gate_p95"] or 0.0)])
    cat = {k: (torch.cat(v).numpy() if v else np.array([])) for k, v in out.items()}
    if scalars:
        s = np.asarray(scalars, dtype=np.float64)
        lam, gmean, gp95 = float(s[:, 0].mean()), float(s[:, 1].mean()), float(s[:, 2].mean())
    else:
        lam = gmean = gp95 = None
    return {
        "probs": cat["prob"], "raw_logits": cat["raw"],
        "forward_probs": cat["f_prob"], "backward_probs": cat["b_prob"],
        "forward_raw_logits": cat["f_raw"], "backward_raw_logits": cat["b_raw"],
        "direction_weight_forward": cat["w_f"], "direction_weight_backward": cat["w_b"],
        "direction_disagreement": np.abs(cat["f_prob"] - cat["b_prob"]),
        "readout_entropy": cat["entropy"], "readout_peak_weight": cat["peak"],
        "labels": cat["y"],
        "lambda_value": lam, "gate_mean": gmean, "gate_p95": gp95,
    }


def run_fold(cfg, fold, data, sc_np, device, out_dir):
    seed = int(cfg["seed"])
    tcfg = dict(DEFAULT_TRAIN_CONFIG)
    tcfg.update({k: cfg[k] for k in DEFAULT_TRAIN_CONFIG if k in cfg})
    dev_t = torch.device(device)

    seed_everything(seed)
    mc = dict(cfg.get("model_config", {}))
    mc["direction_dropout"] = float(tcfg.get("direction_dropout", 0.10))
    model = build_cnn_da_bi_g3_v4(mc).to(dev_t)

    lr_map = {"g1": float(tcfg["lr_group1"]), "g2": float(tcfg["lr_group2"]),
              "g3": float(tcfg["lr_group3"]), "g4": float(tcfg["lr_group4"])}
    wd_map = {"g1": float(tcfg["wd_group1"]), "g2": float(tcfg["wd_group2"]),
              "g3": float(tcfg["wd_group3"]), "g4": float(tcfg["wd_group4"])}
    optimizer, opt_config = build_optimizer_v4(
        model, lr_map=lr_map, wd_map=wd_map,
        enable_direction_lora=bool(mc.get("enable_direction_lora", False)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=int(tcfg["scheduler_patience"]),
        factor=float(tcfg["scheduler_factor"]))

    tr_stft, tr_fc, tr_y, _tr_meta = data["train"]
    dv_stft, dv_fc, dv_y, _dv_meta = data["dev"]
    te_stft, te_fc, te_y, te_meta = data["test"]

    pos_weight = compute_pos_weight(tr_y.numpy())
    pw_t = torch.tensor([pos_weight], dtype=torch.float32, device=dev_t)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw_t)
    gamma_dir = float(tcfg.get("gamma_direction", 0.15))

    batch = int(cfg.get("batch", 32))
    tr_loader = make_train_loader(tr_stft, tr_fc, tr_y, batch, seed)
    dv_loader = make_eval_loader(dv_stft, dv_fc, dv_y, batch)
    te_loader = make_eval_loader(te_stft, te_fc, te_y, batch)
    sc = torch.from_numpy(np.asarray(sc_np, dtype=np.float32)).to(dev_t)

    best_dev = -1.0
    best_epoch = -1
    best_state = None
    best_dev_directional = None
    patience = 0
    history = []

    for epoch in range(1, int(tcfg["epochs"]) + 1):
        apply_warmup(optimizer, epoch, int(tcfg.get("cnn_warmup_epochs", 0)))
        model.train()
        tot, cnt = 0.0, 0
        for stft, fc, labels in tr_loader:
            stft = stft.to(dev_t); fc = fc.to(dev_t); labels = labels.to(dev_t)
            optimizer.zero_grad()
            arr = model(stft, sc=sc, fc=fc, return_arrays=True)
            y = labels.view(-1)
            L_main = criterion(arr["logit"].view(-1), y)
            L_dir = 0.5 * (criterion(arr["z_f"].view(-1), y) +
                           criterion(arr["z_b"].view(-1), y))
            loss = L_main + gamma_dir * L_dir
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["max_grad_norm"]))
            optimizer.step()
            tot += float(loss.item()) * labels.numel(); cnt += labels.numel()
        train_loss = tot / max(cnt, 1)

        dv = predict_full(model, dv_loader, sc, dev_t)
        dev_auc = roc_auc(dv["labels"], dv["probs"])
        dev_auc_f = roc_auc(dv["labels"], dv["forward_probs"])
        dev_auc_b = roc_auc(dv["labels"], dv["backward_probs"])
        monitor = dev_auc if dev_auc is not None else 0.5
        scheduler.step(monitor)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "dev_auc": dev_auc, "dev_auc_forward": dev_auc_f,
                        "dev_auc_backward": dev_auc_b})

        if monitor > best_dev:
            best_dev = monitor
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_dev_directional = {"dev_auc_bidirectional": dev_auc,
                                    "dev_auc_forward": dev_auc_f,
                                    "dev_auc_backward": dev_auc_b}
            patience = 0
        else:
            patience += 1
            if patience >= int(tcfg["early_stop_patience"]):
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(dev_t) for k, v in best_state.items()})
    else:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    te = predict_full(model, te_loader, sc, dev_t)

    n_test = len(te["labels"])
    if not (n_test == len(te_meta["clip_ids"]) == len(te_meta["case_ids"])
            == len(te_meta["edf_files"])):
        raise RuntimeError("TEST prediction/metadata length mismatch")

    model_name = cfg.get("model", "CNN_DA_BI_G3_V4")
    ck_dir = os.path.join(out_dir, "seed%s" % seed, "checkpoints")
    os.makedirs(ck_dir, exist_ok=True)
    ck_path = os.path.join(ck_dir, "%s_%s.pt" % (model_name, fold))
    torch.save({
        "model_name": model_name,
        "architecture": "cnn_da_bi_g3_v4",
        "fold": fold,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_dev": float(best_dev),
        "state_dict": best_state,
        "normalizer_mean": data["normalizer"]["mean"],
        "normalizer_std": data["normalizer"]["std"],
        "pos_weight": pos_weight,
        "optimizer_config": opt_config,
        "model_config": mc,
    }, ck_path)

    return {
        "best_epoch": best_epoch,
        "best_dev": float(best_dev),
        "dev_auc_bidirectional": (best_dev_directional or {}).get("dev_auc_bidirectional"),
        "dev_auc_forward": (best_dev_directional or {}).get("dev_auc_forward"),
        "dev_auc_backward": (best_dev_directional or {}).get("dev_auc_backward"),
        "test_auc": roc_auc(te["labels"], te["probs"]),
        "test_auc_forward": roc_auc(te["labels"], te["forward_probs"]),
        "test_auc_backward": roc_auc(te["labels"], te["backward_probs"]),
        "test_probs": [float(x) for x in te["probs"]],
        "test_raw_logits": [float(x) for x in te["raw_logits"]],
        "test_forward_probs": [float(x) for x in te["forward_probs"]],
        "test_backward_probs": [float(x) for x in te["backward_probs"]],
        "test_forward_raw_logits": [float(x) for x in te["forward_raw_logits"]],
        "test_backward_raw_logits": [float(x) for x in te["backward_raw_logits"]],
        "test_direction_disagreement": [float(x) for x in te["direction_disagreement"]],
        "test_direction_weight_forward": [float(x) for x in te["direction_weight_forward"]],
        "test_direction_weight_backward": [float(x) for x in te["direction_weight_backward"]],
        "test_selective_readout_entropy": [float(x) for x in te["readout_entropy"]],
        "test_selective_readout_peak": [float(x) for x in te["readout_peak_weight"]],
        "lambda_value": te["lambda_value"],
        "gate_mean": te["gate_mean"],
        "gate_p95": te["gate_p95"],
        "test_labels": [int(x) for x in te["labels"]],
        "test_clip_ids": list(te_meta["clip_ids"]),
        "test_case_ids": list(te_meta["case_ids"]),
        "test_edf_files": list(te_meta["edf_files"]),
        "checkpoint": ck_path,
        "optimizer_config": opt_config,
        "train_config": tcfg,
        "pos_weight": pos_weight,
        "n_train": int(len(tr_y)), "n_dev": int(len(dv_y)), "n_test": int(n_test),
        "history": history,
        "epochs_run": len(history),
        "direction_config": dict(model.direction_config),
    }
