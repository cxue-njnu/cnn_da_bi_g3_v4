# -*- coding: utf-8 -*-
"""source_adapted/train_v4.py

Per-fold training/evaluation for CNN_DA_BI_G3_V4.

FAST RUNTIME
------------
- BF16 AMP by default on supported CUDA GPUs; FP16+GradScaler fallback.
- TF32 + cuDNN benchmark in fast mode.
- optional torch.compile(mode="reduce-overhead") with eager fallback.
- pinned CPU DataLoader memory + non_blocking H2D copies.
- minimal training forward: no gate quantile/.item()/readout entropy diagnostics.
- DEV each epoch computes final AUC only.
- full directional/interpretability diagnostics run only after the best checkpoint
  is restored.
"""

import os
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.cnn_da_bi_g3_v4 import (
    build_cnn_da_bi_g3_v4,
    build_optimizer_v4,
)


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

DEFAULT_RUNTIME_CONFIG = {
    "runtime_mode": "fast",          # fast | strict
    "amp": "auto",                   # auto | off | bf16 | fp16
    "tf32": True,
    "cudnn_benchmark": True,
    "compile": "auto",               # auto | on | off
    "compile_mode": "reduce-overhead",
    "pin_memory": True,
    "non_blocking": True,
}


def _as_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def seed_everything(seed, deterministic=True):
    import random

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def configure_runtime(runtime, device):
    runtime = dict(DEFAULT_RUNTIME_CONFIG, **(runtime or {}))
    mode = str(runtime.get("runtime_mode", "fast")).lower()
    is_cuda = torch.device(device).type == "cuda"

    if mode == "strict":
        runtime["amp"] = "off"
        runtime["tf32"] = False
        runtime["cudnn_benchmark"] = False
        runtime["compile"] = "off"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = _as_bool(
            runtime.get("cudnn_benchmark", True)
        )
        if is_cuda:
            tf32 = _as_bool(runtime.get("tf32", True))
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.backends.cudnn.allow_tf32 = tf32
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    return runtime


def resolve_amp_dtype(runtime, device):
    dev = torch.device(device)
    if dev.type != "cuda":
        return None

    amp = str(runtime.get("amp", "auto")).lower()
    if amp in {"off", "false", "0", "none"}:
        return None

    if amp in {"auto", "bf16"}:
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        if amp == "bf16":
            print("[runtime] BF16 unsupported; falling back to FP16", flush=True)

    return torch.float16


def autocast_context(device, amp_dtype):
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast(
        device_type=torch.device(device).type,
        dtype=amp_dtype,
    )


def roc_auc(labels, scores):
    y = np.asarray(labels).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
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


def make_train_loader(stft, fc, labels, batch_size, seed, pin_memory=False):
    g = torch.Generator()
    g.manual_seed(int(seed))
    return DataLoader(
        TensorDataset(stft, fc, labels),
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        generator=g,
        drop_last=False,
        pin_memory=bool(pin_memory),
    )


def make_eval_loader(stft, fc, labels, batch_size, pin_memory=False):
    return DataLoader(
        TensorDataset(stft, fc, labels),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=bool(pin_memory),
    )


def compute_pos_weight(labels):
    y = np.asarray(labels).reshape(-1)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return (
        float(n_neg) / float(max(n_pos, 1))
        if n_pos > 0
        else 1.0
    )


def apply_warmup(optimizer, epoch, warmup_epochs, scale=0.3):
    idx = getattr(optimizer, "v4_group_index", {})
    full = getattr(optimizer, "v4_full_lr", None)
    if full is None:
        return

    low = {idx.get("g2"), idx.get("g3")}
    in_warmup = (
        warmup_epochs > 0
        and epoch <= int(warmup_epochs)
    )

    for gi, group in enumerate(optimizer.param_groups):
        if gi in low and gi is not None:
            group["lr"] = (
                full[gi] * scale
                if in_warmup
                else full[gi]
            )


def maybe_compile_model(model, runtime, device):
    mode = str(runtime.get("compile", "auto")).lower()
    if (
        torch.device(device).type != "cuda"
        or mode in {"off", "false", "0", "none"}
        or not hasattr(torch, "compile")
    ):
        return model, "off"

    try:
        compiled = torch.compile(
            model,
            mode=str(runtime.get("compile_mode", "reduce-overhead")),
        )
        return compiled, "enabled"
    except Exception as exc:
        print(
            "[runtime] torch.compile setup failed; using eager: %r" % exc,
            flush=True,
        )
        return model, "fallback_setup"


def _to_device(x, device, non_blocking):
    return x.to(device, non_blocking=bool(non_blocking))


@torch.inference_mode()
def predict_auc_only(model, loader, sc, device, amp_dtype=None,
                     non_blocking=False):
    """Fast DEV path: final score + labels only."""
    model.eval()
    probs = []
    labels_all = []

    for stft, fc, labels in loader:
        stft = _to_device(stft, device, non_blocking)
        fc = _to_device(fc, device, non_blocking)

        with autocast_context(device, amp_dtype):
            logits = model(stft, sc=sc, fc=fc)

        probs.append(torch.sigmoid(logits).detach().float().cpu())
        labels_all.append(labels.detach().float().cpu())

    return (
        torch.cat(labels_all).numpy() if labels_all else np.array([]),
        torch.cat(probs).numpy() if probs else np.array([]),
    )


@torch.inference_mode()
def predict_directional_auc_only(model, loader, sc, device, amp_dtype=None,
                                 non_blocking=False):
    """Run once after best checkpoint: final + forward + backward AUC scores."""
    model.eval()
    out = {"y": [], "p": [], "pf": [], "pb": []}

    for stft, fc, labels in loader:
        stft = _to_device(stft, device, non_blocking)
        fc = _to_device(fc, device, non_blocking)

        with autocast_context(device, amp_dtype):
            arr = model(
                stft,
                sc=sc,
                fc=fc,
                return_train_arrays=True,
            )

        out["p"].append(torch.sigmoid(arr["logit"]).detach().float().cpu())
        out["pf"].append(torch.sigmoid(arr["z_f"]).detach().float().cpu())
        out["pb"].append(torch.sigmoid(arr["z_b"]).detach().float().cpu())
        out["y"].append(labels.detach().float().cpu())

    return {
        k: (torch.cat(v).numpy() if v else np.array([]))
        for k, v in out.items()
    }


@torch.inference_mode()
def predict_full(model, loader, sc, device, amp_dtype=None,
                 non_blocking=False):
    """Full TEST/mechanism path, run only after best checkpoint is restored."""
    model.eval()
    keys = (
        "prob", "raw",
        "f_prob", "b_prob",
        "f_raw", "b_raw",
        "w_f", "w_b",
        "entropy", "peak", "y",
    )
    out = {k: [] for k in keys}
    scalars = []

    for stft, fc, labels in loader:
        stft = _to_device(stft, device, non_blocking)
        fc = _to_device(fc, device, non_blocking)

        with autocast_context(device, amp_dtype):
            arr = model(
                stft,
                sc=sc,
                fc=fc,
                return_arrays=True,
            )

        out["raw"].append(arr["logit"].detach().float().cpu())
        out["f_raw"].append(arr["forward_raw_logit"].detach().float().cpu())
        out["b_raw"].append(arr["backward_raw_logit"].detach().float().cpu())
        out["prob"].append(torch.sigmoid(arr["logit"]).detach().float().cpu())
        out["f_prob"].append(
            torch.sigmoid(arr["forward_raw_logit"]).detach().float().cpu()
        )
        out["b_prob"].append(
            torch.sigmoid(arr["backward_raw_logit"]).detach().float().cpu()
        )
        out["w_f"].append(
            arr["direction_weight_forward"].detach().float().cpu()
        )
        out["w_b"].append(
            arr["direction_weight_backward"].detach().float().cpu()
        )
        out["entropy"].append(
            arr["readout_entropy"].detach().float().cpu()
        )
        out["peak"].append(
            arr["readout_peak_weight"].detach().float().cpu()
        )
        out["y"].append(labels.detach().float().cpu())

        gm = arr.get("gate_mean")
        if gm is not None:
            lam = arr["lambda_value"]
            lam = float(lam.detach().float().cpu()) if torch.is_tensor(lam) else float(lam)
            scalars.append([
                lam,
                float(gm),
                float(arr.get("gate_p95") or 0.0),
            ])

    cat = {
        k: (torch.cat(v).numpy() if v else np.array([]))
        for k, v in out.items()
    }

    if scalars:
        s = np.asarray(scalars, dtype=np.float64)
        lam = float(s[:, 0].mean())
        gmean = float(s[:, 1].mean())
        gp95 = float(s[:, 2].mean())
    else:
        lam = gmean = gp95 = None

    return {
        "probs": cat["prob"],
        "raw_logits": cat["raw"],
        "forward_probs": cat["f_prob"],
        "backward_probs": cat["b_prob"],
        "forward_raw_logits": cat["f_raw"],
        "backward_raw_logits": cat["b_raw"],
        "direction_weight_forward": cat["w_f"],
        "direction_weight_backward": cat["w_b"],
        "direction_disagreement": np.abs(cat["f_prob"] - cat["b_prob"]),
        "readout_entropy": cat["entropy"],
        "readout_peak_weight": cat["peak"],
        "labels": cat["y"],
        "lambda_value": lam,
        "gate_mean": gmean,
        "gate_p95": gp95,
    }


def run_fold(cfg, fold, data, sc_np, device, out_dir):
    seed = int(cfg["seed"])
    tcfg = dict(DEFAULT_TRAIN_CONFIG)
    tcfg.update({
        k: cfg[k]
        for k in DEFAULT_TRAIN_CONFIG
        if k in cfg
    })

    runtime = dict(DEFAULT_RUNTIME_CONFIG)
    runtime.update(cfg.get("runtime_config", {}))

    dev_t = torch.device(device)
    runtime = configure_runtime(runtime, dev_t)

    seed_everything(
        seed,
        deterministic=(str(runtime["runtime_mode"]).lower() == "strict"),
    )

    amp_dtype = resolve_amp_dtype(runtime, dev_t)
    pin_memory = (
        _as_bool(runtime.get("pin_memory", True))
        and dev_t.type == "cuda"
    )
    non_blocking = (
        _as_bool(runtime.get("non_blocking", True))
        and pin_memory
        and dev_t.type == "cuda"
    )

    mc = dict(cfg.get("model_config", {}))
    mc["direction_dropout"] = float(
        tcfg.get("direction_dropout", 0.10)
    )

    # Keep raw_model authoritative for optimizer/state_dict/checkpoints.
    raw_model = build_cnn_da_bi_g3_v4(mc).to(dev_t)

    lr_map = {
        "g1": float(tcfg["lr_group1"]),
        "g2": float(tcfg["lr_group2"]),
        "g3": float(tcfg["lr_group3"]),
        "g4": float(tcfg["lr_group4"]),
    }
    wd_map = {
        "g1": float(tcfg["wd_group1"]),
        "g2": float(tcfg["wd_group2"]),
        "g3": float(tcfg["wd_group3"]),
        "g4": float(tcfg["wd_group4"]),
    }
    optimizer, opt_config = build_optimizer_v4(
        raw_model,
        lr_map=lr_map,
        wd_map=wd_map,
        enable_direction_lora=bool(
            mc.get("enable_direction_lora", False)
        ),
    )

    exec_model, compile_status = maybe_compile_model(
        raw_model,
        runtime,
        dev_t,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=int(tcfg["scheduler_patience"]),
        factor=float(tcfg["scheduler_factor"]),
    )

    tr_stft, tr_fc, tr_y, _tr_meta = data["train"]
    dv_stft, dv_fc, dv_y, _dv_meta = data["dev"]
    te_stft, te_fc, te_y, te_meta = data["test"]

    pos_weight = compute_pos_weight(tr_y.numpy())
    pw_t = torch.tensor(
        [pos_weight],
        dtype=torch.float32,
        device=dev_t,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw_t)
    gamma_dir = float(tcfg.get("gamma_direction", 0.15))

    batch = int(cfg.get("batch", 32))
    tr_loader = make_train_loader(
        tr_stft, tr_fc, tr_y, batch, seed,
        pin_memory=pin_memory,
    )
    dv_loader = make_eval_loader(
        dv_stft, dv_fc, dv_y, batch,
        pin_memory=pin_memory,
    )
    te_loader = make_eval_loader(
        te_stft, te_fc, te_y, batch,
        pin_memory=pin_memory,
    )

    sc = torch.from_numpy(
        np.asarray(sc_np, dtype=np.float32)
    ).to(dev_t)

    use_scaler = (
        amp_dtype == torch.float16
        and dev_t.type == "cuda"
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    best_dev = -1.0
    best_epoch = -1
    best_state = None
    patience = 0
    history = []

    def training_forward(stft, fc):
        nonlocal exec_model, compile_status
        try:
            return exec_model(
                stft,
                sc=sc,
                fc=fc,
                return_train_arrays=True,
            )
        except Exception as exc:
            if exec_model is not raw_model:
                print(
                    "[runtime] compiled forward failed; switching to eager: %r"
                    % exc,
                    flush=True,
                )
                exec_model = raw_model
                compile_status = "fallback_runtime"
                return raw_model(
                    stft,
                    sc=sc,
                    fc=fc,
                    return_train_arrays=True,
                )
            raise

    for epoch in range(1, int(tcfg["epochs"]) + 1):
        apply_warmup(
            optimizer,
            epoch,
            int(tcfg.get("cnn_warmup_epochs", 0)),
        )

        raw_model.train()
        # compiled wrapper observes raw_model.training because it wraps raw_model.
        loss_sum = torch.zeros((), device=dev_t, dtype=torch.float32)
        cnt = 0

        for stft, fc, labels in tr_loader:
            stft = _to_device(stft, dev_t, non_blocking)
            fc = _to_device(fc, dev_t, non_blocking)
            labels = _to_device(labels, dev_t, non_blocking)

            optimizer.zero_grad(set_to_none=True)

            with autocast_context(dev_t, amp_dtype):
                arr = training_forward(stft, fc)
                y = labels.view(-1)
                L_main = criterion(arr["logit"].view(-1), y)
                L_dir = 0.5 * (
                    criterion(arr["z_f"].view(-1), y)
                    + criterion(arr["z_b"].view(-1), y)
                )
                loss = L_main + gamma_dir * L_dir

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    raw_model.parameters(),
                    float(tcfg["max_grad_norm"]),
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    raw_model.parameters(),
                    float(tcfg["max_grad_norm"]),
                )
                optimizer.step()

            loss_sum += loss.detach().float() * labels.numel()
            cnt += labels.numel()

        train_loss = float(loss_sum.item()) / max(cnt, 1)

        # DEV fast path: no direction diagnostics, no gate quantile, no entropy.
        try:
            dv_labels, dv_probs = predict_auc_only(
                exec_model,
                dv_loader,
                sc,
                dev_t,
                amp_dtype=amp_dtype,
                non_blocking=non_blocking,
            )
        except Exception as exc:
            if exec_model is not raw_model:
                print(
                    "[runtime] compiled DEV failed; switching to eager: %r"
                    % exc,
                    flush=True,
                )
                exec_model = raw_model
                compile_status = "fallback_runtime"
                dv_labels, dv_probs = predict_auc_only(
                    raw_model,
                    dv_loader,
                    sc,
                    dev_t,
                    amp_dtype=amp_dtype,
                    non_blocking=non_blocking,
                )
            else:
                raise

        dev_auc = roc_auc(dv_labels, dv_probs)
        monitor = dev_auc if dev_auc is not None else 0.5
        scheduler.step(monitor)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_auc": dev_auc,
        })

        if monitor > best_dev:
            best_dev = monitor
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in raw_model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= int(tcfg["early_stop_patience"]):
                break

    if best_state is not None:
        raw_model.load_state_dict({
            k: v.to(dev_t)
            for k, v in best_state.items()
        })
    else:
        best_state = {
            k: v.detach().cpu().clone()
            for k, v in raw_model.state_dict().items()
        }

    # Best-checkpoint directional DEV scores: one inexpensive pass, diagnostics off.
    best_dv = predict_directional_auc_only(
        raw_model,
        dv_loader,
        sc,
        dev_t,
        amp_dtype=amp_dtype,
        non_blocking=non_blocking,
    )
    best_dev_directional = {
        "dev_auc_bidirectional": roc_auc(best_dv["y"], best_dv["p"]),
        "dev_auc_forward": roc_auc(best_dv["y"], best_dv["pf"]),
        "dev_auc_backward": roc_auc(best_dv["y"], best_dv["pb"]),
    }

    # TEST runs once with full mechanism diagnostics.
    te = predict_full(
        raw_model,
        te_loader,
        sc,
        dev_t,
        amp_dtype=amp_dtype,
        non_blocking=non_blocking,
    )

    n_test = len(te["labels"])
    if not (
        n_test
        == len(te_meta["clip_ids"])
        == len(te_meta["case_ids"])
        == len(te_meta["edf_files"])
    ):
        raise RuntimeError("TEST prediction/metadata length mismatch")

    runtime_record = dict(runtime)
    runtime_record.update({
        "amp_resolved": (
            "bf16" if amp_dtype == torch.bfloat16
            else "fp16" if amp_dtype == torch.float16
            else "off"
        ),
        "compile_status": compile_status,
        "pin_memory_resolved": bool(pin_memory),
        "non_blocking_resolved": bool(non_blocking),
    })

    model_name = cfg.get("model", "CNN_DA_BI_G3_V4")
    ck_dir = os.path.join(
        out_dir,
        "seed%s" % seed,
        "checkpoints",
    )
    os.makedirs(ck_dir, exist_ok=True)
    ck_path = os.path.join(
        ck_dir,
        "%s_%s.pt" % (model_name, fold),
    )

    torch.save({
        "model_name": model_name,
        "architecture": "cnn_da_bi_g3_v4_scfix_fast",
        "implementation_version": "V4_SCFIX_FAST_R1",
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
        "runtime_config": runtime_record,
    }, ck_path)

    return {
        "best_epoch": best_epoch,
        "best_dev": float(best_dev),
        "dev_auc_bidirectional": best_dev_directional["dev_auc_bidirectional"],
        "dev_auc_forward": best_dev_directional["dev_auc_forward"],
        "dev_auc_backward": best_dev_directional["dev_auc_backward"],
        "test_auc": roc_auc(te["labels"], te["probs"]),
        "test_auc_forward": roc_auc(te["labels"], te["forward_probs"]),
        "test_auc_backward": roc_auc(te["labels"], te["backward_probs"]),
        "test_probs": [float(x) for x in te["probs"]],
        "test_raw_logits": [float(x) for x in te["raw_logits"]],
        "test_forward_probs": [float(x) for x in te["forward_probs"]],
        "test_backward_probs": [float(x) for x in te["backward_probs"]],
        "test_forward_raw_logits": [
            float(x) for x in te["forward_raw_logits"]
        ],
        "test_backward_raw_logits": [
            float(x) for x in te["backward_raw_logits"]
        ],
        "test_direction_disagreement": [
            float(x) for x in te["direction_disagreement"]
        ],
        "test_direction_weight_forward": [
            float(x) for x in te["direction_weight_forward"]
        ],
        "test_direction_weight_backward": [
            float(x) for x in te["direction_weight_backward"]
        ],
        "test_selective_readout_entropy": [
            float(x) for x in te["readout_entropy"]
        ],
        "test_selective_readout_peak": [
            float(x) for x in te["readout_peak_weight"]
        ],
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
        "runtime_config": runtime_record,
        "model_config": mc,
        "pos_weight": pos_weight,
        "n_train": int(len(tr_y)),
        "n_dev": int(len(dv_y)),
        "n_test": int(n_test),
        "history": history,
        "epochs_run": len(history),
        "direction_config": dict(raw_model.direction_config),
        "implementation_version": "V4_SCFIX_FAST_R1",
    }
