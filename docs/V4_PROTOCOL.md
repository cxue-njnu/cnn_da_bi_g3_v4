# CNN_DA_BI_G3_V4 - Protocol

## 0. Task scope

Task EXP08_V22_V4_CNN_ENCODED_DIRECTION_ADAPTIVE_BI_G3_IMPLEMENTATION_V1; mode IMPLEMENT_CODE_TEST_AND_ONE_FOLD_READY_ONLY.
This package implements code + synthetic tests + CLI + metrics; it does NOT run real folds.

## 1. Read-only inputs
- V2_2_ROOT = E:/seizure pred/V2_2_FULL_PACKAGE (read-only)
- CNN_BASELINE_REFERENCE = baseline_seizure_prediction_cnn_v22 (read-only)
- BI_G3_REFERENCE = baseline_bi_g3_prime2_v22 (read-only)

None of the reference / V2.2 feature stores, folds, results, checkpoints, model code, metrics or
config are modified. All new writes are under E:/seizure pred/cnn_da_bi_g3_v4.

## 2. Data protocol (strict reuse)

- patient-specific event-LOSO, 117 folds (read as-is).
- No fold rebuild, no train/dev/test change, no clip_id/case_id/edf_file/label change, no negative
  resampling.
- Input: STFT (53,18,70), FC (53,18,18), SC (18,18).
- Reused V2.1 features + V2.2 incremental features read only (feature lookup mirrors the G3Prime2
  baseline index, default preferring V2_1_REUSED).

## 3. Normalization (G3Prime2 / TRAIN-only, per fold)

- Statistics over TRAIN clips x 53 time steps preserving 18 nodes x 70 frequency bins.
- mean/std shape (18,70), incremental sum/sumsq/count accumulators.
- DEV and TEST never contribute to the normalizer.

## 4. Directory layout

cnn_da_bi_g3_v4/  with README.md, configs/, docs/, source_adapted/, runs/seed42/, metrics/seed42/,
tests/, compute_v4_metrics.py. See README for the tree.

## 5. Running (real folds only with explicit owner authorization)

python source_adapted/run_cnn_da_bi_g3_v4.py --seed 42 --device cuda --batch 32 --parallel 1
python source_adapted/run_cnn_da_bi_g3_v4.py --seed 42 --device cuda --batch 32 --parallel 1 --fold chb01_event03
python compute_v4_metrics.py --seed 42

Flag set: --seed --device --batch --parallel --fold --root --out (+ --cnn-warmup-epochs).
Defaults: seed=42 device=cuda batch=32 parallel=1 (owner may later use --parallel 2).

## 6. Outputs (per fold)

- runs/seed<seed>/results/CNN_DA_BI_G3_V4_<fold>.json
- runs/seed<seed>/results/CNN_DA_BI_G3_V4_<fold>_predictions.csv
- runs/seed<seed>/results/CNN_DA_BI_G3_V4_<fold>.done.json
- runs/seed<seed>/checkpoints/CNN_DA_BI_G3_V4_<fold>.pt

Result JSON carries model, fold, seed, best_epoch, best_dev, test_auc, test_logits, test_raw_logits,
test_labels, test_clip_ids, test_case_ids, test_edf_files, test_forward_raw_logits,
test_backward_raw_logits, test_forward_probs, test_backward_probs, direction_weight_forward,
direction_weight_backward, direction_disagreement, selective_readout_entropy,
selective_readout_peak_weight, lambda_value, gate_mean, gate_p95, checkpoint, optimizer_config,
model_config.

Predictions CSV columns: clip_id, case_id, edf_file, y_true, y_score, raw_logit, y_score_forward,
y_score_backward, direction_weight_forward, direction_weight_backward, direction_disagreement.

Resume rule: skip only when checkpoint + result JSON + predictions CSV + DONE JSON (status DONE) are
all present, readable and non-empty; otherwise partial => rerun. No hashing.

## 7. Training protocol

- epochs 80, early_stop_patience 10, monitor DEV ROC-AUC.
- ReduceLROnPlateau(mode max, patience 4, factor 0.5); gradient clip 5.0.
- AdamW, 4 parameter groups (see V4_MODEL_DESIGN.md).
- Loss L = L_main + gamma*L_dir, gamma=0.15; pos_weight = N_train_neg / N_train_pos.
- TRAIN loader shuffle=True num_workers=0; NO WeightedRandomSampler.
- checkpoint on DEV AUC only; TEST once on best-DEV; TEST never used for tuning.
- Two-stage CNN warm-up default 3 epochs (G3 base + connectivity at low LR, then joint).

## 8. Metrics (compute_v4_metrics.py --seed 42)

Fold ROC-AUC / AP / PR-AUC / Log Loss / Sensitivity / Specificity / Balanced Accuracy, forward AUC,
backward AUC, final AUC, direction weight mean/std, direction disagreement, readout entropy;
patient-level macro AUC / AP / PR-AUC and event-weighted AUC.
Outputs metrics_v4_folds.csv, metrics_v4_cases.csv, metrics_v4_models.csv,
metrics_v4_directionality.csv, metrics_v4_summary.json.

## 9. Tests (python -m pytest tests -q -> 20 PASS)

T1 CNN shape; T2 no time/node loss; T3 SC/FC latent agg; T4 forward/backward r (2,128);
T5 selective readout weights sum to 1; T6 direction weights sum to 1 in [0,1]; T7 reversal symmetry
(eval, forward<->backward swap, final approx); T8 finite gradients; T9 normalizer (18,70) TRAIN-only;
T10 end-to-end synthetic fold (real dataset+model+opt+DEV+TEST+checkpoint+result+prediction+DONE);
T11 resume complete-skip vs partial-rerun; T12 metrics (AUC/AP/PR/LL match sklearn on manual arrays).
