# CNN_DA_BI_G3_V4 - Implementation Report

Task: EXP08_V22_V4_CNN_ENCODED_DIRECTION_ADAPTIVE_BI_G3_IMPLEMENTATION_V1
Mode: IMPLEMENT_CODE_TEST_AND_ONE_FOLD_READY_ONLY

## What was implemented

A complete, runnable V4 model workspace under E:/seizure pred/cnn_da_bi_g3_v4
(read-only reuse of V2_2_FULL_PACKAGE, baseline_seizure_prediction_cnn_v22 and
baseline_bi_g3_prime2_v22; no reference/V2.2 file is modified).

Components:
- source_adapted/models/multiscale_stft_cnn.py - shared-per-electrode multi-scale STFT CNN
  with time-reversal-symmetric local kernels and a raw STFT skip.
- source_adapted/models/g3_core_v4.py - connectivity-conditioned selective SSM core
  (state_dim 128, context_rank 32, 2 layers, CNN-latent SC/FC context, real Delta/Abar,
  Adaptive selective readout inputs).
- source_adapted/models/cnn_da_bi_g3_v4.py - composite: CNN + SC/FC latent aggregation +
  shared fusion + G3 (forward/backward) + differentiable selective readout + direction-adaptive
  gate + classifier + direction auxiliary loss, and a 4-group AdamW optimizer builder.
- source_adapted/{dataset,folds,normalization}.py - read-only fold/feature access and a
  TRAIN-only (18,70) normalizer.
- source_adapted/{train_v4,run_cnn_da_bi_g3_v4}.py - per-fold train/eval and supervisor CLI.
- compute_v4_metrics.py - fold / patient-level CSV + JSON metrics.
- configs, docs (design + protocol), README, pytest.ini.

V4 mechanisms all wired in: selective readout uses y_t = W_out(C_t h_t) with temporal-node
softmax attention (not only h_T); direction-adaptive gate replaces fixed 50:50 mean fusion;
directional auxiliary loss gamma=0.15; direction dropout 0.10 train-only; direction LoRA scaffold
left disabled.

## Test results

python -m pytest tests -q  => 20 passed

T1 CNN shape 53x18x96 PASS; T2 no time/node loss PASS; T3 SC/FC latent agg (2,53,18,96) PASS;
T4 forward/backward r (2,128) PASS; T5 selective-readout weights sum to 1 PASS; T6 direction
weights sum to 1 in [0,1] PASS; T7 reversal symmetry (eval, forward<->backward swap, final prob
within tolerance) PASS; T8 finite gradients across all key modules PASS; T9 TRAIN-only normalizer
(18,70) PASS; T10 end-to-end synthetic fold through the real dataset pipeline -> complete
checkpoint/result/predictions/DONE PASS; T11 complete DONE set skipped, partial rerun PASS;
T12 metrics AUC/AP/PR/log-loss verified against sklearn on manual arrays PASS.

Full checks passed: py_compile on all modules; pytest; runner --help; compute --help.

## Parameter count

Total ~366,183 trainable parameters (also total, fully covered: 409 parameter tensors, one group
each). Breakdown by optimizer group:
- Group 1 (CNN encoder + fusion): 119,168
- Group 2 (G3 shared/base):        165,888
- Group 3 (connectivity residual):  18,594
- Group 4 (readout/dir/classifier): 62,533

Well below 5M, so no parameter-size blocker. If it were >5M the main contributors would be the
multi-scale CNN branch channels and the 128-dim 2-layer SSM; that was not reached.

## How to run

C:/Users/Administrator/.conda/envs/tf210/python.exe -m py_compile source_adapted/run_cnn_da_bi_g3_v4.py source_adapted/models/cnn_da_bi_g3_v4.py
C:/Users/Administrator/.conda/envs/tf210/python.exe -m pytest tests -q
C:/Users/Administrator/.conda/envs/tf210/python.exe source_adapted/run_cnn_da_bi_g3_v4.py --help
C:/Users/Administrator/.conda/envs/tf210/python.exe compute_v4_metrics.py --help

Real fold run (owner-authorized later):
  python source_adapted/run_cnn_da_bi_g3_v4.py --seed 42 --device cuda --batch 32 --parallel 1
  python source_adapted/run_cnn_da_bi_g3_v4.py --seed 42 --device cuda --batch 32 --parallel 1 --fold chb01_event03
  python compute_v4_metrics.py --seed 42

## Status flags

REAL_FOLD_TRAINING_EXECUTED=false
FULL_117_FOLD_TRAINING_EXECUTED=false
V2_2_MUTATED=false
CNN_BASELINE_MUTATED=false
BI_G3_BASELINE_MUTATED=false

Next recommended action: Owner reviews implementation, then explicitly authorizes a one-fold real
pilot (chb01_event03) on GPU.