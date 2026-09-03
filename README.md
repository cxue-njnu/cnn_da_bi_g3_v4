# CNN_DA_BI_G3_V4 - CNN-Encoded Direction-Adaptive Bidirectional
# Connectivity-Conditioned Selective State Model

Experimental workspace for the V4 model.

Task         EXP08_V22_V4_CNN_ENCODED_DIRECTION_ADAPTIVE_BI_G3_IMPLEMENTATION_V1
Model        CNN_DA_BI_G3_V4
Mode         IMPLEMENT_CODE_TEST_AND_ONE_FOLD_READY_ONLY (no real folds executed)
Workspace    E:/seizure pred/cnn_da_bi_g3_v4

## Tree

cnn_da_bi_g3_v4/
├─ README.md
├─ configs/cnn_da_bi_g3_v4_seed42.json
├─ docs/V4_MODEL_DESIGN.md              V4_PROTOCOL.md
├─ source_adapted/
│    ├─ run_cnn_da_bi_g3_v4.py          (supervisor + CLI)
│    ├─ train_v4.py                     (per-fold train/eval)
│    ├─ dataset_v4.py                   (read-only V2.1+V2.2 feature access)
│    ├─ folds_v4.py                     (read-only patient_event_loso)
│    ├─ normalization.py                (TRAIN-only 18x70 normalizer)
│    └─ models/
│         ├─ cnn_da_bi_g3_v4.py         (composite + optimizer 4 groups)
│         ├─ multiscale_stft_cnn.py     (multi-scale shared-electrode CNN)
│         └─ g3_core_v4.py              (connectivity-conditioned SSM core)
├─ runs/seed42/{results,checkpoints,worker_logs}
├─ metrics/seed42/
├─ tests/  (conftest + 6 modules; python -m pytest tests -q -> 20 PASS)
└─ compute_v4_metrics.py

## Quick checks

    python -m py_compile source_adapted/run_cnn_da_bi_g3_v4.py source_adapted/models/cnn_da_bi_g3_v4.py
    python -m pytest tests -q
    python source_adapted/run_cnn_da_bi_g3_v4.py --help
    python compute_v4_metrics.py --help

Use the tf210 conda env when running locally:
    C:/Users/Administrator/.conda/envs/tf210/python.exe -m pytest tests -q

## CLI (real run; requires explicit owner authorization)

    python source_adapted/run_cnn_da_bi_g3_v4.py --seed 42 --device cuda --batch 32 --parallel 1
    python source_adapted/run_cnn_da_bi_g3_v4.py --seed 42 --device cuda --batch 32 --parallel 1 --fold chb01_event03
    python compute_v4_metrics.py --seed 42

## Scope / status

- Model implemented and synthetic-tested; ~366K parameters (<5M, no parameter blocker).
- No real fold trained. FULL_117_FOLD_TRAINING_EXECUTED=false.
- Content summary in implementation_report.md (generated at the end of implementation).
- Reference + V2.2 package untouched (V2_2_MUTATED=false, CNN_BASELINE_MUTATED=false,
  BI_G3_BASELINE_MUTATED=false).

See docs/V4_MODEL_DESIGN.md (architecture) and docs/V4_PROTOCOL.md (data/protocol/test rules).