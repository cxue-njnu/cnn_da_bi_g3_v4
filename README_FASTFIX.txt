CNN_DA_BI_G3_V4 FASTFIX R1

Replace these repository files:
- source_adapted/train_v4.py
- source_adapted/dataset_v4.py
- source_adapted/run_cnn_da_bi_g3_v4.py
- source_adapted/models/cnn_da_bi_g3_v4.py
- source_adapted/models/g3_core_v4.py
- configs/cnn_da_bi_g3_v4_seed42.json

Important:
1. This bundle fixes SC aggregation and backward SC-feature time alignment.
   That changes model semantics, so old pre-fix fold results must not be mixed.
2. DONE markers now require implementation_version=V4_SCFIX_FAST_R1, so old
   DONE files are automatically treated as pending and rerun.
3. Fast mode defaults:
   AMP auto (BF16 preferred), TF32 on, cuDNN benchmark on, torch.compile auto,
   pinned memory and non-blocking H2D.
4. Strict validation:
   --runtime-mode strict
   forces deterministic FP32, TF32 off, benchmark off, compile off.
5. If torch.compile is unstable on the remote Windows environment:
   add --compile off. Other fast optimizations remain active.

Recommended one-fold validation:
python source_adapted\run_cnn_da_bi_g3_v4.py --seed 42 --device cuda --batch 32 --parallel 1 --fold chb08_event02 --runtime-mode fast --compile off

Then benchmark compile separately:
python source_adapted\run_cnn_da_bi_g3_v4.py --seed 43 --device cuda --batch 32 --parallel 1 --fold chb08_event02 --runtime-mode fast --compile on --out .\runs_compile_benchmark
