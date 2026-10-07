# PHAT-JeT trigger-envelope research log

Goal (set 2026-10-07): PHAT-JeT inside the CMS CTL2 envelope — **≤173k LUT (10% of VU13P), <30 pipeline
stages (<100 ns @ 300 MHz), II=1, 0 DSP/BRAM** — at JEDI-Linear accuracy or better:
**≥78.0% at N=32, ≥80.9% at N=64** (pT-sorted JEDI-Linear, post-route, from their paper).
All hardware numbers here are da4ml estimates at the rebuttal convention (300 MHz, `hard_dc=2`,
`HWConfig(1,-1,-1)`, latency cutoff 4.0, inputs kif (1,8,16)) unless marked post-route.

Everything is under `/j-jepa-vol/phat-jet-aaron/` (scripts/, runs/, plots/, results/, logs/, env/).
Other people's directories (`/j-jepa-vol/phatjet-fpga` = Zihan's July campaign, `/j-jepa-vol/l1-jet-id` = data)
are read-only inputs.

## 2026-10-07 — orientation and findings

### Where the LUTs go (from Zihan's traces, read-only)
- Best quantized PHAT-JeT designs: 79.6% @ 385k LUT / 27 stages (N=128); 80.3% @ 487k / 41 stages (N=64).
  Nothing with ≥75% is inside the envelope. JEDI-Linear re-traced by Zihan: 80.4% @ 122k / 12 stages.
- Component costs (N=64 flagship, 544k LUT): local attention ≈200k, global patch attention ≈245k,
  head ≈55k, embedding ≈37k; GMP and FFN small. **Attention (activation×activation) is ~80% of the design.**
- GMP bolted onto JEDI-Linear at N=128 added ~145k LUT for +0.06 pts → GMP as a 2-D one-hot grid on all
  particles is unaffordable in the envelope.
- Float ceilings of the hardware architecture (tanh pre-norm, FFN×1, no LN): 81.1% (N=64), 80.9% (N=128
  d=16), 82.1% (N=128 "rich" teacher, 20.6k params). Paper float model: 81.8%.
- The two data files differ in jet order (labels agree at chance) → teacher logits must be recomputed on
  `jets_150x3_kt.npz`, which is what all my runs use (paper units, kT-sorted, padding at the end).
- User: in QAT the attention bits collapsed as well as GMP → protect attention lanes (bit floors, lower β).

### Physics plots (`plots/grid_stats`, `plots/diag`)
- Leading-K kT-ordered particles carry 94.6% (K=32) / 99.8% (K=64) of jet pT; 94% of jets have <64
  real particles, 41% have <32. ⇒ attention over the leading 32 only is a physics-justified pruning.
- Static vs dynamic GMP grid at paper resolution (δ=0.2 scaled) changes <5% of interacting pairs;
  odd bin counts matter only for coarse grids (δ≳0.4). Grid extent ±3.3 scaled (±0.4 raw) clips 1e-4.
- Paper-scale grid sweep (`runs/s1`, TF2.11 reimplementation of the paper model): kept dyn-δ0.2 (reference),
  core7, nogmp, 8×8; cancelled finer grids (unaffordable in hardware).

### Tooling built
- `scripts/phat_variants.py` — Keras3/HGQ2 builder, da4ml-traceable. Knobs: `global_mode` mha|mix|mean|none,
  `gmp_mode` grid|sep|none, `gmp_channels`, `attn_particles` (attention over leading K only), `share_qk`,
  heads, `use_head1`, `attn_bits` (cap), `ATTN_FLOOR` (min bits on attention lanes), `parallel_attn`.
- `scripts/train_variant.py` — float or QAT; β controller steering log(EBOPs) to `--ebops_target`;
  `--attn_floor`, `--attn_beta_scale`; distillation from precomputed teacher logits; `--w_l1`.
- `scripts/eval_ckpt.py` — load any campaign/own `.keras`, re-evaluate, `--trace` (LUT/FF/stages/latency).
- `scripts/loop_status.py` — aggregates all runs → `plots/loop/status.md`, `float_screen.png`, `qat_fronts.png`.
- `make_var_jobs.py` — k8s jobs per stage (labels `study=phatjet-var`).

### Experiment plan
1. **f1 float screen** (N=64 & 32, 2 seeds, 300 ep): baseline vs global mix/mean/none, local none, GMP
   none/sep/5×5/8-ch, patch 4, d24, attention over leading 32/16, shared QK, 1–2 heads, no head1.
   Output: accuracy cost of each cut. Keep cuts costing <0.3 pts.
2. **q1 QAT** (baseline arch, from scratch, 1000 ep): EBOPs targets 120k/250k (N=64), 80k/150k (N=32),
   with/without attention protection (`attn_floor 3`, `attn_beta_scale 0.3`). Output: EBOPs→LUT map for
   this build and whether protection helps.
3. **q2**: QAT of the best f1 variant(s) with distillation (teacher = Zihan's rich128 re-scored on my data,
   or own float teacher), targets chosen from q1's EBOPs→LUT map. Trace every Pareto point on the cluster
   (CPU jobs); bit-exact Verilog for anything inside the envelope.
4. Iterate: tighten target / add cuts / change N.

### Decisions log
- Cancelled own grid jobs so-d02-k8, se-d02-k8, so-d02-k7, so-d04-k5, dyn-d08-k8, cnt15, cnt11, so-d06
  (user authorised cancelling own jobs; grids finer than core7 are unaffordable).
- Seed processes staggered 60 s (two startup crashes with 3 processes initialising CUDA at once);
  node hcc-nrp-shor-c6017.unl.edu excluded (cuBLAS init failure).
- venv on the volume (python3.10 + keras 3.12 + jax[cuda12] 0.6 + hgq2 0.1.9 + da4ml 0.6); jobs need
  `LD_PRELOAD=/opt/conda/lib/libstdc++.so.6 KERAS_BACKEND=jax`.

## 2026-10-07 23:xx — loop tick 2
- Grid sweep (`runs/s1`): core7-k3 pod evicted at epoch ~110 (node lost its GPUs; val ≈80.3–80.5% at that point),
  se-d0825 crashed at step 1 with `cublas error` on hcc-nrp-shor-c6017 (same node that killed nogmp), nogmp-r2 stuck
  in ImagePullBackOff on epic001.clemson.edu. Excluded nodes hcc-nrp-shor-c6017/c5925, epic001; resubmitted
  nogmp-r3, se-d0825-k3-r2, core7-k3-r2. dyn-d02-k8 (paper reference) healthy at epoch ~70, val ≈80.3–80.5%.
- venv install finished downloading (5.7 GB); smoke tests (import, float, QAT with attention protection, pruned
  variant) queued to run as soon as pip exits. f1 (float screen, 12 GPU jobs × 4 runs) and q1 (8 QAT jobs)
  launch right after the smoke tests pass.

## 2026-10-07 ~00:05 — loop ticks 3–4: stages f1 and q1 launched
- venv OK (keras 3.12.4, hgq 0.1.9, da4ml 0.6.0, jax 0.6.2 cuda12). Jobs need `KERAS_BACKEND=jax` (else Keras 3
  imports TensorFlow) and `LD_PRELOAD=/opt/conda/lib/libstdc++.so.6`.
- Smoke tests on CPU: float OK; QAT from scratch OK (β controller fixed to assign `layer._beta` like hgq's
  BetaScheduler); float→Q transfer needs the repo-validated init (bw 10/10, i0_w 3, i0_a 8) — now the default
  whenever `--float_ckpt` is given (7/7/0/0 from scratch, as the July campaign). Transfer check is now relative
  (≥90% of the donor's accuracy). Note: `count_params()` of a Q model includes per-element quantizer variables
  (~100k), not just weights.
- **f1 float screen launched** (12 jobs × 4 runs, `runs/f1`): b64 baseline; global mix/mean/none; local none+mix;
  GMP none/sep/5×5/8-ch; patch 4; d24; attention over leading 32/16; shared QK; 1–2 heads; no head1; N=32 set.
  First run (b64-lnone-gmix) training at ~2 s/epoch on an A10.
- **q1 QAT launched** (8 jobs, `runs/q1`, baseline arch from scratch, 1000 ep): N=64 targets 120k/250k EBOPs,
  N=32 targets 80k/150k, each ± attention protection (`--attn_floor 3 --attn_beta_scale 0.3`).
  Purpose: EBOPs→LUT map for this build + does protection stop the attention collapse.
- Next: when f1 lands → pick cuts costing <0.3 pts → q2 QAT with distillation (teacher logits recomputed on
  jets_150x3_kt) → trace every Pareto point (CPU jobs) → Verilog + bit-exact for anything inside the envelope.

## 2026-10-07 ~00:20 — first f1 result
- `f1/b64-lnone-gmix` seed0: **80.36% float** at N=64 with NO local attention and global patch attention replaced
  by a constant-weight token mixer (8 tokens × 16 → Dense(128) → Dense(16), broadcast). 18.2k params, 300 epochs
  in 6 min (≈1 s/epoch). Curve still rising slowly at the end. This removes every activation×activation
  multiply except none at all — the model is embed → GMP → mean tokens → mix → FFN → mean → head.
  Baseline b64 expected ≈81.1% (Zihan's float f64 teacher) → cost of dropping both attention blocks ≈0.7 pts.
- Launched **q2** (4 jobs): QAT of this architecture from scratch, N=64 targets 60k/120k EBOPs, N=32 40k/80k.

## 2026-10-07 ~00:50 — f1 results so far (N=64 float, 300 ep, seed0 unless noted)
| variant | val acc | note |
|---|---|---|
| b64-gmpsep | **81.17%** | separable GMP (1-D η pass + 1-D φ pass): same accuracy as the 2-D grid at ~1/8 the scatter/gather cost |
| b64-nogmp | 80.82% | GMP worth only ≈0.3 pts in float at N=64 |
| b64-lnone-gmix | 80.36 / 80.50% (2 seeds) | no attention at all |
| b32-nogmp | 78.89% | N=32 without GMP already above the JEDI N=32 target (78.0%) in float |
- Launched **f2** (10 configs × 2 seeds): attention-free × separable GMP, ± local attention over leading 32,
  d24, FFN×2, 600-epoch schedule, N=32 versions.
- Replaced the three not-yet-started 2-D-grid q2 arms with **q3**: QAT of attention-free + separable GMP
  (N=64 targets 60k/120k EBOPs, N=32 40k/80k). q2-02 (N=32, 2-D grid, 40k) left running as a reference.
- Builder fix: tanh embedding is now a `QUnaryFunctionLUT` layer (da4ml allows only ReLU inline); the one
  QAT job that had started on the old builder (q2-00) was resubmitted.
- Pipeline validated end-to-end on CPU: float → QAT (float transfer, relative sanity check) → `.keras` → rebuild
  from `config.json` + `load_weights` → da4ml trace (N=64 trace ≈75 s). Added `scripts/trace_run.py` (traces up
  to 6 Pareto checkpoints of a QAT run, spread in EBOPs) and `scripts/teacher_logits.py` (train-set logits of a
  float run for distillation).
- GPU failures mid-job on nautilus-ext-gpu01.fullerton.edu (CUDA-graph error at the end of b64-gmix-att16
  seed0, then JAX fell back to CPU at 485 ms/step for the next run) → jobs f1-07/f1-08 killed and their configs
  (gmix-att16, shareqk, h1, h2) resubmitted; f1-06 (att32, gmix-att32) also resubmitted. Nodes
  k8s-chase-ci-10.calit2 and nautilus-ext-gpu01.fullerton added to the exclusion list. Lesson: the job
  script should abort when JAX reports no GPU instead of silently training on CPU — add a check.
- More f1: b64-gmpsep seed1 81.15% (81.16 ± 0.01), b64-gmean 80.82%, b64-nogmp 80.80 ± 0.02, b32-gmix-p4 79.16%.
- Teacher logits from `f1/b64-gmpsep/seed0` (81.17%) written; launched **q4**: distilled QAT (α=0.5, T=2) of the
  attention-free + separable-GMP model, N=64 targets 60k/120k EBOPs, N=32 40k/80k. Future jobs run with
  `JAX_PLATFORMS=cuda` so a dead GPU aborts the run instead of falling back to CPU. Status/trace commands now use
  the 7-day pod `anrunw-grid-explore-2`.
- 01:10 — third GPU death (same CUDA-graph error) on nautilus-ext-gpu01 killed job f1-04; gmpch8 + p4 resubmitted
  (f1-00-r4). Other pods on that node train normally (multi-GPU node, one bad device). b64-gmp5 (5×5 grid) =
  81.03%: GMP resolution can be cut 2.5× without loss.

## 2026-10-07 ~01:30 — β controller was the problem; switched to HGQ2's BetaPID
- My controller multiplied β by exp(kp·err + ki·∫err) every epoch — an integrator in log space. With EBOPs starting
  ~2 decades above target it drove β up ~×1.6/epoch: EBOPs 6M → 20k in 40 epochs, val acc 70% → 55% and never
  recovered (q2/q32-lg-t40k; same signature in all q1 runs). Cancelled all running q1/q2 arms.
- Replaced by `hgq.utils.sugar.beta_pid.BetaPID` (log-space PID on the *level*, warm-up, max_beta, damping) wrapped
  to keep the per-layer attention β scale. Gains p=0.1, i=0.002 (campaign: 0.1/0.001), β0=2e-8, max 3e-5,
  damp 0.02. Smoke: β rises ~1%/epoch at a 2-decade error — a ramp over hundreds of epochs, as intended.
- Relaunched q3 (attention-free + sep GMP) and q4 (same + distillation) with **2000 epochs** (≈1 s/epoch);
  q1 (baseline arch ± attention protection) resubmitted at 1000 epochs. Note: `beta` is logged by the PID callback.
- Float baseline **b64 = 81.26%** (seed0). Cuts relative to it: separable GMP −0.1, 5×5 grid −0.2, no GMP −0.45,
  global attention → mean token −0.45, no attention at all −0.8. b64-nohead1 was at 81.06% when its node died.
- nautilus-ext-gpu01.fullerton.edu lost all its GPUs (UnexpectedAdmissionError) and killed f1-05/f1-09 as well;
  their unfinished configs resubmitted (gmix-d24, gmix-p4, nohead1, gmix-att32-h2-shareqk).
- 01:45 — new failure class on several nodes (chi-dgx-node04, gpu-12/14.mghpcc, k8s-chase-ci-10): cuDNN
  "Failed to determine best convolution algorithm / All algorithms tried" on the depthwise GMP conv backward pass
  (XLA autotuner). Killed ~18 affected float/QAT jobs; jobs now run with
  `XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false` (fallback algo instead of error), default GPU memory
  preallocation (MEM_FRACTION 0.9), those nodes excluded, and a run is skipped if its result.json exists.
  Resubmitted: f1 (14 configs, -r7), f2 (all, -r2), q3 (3 arms, -r2), q4 (all, -r2). q3-00 (q64-lgsep-t60k) kept
  running: val 75.4% at EBOPs 1.0M, β 2.25e-8 (ramping as designed).
- Lesson: `kubectl apply -f <dir>/` resurrects deleted jobs from old manifests — superseded yamls now live in
  `jobs/<stage>/old/`.
- f1/f2 so far: b64-gnone 80.57% (no global stage), f2/b64-lg-nogmp 80.00%, f2/b64-lg-sep 79.67% (1 seed; lower
  than lg with the 2-D grid, 80.43 — to be confirmed with seed1 and the 600-epoch arm).
- 02:00 — q3/q64-lgsep-t60k (new controller): epoch 297/2000, EBOPs 6.6M → 520k, val 78–79% (float ceiling of this
  arch ≈79.7–80.4%), β rising gently. Tracing 4 early Pareto points to calibrate EBOPs→LUT for the attention-free
  model. XLA conv fallback confirmed working (gmpch8 ran on a resubmitted job: 81.11%, i.e. GMP on 8 of 16
  channels is free). Global-attention → constant mixer costs 0.3 pts with local attention kept (gmix 80.97%).
- 02:15 — **Root cause of both cuDNN failure classes: Tesla V100 (Volta) nodes** (chi-dgx-node04, cph-dgx-node7/9).
  Grouped (depthwise) conv fails there with jax 0.6 / cuDNN 9 ("All algorithms tried", then "cudnn status 5003"
  with the fallback algo); A10, 2080 Ti and 4090 nodes run the same code fine. V100 removed from the allowed GPU
  list; the 10 V100-hosted jobs killed and resubmitted (h1/h2, b32-sep arms, q32-lgsep, all q4, q32-t80k-prot).
- 02:30 — **first traces of the attention-free + separable-GMP QAT run** (q3/q64-lgsep-t60k, epoch ≈340):
  79.1% @ 469k EBOPs → 752k LUT, **15 stages = 50 ns**; 79.2% @ 593k EBOPs → 900k LUT, 15 stages.
  LUT ≈ 1.6 × EBOPs for this architecture ⇒ the 173k-LUT budget ≈ 105k EBOPs; the running targets
  (60k/120k at N=64, 40k/80k at N=32) bracket it. Latency is no longer a constraint without attention.
- User suggestion: Linformer-style attention (constant N→k projections of K/V, single head) as a cheaper global
  mechanism — adding as `local_mode=linf` (k=4/8, 1 head) to the float screen.
- 02:50 — Linformer variant (`--local linf --linf_k k`): Q from all (or leading-K) particles, K/V projected N→k
  with constant E/F (`QEinsumDense`, traceable), softmax over k; act×act cost 2·N·k·d. Smoke OK incl. trace.
  Launched **f3** (8 configs × 2 seeds): k=4/8, 1 or 4 heads, ± global mixer, ± GMP, leading-32, N=32 versions.
- f1: b64-att32 81.19 ± 0.03, b64-p4 81.18 ± 0.04 (both lossless vs 81.29 baseline).
- 03:05 — Protocol: training now uses a fixed 10% validation split of the 620k training jets for monitoring and
  Pareto selection; the 260k `x_val` file is the untouched TEST set (reported as `test_acc` in result.json and by
  eval_ckpt/trace). Runs started before this (f1, part of f2, q3-00) selected on the test set like the July
  campaign did; their final-epoch numbers are unbiased, Pareto-selected QAT points will be re-evaluated on test.
- Added `--sort pt` (descending-pT order = what CTL2 delivers, no sorter; leading-K = JEDI's top-K-by-pT) shared by
  train/eval/teacher code; verified monotone pT and padding at the end. Launched **f4** (8 configs × 2 seeds):
  pT-sorted full model, sep-GMP, att32, attention-free, Linformer k=4/8, N=32 versions.
- 03:30 — mid-run `cudnn status 5003` (CUDNN_STATUS_EXECUTION_FAILED_CUDA) also on A10/4090 after the env change
  (MEM_FRACTION=0.9 preallocation + lenient conv picker). Reverted the job env to the one that ran the first f1 jobs
  cleanly (PREALLOCATE=false, no XLA flag; V100 stays excluded) and resubmitted the failed/pending jobs
  (f1 h1/h2/shareqk/att16 -r10, nohead1/att32-h2-shareqk -r9, f2 gmix-sep -r4, f3 -r2, f4 -r2, q1/q3/q4 -r4).
  All deletes are my own `anrunw-*` jobs only.
- 04:00 — **Leaderboard**: https://aaronw5.github.io/phat-jet-leaderboard/ (repo aaronw5/phat-jet-leaderboard, Pages
  from main:/docs, commits authored by aaronw5). `scripts/make_leaderboard.py` collects every run (float: mean ± std
  over seeds of test accuracy / AUC / W,Z,t background rejection @80% TPR; QAT: every traced checkpoint with EBOPs,
  LUT, %VU13P, FF, DSP, BRAM, stages, latency, II, envelope flag) + reference rows (paper, JEDI-Linear, rebuttal);
  rows expand to Overview / Diagram (SVG from the config) / Specs / Code (training command + builder sources).
  `publish_leaderboard.sh` regenerates and pushes; the loop runs it when results change. Older float runs were
  re-evaluated with the paper metrics (`eval_all_float.sh` → merged into result.json).
