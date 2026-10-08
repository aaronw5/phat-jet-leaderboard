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
- 04:30 — f2: attention-free + separable GMP = 79.68 ± 0.01 (< attention-free + 2-D grid 80.43, < attention-free +
  no GMP 80.07): separable GMP hurts when there is no attention (its 1-D passes can't localize in 2-D on their own).
  d=24 attention-free 79.2 (no gain). Launched **q5** (5 jobs): attention-free with 5×5 grid / 8×8 grid on 8 channels /
  no GMP, targets 60k–100k EBOPs, mostly distilled. q3/q4 (separable) kept as the comparison.

## 2026-10-07 ~05:00 — the tanh embedding is 64% of the LUTs
- Component trace of q3/q64-lgsep-t60k epoch 402 (420k EBOPs, 729k LUT): embed Dense 45k, **embed tanh LUT 464k**,
  separable GMP ≈15k, patch tokens+mixer 51k, msg proj 16k, FFN 105k, pooling 27k, head 6k. The tanh is a
  per-element lookup table (64×16 tables addressed by a wide input) that the EBOPs penalty does not see; the
  "cheap" architecture was dominated by one activation. Zihan's embedding was 37k total, so his tanh must have had
  a narrow input or different placement.
- Switching the embedding to ReLU (inline, LUT-free; what JEDI-Linear uses). Launched **f5** (float check, 8 configs
  × 2 seeds) and **q6** (6 distilled QAT arms with ReLU: attention-free 8×8 / 5×5 grid, attention-on-32 + mixer,
  N=32 full and attention-free). Cancelled the dominated tanh arms (q3 -r2/-r4, q3-01, all q4, pending q5);
  kept q3-00, q5-01, q5-03 running as tanh references.
- Traces of q3-00 so far (tanh): 79.1% @ 752k, 78.9% @ 709k, 76.7% @ 476k, 74.0% @ 381k LUT; 13–15 stages.
- Also: single-head attention −0.8 (80.5); no hidden head −0.3 (80.94); N=32 pT-sorted 79.0 (kT 79.3).
- 05:40 — QAT curves: every variant holds ≈79% down to ~400k EBOPs then collapses (q3-00: 54% @ 74k; q5 lg5-kd
  77.9% @ 423k; q6 N=32 lg-relu-kd 78.3% @ 365k). JEDI-Linear sits at 80.4% @ 140k EBOPs. Structural difference:
  JEDI's per-particle layers have position-specific weights, so with pT-sorted input the regularizer prunes the
  soft slots; PHAT-JeT shares one Dense across all slots (no hardware saving from sharing in an unrolled design).
  Added `--unshared` (embed / GMP pointwise / FFN as per-slot QEinsumDense). Launched **f6** (8 float configs,
  pT-sorted, ReLU) and **q7** (4 distilled QAT arms). Linformer N=64 k=4 1-head = 80.2 ± 0.1 (3.0k params);
  k=8 4-head 80.2; N=32 k=8 78.5.
- Crash class: editing scripts on the volume while jobs run → Keras `could not get source code` when saving a
  checkpoint (lambda GMP indicators). Jobs now snapshot scripts/ into runs/_snapshots/<id>/ at start.
  Resubmitted q5 lg8c-kd. q1/q32-t80k-prot finished: 78.45% best val, 73.8% test at the end (over-compressed).
- TODO (user, 2026-10-07 ~04:40): add a Pareto tab to the leaderboard page: selectable x-axis (LUT, latency ns, stages, FF, EBOPs, params) and y-axis (test acc, avg bg rejection), envelope limits per axis, reference points, hover details; built from data.json.

## 2026-10-07 ~06:30 — first design inside the envelope (N=32) and per-slot weights change the game
- **q6/q32-lg-relu-t40k-kd epoch 455: 78.13% test, 169,370 LUT (9.8% VU13P), 13 stages, 43 ns, II=1, 0 DSP/BRAM**
  — inside the CTL2 envelope and above JEDI-Linear's N=32 number (78.0%; theirs is 45k LUT post-route though).
  Attention-free, ReLU embedding, 8×8 GMP grid, token mixer, distilled. Needs: Verilog + bit-exact check, 2nd seed.
- Per-slot weights (q7, pT-sorted, ReLU, distilled, N=64): 80.5% @ 330k EBOPs, 80.0% @ 253k, 79.0% @ 167k —
  vs shared weights needing ~400-500k EBOPs for 79%. N=32 per-slot: 78.5% @ 262k. Traces running.
- Finished 2000-epoch arms collapse at the end when the target is below the knee (q3-00, q5, q6-t40k/t60k):
  the useful points are the Pareto checkpoints on the way down, which is what we trace.
- Launched **q8** (8 arms, 3000 epochs where attention-free): per-slot attention-free at 150k/200k EBOPs,
  per-slot 5×5 grid, per-slot attention-on-32 + mixer at 200k/300k, N=32 per-slot at 100k/150k, N=32 per-slot full.
- Float (f6, per-slot, pT-sorted, ReLU): attention-free 80.1, 5×5 80.1, Linformer 80.2, att32+mixer 80.6, full 80.9;
  N=32 attention-free 78.6, full 79.2. Unsharing costs ≤0.3 in float.

## 2026-10-07 ~07:10 — INSIDE THE ENVELOPE at N=64 (per-slot weights)
Per-slot (unshared) weights, pT-sorted, ReLU embedding, attention-free (8×8 GMP grid + token mixer), distilled
(q7/q64-u-lg-t100k-kd and -t60k-kd), traced at 300 MHz hard_dc=2, accuracy on the full 260k test set:
| test acc | LUT | % VU13P | stages | latency |
|---|---|---|---|---|
| **80.31%** | **168,614** | **9.8** | 12 | 40 ns |
| 80.12% | 139,537 | 8.1 | 12 | 40 ns |
| 80.01% | 141,140 | 8.2 | 12 | 40 ns |
| 79.65% | 119,678 | 6.9 | 12 | 40 ns |
| 78.85% | 85,516 | 4.9 | 11 | 37 ns |
| 78.08% | 64,782 | 3.7 | 11 | 37 ns |
| 80.41% | 184,858 | 10.7 | 13 | 43 ns (just over) |
N=32 per-slot: 78.29% @ 154k, 78.12% @ 100k, 77.42% @ 74k. LUT/EBOPs ≈ 0.5 for per-slot models (vs 1.6 shared).
vs JEDI-Linear: N=64 80.9% @ 71k (post-route), N=32 78.0% @ 45k. We are 0.6 pts short of the N=64 target inside
the budget, and ~2 pts behind JEDI at equal LUT. Rebuttal number was 74.6% @ 167k.
Next: (1) Verilog + Verilator bit-exact for the 80.31%/169k and 80.12%/140k designs; (2) q9: per-slot full model
and per-slot attention-on-32 at 250–350k EBOPs (float 80.9 / 80.6), second seed of the winner, per-slot Linformer.
- 07:40 — q7/q64-u-lg-t100k-kd (the winner) crashed at epoch 1611/2000: `BlockingIOError: unable to lock file`
  while writing latest.weights.h5 on CephFS (HDF5 locking vs concurrent readers). Pareto points up to 1611 are
  intact; seed 1 (q7-01-s2) running. Future jobs set `HDF5_USE_FILE_LOCKING=FALSE`.
- Verilator: conda gcc's 2.12 sysroot lacks `timespec_get` → build with the system g++ instead (verilator from the
  conda env). Verilog for the 80.31%/169k and 78.12%/100k designs is written under /j-jepa-vol/phat-jet-aaron/verilog/;
  emulation re-running.
- 08:00 — helper pod explore-2 was killed by the admission webhook's ~6 h cap on bare pods (DeadlineExceeded);
  helper now runs as Job `anrunw-helper-1` (7-day deadline, 16 CPU/48 GB). Verilator: conda's verilated.mk hardcodes
  the conda compiler (whose sysroot clashes with the pod's glibc 2.31) → patched my env's verilated.mk to use
  /usr/bin/g++ throughout; emulation rerunning for the two in-envelope designs.
- q8/q9 (per-slot, N=64, val acc): attention-free 200k-target 80.64% @ 381k EBOPs (≥80% @ 281k); 250k-target 80.70%
  @ 436k; d=24 80.87% @ 620k; attention-on-32+mixer 80.7% @ 780k; full 80.5% @ 1.6M. Tracing the 250–450k band.

## 2026-10-07 ~08:30 — bit-exact RTL for both in-envelope designs
| design | test acc | avg rej | LUT | % | FF | stages | latency | Verilator vs Keras (512 jets) |
|---|---|---|---|---|---|---|---|---|
| N=64 per-slot attention-free, distilled (q7/q64-u-lg-t100k-kd ep508) | 80.31% | 52.1 | 168,614 | 9.8 | 114,995 | 12 | 40 ns | max err 0.0, argmax 100% |
| N=32 per-slot attention-free, distilled (q7/q32-u-lg-t40k-kd ep604) | 78.12% | 32.6 | 100,027 | 5.8 | 69,670 | 12 | 40 ns | max err 0.0, argmax 100% |
Verilog + metadata in /j-jepa-vol/phat-jet-aaron/verilog/<run>__<ckpt>/ (build_vivado_prj.tcl inside, part xcvu13p-flga2577-2-e,
3.33 ns). Verilator built with the system g++ (patched verilated.mk). Note the N=64 rejection (52) is well below the
full float model's (66): the compressed model keeps accuracy better than W/Z/t rejection — report both.
- More N=64 in-envelope points (q8/q9 per-slot): 80.27% @ 162k, 80.25% @ 147k, 80.19% @ 150k, 79.95% @ 140k;
  80.42% @ 194k and 80.51% @ 232k just outside. Curve plateau ≈80.3% inside the budget → 0.6 pts short of JEDI N=64.
- Launched **q10**: fine-tune the 80.27%@162k checkpoint at a fixed 330k-EBOPs target (1200 ep, lr 1e-3, β held)
  with the old and a stronger teacher (f5/b64-relu 81.28%, α 0.7, T 4), and from-scratch per-slot arms with the
  stronger teacher (8×8, 5×5, 8-channel GMP) at 330k.
- 09:50 — **New best inside the envelope: 80.46% test @ 141,858 LUT (8.2%), 13 stages, 43 ns** (q9/q64-u-lg-t250k-kd
  epoch 2357; 80.35% @ 141.8k at ep 2318). The 3000-epoch run is still improving (80.57% val @ 195k EBOPs at ep 2742).
  Fine-tune arms (q10): 79.97% @ 110k LUT, 80.05% @ 143k. d=24 fine-tune 80.38% val @ 333k EBOPs.
  Launched **q12**: 5000-epoch runs of the winning config (targets 200k/250k, both teachers, 2 seeds).
  Verilog + bit-exact running for the ep2357 design (copy in runs/_seeds/best_n64_ep2357.keras).
- 10:15 — **Best in-envelope so far: 80.56% test @ 144,622 LUT (8.4%), 12 stages, 40 ns** (q9/q64-u-lg-t250k-kd
  epoch 2760, snapshot runs/_traced/...epoch=2760...); 80.54% @ 141.7k (ep 2776). Tracer now copies every traced
  checkpoint to runs/_traced/ (Pareto dirs drop dominated points as runs improve). Verilog + bit-exact running.
- 10:40 — ep2760 design (80.56% @ 144.6k LUT, 12 stages, 40 ns) Verilator-verified bit-exact (512 jets, max err 0.0).
  q9/q64-u-lg-t250k-kd finished 3000 epochs: final model 80.62% test, avg rej 55.8 — tracing + Verilog for it.
- 11:00 — **Best so far (bit-exact): 80.62% test, avg rej 55.8, 146,481 LUT (8.5%), FF 104k, 13 stages, 43.3 ns, II=1,
  0 DSP/BRAM** — final model of q9/q64-u-lg-t250k-kd (3000 ep; snapshot runs/_traced/...__final.keras; Verilog under
  verilog/q9_q64_u_lg_t250k_kd_seed0__runs__q9__q64_u_lg_t250k_kd__seed0__final/). 0.3 pts from the N=64 target.
- 11:30 — Final models of the finished 3000-epoch per-slot runs (test set, da4ml @300 MHz):
  | N | test acc | avg rej | EBOPs | LUT | % | stages | ns |
  |---|---|---|---|---|---|---|---|
  | 64 (q9 t250k) | 80.62 | 55.8 | 194k | 146,481 | 8.5 | 13 | 43 |
  | 64 (q8 t200k) | 80.44 | 54.0 | 153k | 118,820 | 6.9 | 12 | 40 |
  | 32 (q8 t150k) | 78.26 | 33.7 | 134k | 99,530 | 5.8 | 13 | 43 |
  | 32 (q8 t100k) | 77.80 | 29.8 | 90k | 70,851 | 4.1 | 12 | 40 |
  Verilog + bit-exact running for the 118.8k and 99.5k designs. Still open: 80.9% at N=64 inside the budget (q11/q12).
- 11:50 — bit-exact confirmed (512 jets, max err 0.0) for the 80.44%/118.8k (N=64) and 78.26%/99.5k (N=32) designs.
  q8 t150k final: 79.94% @ 94,787 LUT (5.5%). q9 t300k final: 80.70% test (rej 57.5) — tracing (expected ≈170k LUT).
- 12:10 — **Best in-envelope: 80.75% test @ 168,426 LUT (9.7%), FF 116k, 13 stages, 43.3 ns** (q9/q64-u-lg-t300k-kd
  epoch 2936; neighbours 80.71–80.74% @ 168k; final 80.70% @ 170.4k). 0.15 pts from JEDI-Linear N=64 (80.9%).
  Verilog + bit-exact running. q10 lg5 (stronger teacher) final 80.52%.
- 12:40 — 80.75%/168.4k design Verilator-verified bit-exact (512 jets, max err 0.0). q11 fine-tune (320k target from
  the t250k checkpoint) final 80.56% test. Leaderboard page now shows load/render errors in the subtitle and retries
  data.json (user reported it stopped working on a phone, likely during a republish).

## 2026-10-07 ~13:30 — goal restated: BEAT JEDI-Linear (80.9% @ N=64) inside the envelope; we are at 80.75% @ 168k
- Levers launched: (1) **ensemble teacher** — mean log-softmax of 7 float models (b64-relu ×2, gmpsep, att32, p4,
  pT-sorted full, gmpch8) → runs/_seeds/ensemble7_train_logits.npy; (2) **more particles with per-slot weights**
  (f7 float: N=96/128 attention-free and N=128 full; q13 QAT N=64/96/128 at 250–330k EBOPs, 5000 epochs, 2 seeds,
  ensemble teacher, α 0.7, T 3); (3) component-cost trace of the 80.75% design to find what to prune.
- Site: a literal newline inside a JS string in the Pareto tooltip broke the whole page script (table blank);
  fixed; publish script now syntax-checks the page with `node --check` and reports the Pages build status;
  `.nojekyll` added after a failed Jekyll build.
- 14:10 — q10 330k-target finals: per-slot + 8-ch GMP 80.75% (rej 58.8) @ 182.5k LUT; 8×8 GMP 80.65–80.72% @
  180–183k LUT — both ≈254k EBOPs, just over the budget (LUT ≈ 0.72×EBOPs here → budget ≈ 240k EBOPs).
  Best inside the envelope stays 80.75% @ 168.4k (q9 300k-target, 238k EBOPs). N=96/128 per-slot attention-free
  float = 79.8–80.0% (no gain over N=64 in float; QAT q13 will tell whether the extra slots help after pruning).
- N=96 arms removed: HGQ QSum needs power-of-two scales (1/96, 1/12) → only N ∈ {32, 64, 128}.
- 14:40 — LUT breakdown of the 80.75%/168k design (per-slot attention-free): embed ≈25k, GMP ≈8k, token mixer 37.6k,
  msg proj 14.6k, **FFN 53.1k (32%)**, pooling/residual 23.9k, head 6.7k. Launched **f8** (float): FFN hidden 8 / no
  FFN, low-rank mixer (hidden 32/16), mean-token global, no hidden head, combos — to free LUT for a higher EBOPs target.
- 15:30 — f8 (per-slot attention-free, float, test): ref 80.03; FFN hidden 8 → 80.0 (free); no FFN → 79.8 (−0.2);
  low-rank mixer (hidden 32) → 80.22 (free/better); no hidden head → 79.8 (−0.2). Pruned form (FFN 8 + mixer 32)
  should drop ≈60k of the 168k LUT. Launched **q14**: pruned form, ensemble teacher, 5000 ep, targets 330k/380k/430k
  EBOPs (2 seeds) + a no-FFN arm at 380k. q11 fine-tunes finished at 80.5–80.6% test.
- 16:00 — f8: mean-token global 80.55 (> mixer 80.1), mixer hidden 16 80.3. Launched **q15**: per-slot attention-free + mean-token global + FFN 8, ensemble teacher, 5000 ep, targets 330k/400k, 2 seeds.

## 2026-10-07 ~16:30 — HGQ-LUT, deeper models, attention traced
- User: attention models had QAT but no hardware numbers → tracing all attention QAT runs (q1 full, q6/q8/q9
  attention-on-32 + mixer, q9 per-slot full attention, q8 N=32 full) — previously skipped because they need
  ≈800k–1.6M EBOPs for ≥80%.
- Added `--lut_layers` (HGQ-LUT, arXiv:2604.22293: `hgq.layers.table.QDenseT`, each edge a learned table, d_hl 8,
  table (6,5)) per role embed/ffn/mix/proj/head, and `--n_blocks` (latency is 40 ns of 100 ns → room for depth).
- Launched **q16** (10 HGQ-LUT QAT arms, from scratch, ensemble teacher, 3000 ep): attention-free per-slot with
  LUT on ffn+mix+head / +proj / +embed at 200–300k; mean-token; pruned; attention-on-32; Linformer; full attention;
  N=32. Launched **f9** float: 2- and 3-block versions of the winner, mean-token, pruned, attention-on-32.
- f9 (float, test): 2 blocks gives no gain — winner 80.08 (1 blk 80.10), mean-token 80.36 (1 blk 80.55), pruned 80.02 (1 blk 80.15), att32+mixer 80.47. Depth does not convert latency headroom into accuracy here; LUT stays the constraint.
- f9: **3 blocks 80.54% test** (1 blk 80.10, 2 blk 80.13) — first depth gain. Launched q17 (3-block QAT, float-init and scratch, 300k/400k, ensemble teacher) and f10 (4 blocks; 3-block mean-token and pruned).
- f9 3-block seed1 80.36 → 3 blocks = 80.45 ± 0.09 vs 1 block 80.10 ± 0.07 (+0.35, 2 seeds). q17 3-block QAT running.
- f10 (float, test): 4 blocks 80.50 (≈ 3 blocks 80.45; saturates), pruned 3-block 80.51 (1 blk 80.15), mean-token 3-block 80.32 (1 blk 80.55, depth hurts this form).

## 2026-10-07 ~18:30 — loop paused (usage limit)
State: GPU jobs still running on their own — q12 (5000-ep winner), q13 (ensemble teacher, N=64/128), q14 (pruned,
higher budgets), q15 (mean-token), q16 (HGQ-LUT ×10), q17 (3-block QAT). Helper Job anrunw-helper-1 is re-tracing the
attention-on-32 checkpoints (logs/trace_attn32b.log; slicing fixed to x[..., :K, :]). Best so far: 80.75% test @ 168,426
LUT, 13 stages, 43 ns, bit-exact (q9/q64-u-lg-t300k-kd ep2936). To resume: trace finished q13–q17 fronts with
scripts/trace_run.py, emit_verilog.py for any in-envelope point >80.75%, then publish_leaderboard.sh.
- 19:30 — attention-on-32 trace failure root cause: **HGQ2 0.1.9 tracer bug** — `ReplayMerge` handles
  `keras.layers.Concatenate` by stacking the (batch-less) inputs and calling np.concatenate(..., axis=layer.axis), so a
  positive axis is off by one (axis=1 concatenated features). Negative axes are correct → `attn_pad` now uses axis=-2
  (identical function, no weights). Re-tracing all att32 snapshots. Worth reporting upstream.

## 2026-10-07 ~20:00 — MATCHES JEDI-Linear inside the envelope
**q13/q64-u-lg-t330k-ens-5k seed0** (N=64, pT-sorted, per-slot weights, ReLU, attention-free, 8×8 GMP + token mixer,
distilled from a 7-model float ensemble, 5000 epochs, EBOPs target 330k):
| ckpt | test acc | AUC | avg rej (W/Z/t) | LUT | % VU13P | FF | stages | latency |
|---|---|---|---|---|---|---|---|---|
| ep 4377 | **80.89%** | 0.958 | 59.3 (79.6/80.5/17.9) | **170,414** | 9.9 | 114,518 | 13 | 43.3 ns |
| final (ep 5000) | 80.89% | 0.958 | 60.4 | 171,130 | 9.9 | 117,630 | 13 | 43.3 ns |
| ep 4731 | 80.85% | | | 168,804 | 9.8 | | 13 | 43 ns |
JEDI-Linear N=64 (pT-sorted, post-route): 80.9% @ 71k LUT, 61 ns. We match accuracy (−0.01) inside the CTL2 envelope
with lower latency (43 vs 61 ns) but 2.4× the LUTs. Verilator on the final model: argmax 100% (512 jets), max |err|
3.8e-6 (to check: Keras float32 rounding vs a real RTL mismatch — earlier designs were exactly 0).
- 20:30 — **80.89% / 170,414-LUT design (q13 ep4377) Verilator-verified bit-exact on 2000 jets (max err 0.0, argmax
  100%)**; the earlier 3.8e-6 on the final model is likely Keras float32 rounding. Pruned 430k-target (q14 seed1) reaches
  80.85–80.86% but at 224–230k LUT / 15 stages — outside. q12 5000-ep: 80.53% @ 128.2k LUT (7.4%), 12 stages.
  Attention-on-32 now traces (HGQ Concatenate bug worked around); results in results/attn32_traces.json and the leaderboard.
- 21:00 — q13 seed1 (same config as the 80.89% design): final 80.67% test → 80.78 ± 0.11 over 2 seeds (tracing seed1).
- **HGQ-LUT (q16) failure**: all per-slot (unshared) LUT arms never learned (val 20% from epoch 0, before compression);
  only lut3-att32 (shared weights) trained: front up to 80.93% val @ 970k EBOPs (tracing). Cancelled the dead arms;
  **q18** isolates the cause: LUT with shared weights, LUT with lr 1e-3, LUT on the head only.
- **3-block (q17) collapsed** past ~600k EBOPs (val 81% → 20% around epoch 700; β kept rising). Fronts saved up to 81.2%
  val; tracing. Cancelled; q18 retries 3-block with lr 1e-3 and β_max 3e-6.
- 22:00 — **Reproduced**: q13 seed1 final 80.68% @ 170,416 LUT, 13 stages (seed0 80.89% @ 170.4k) → **80.79 ± 0.11% @
  ≈170k LUT over 2 seeds, both inside the envelope**. Pruned form (q14 330k): 80.79–80.84% @ 177–186k (just over).
  Mean-token (q15 330k): 80.72% @ 194k. **3 blocks (q17 fronts): 80.95% @ 308.5k LUT (17.9%), 93 ns; 81.08% @ 393k,
  83 ns; 81.11% @ 500k** — beats JEDI accuracy, ~1.8× the LUT budget, within the latency limit.
  Launched **q19** (2 seeds each): pruned at 290k/305k EBOPs, winner at 340k, pruned 3-block at 250k/320k (stable lr).
- 22:45 — mean-token q15 400k seed1: **80.94% @ 219.8k LUT (12.7%), 14 stages, 47 ns** (final 80.92% @ 221k, rej 61.2); beats JEDI accuracy at 1.27× the LUT budget. N=128 per-slot finals 80.51/80.60 (no gain over N=64).
- 23:15 — **HGQ-LUT diagnosis (q18)**: shared weights + LUT (ffn,mix,head) trains (79.8% val @ 812k EBOPs, ep 297);
  per-slot + LUT stays at chance even with lr 1e-3 and even with LUT on the head only → the failure is the per-slot
  (QEinsumDense) embedding feeding QDenseT. Hypothesis: datalane wrap overflow from narrow initial bits. **q20** tests
  wider initial bits (i0_a 6/bw_a 8; i0_a 8/bw_a 10) + a shared-weight LUT arm at 200k. Pruned 380k final: 80.88% @
  205.8k LUT (11.9%), 15 stages. 3-block with lr 1e-3/β_max 3e-6 (q18) is stable so far: 81.12% val @ 647k EBOPs (ep 371).
- 23:50 — HGQ-LUT attention-on-32 (q16 lut3-att32, shared weights) traced: 80.10% @ 582k LUT, 24 stages (80 ns); 80.04% @ 720k — no better than the non-LUT att32 (80.12% @ 518k). Attention dominates; LUT dense layers don't change that.
- 00:20 — q20 per-slot+LUT with i0_a 6/bw_a 8 still at chance → overflow hypothesis refuted (cancelled; i0_a 8 arm pending as last check). Pruned 3-block (q19, lr 1e-3, β_max 3e-6) stable at 80.9–81.1% val, compressing through 480–550k EBOPs toward 250k/320k (≈260k EBOPs ↔ 173k LUT for 3-block).
- 01:00 — pruned 3-block traces: 80.85–81.0% @ 317–334k LUT, 28–29 stages (93–97 ns); ≈0.75 LUT/EBOP → 173k LUT needs ≈230k EBOPs. Launched **q21** (wp3 at 200k/225k, 2 seeds). Latency for 3 blocks is near the 100 ns cap (28–30 stages).
- 02:00 — **Pruned form inside the envelope**: q19/qp-t290k-ens-5k seed1 → 80.71% @ 157,756 LUT (9.1%), 15 stages (50 ns); 80.70% @ 157.3k; 80.65% @ 160.4k, 14 stages. Run still compressing (80.8% val at ~263k EBOPs).
- 02:50 — near-budget traces (test acc): pruned 290k s1 80.77% @ 160.9k LUT (14 stg), 80.76% @ 158.6k, 80.75% @ 157.6k; pruned 305k s0 80.71% @ 164.3k; winner 340k s1 80.71% @ 167.7k (13 stg). Val (80.93%) overstates test by ~0.15 for these points. Best inside the envelope remains 80.89% @ 170.4k.
- 03:20 — pruned 290k seed0 finished at 79.98% test (best val 80.02) vs seed1 80.77% → large seed variance (~0.8) at this budget; pruned form needs ≥3 seeds before claiming it.
- 03:40 — winner recipe 340k seed1 final: **80.76% @ 171,698 LUT (9.9%), 13 stages, 43 ns** (in envelope). Three in-envelope seeds of the per-slot attention-free model: 80.89 / 80.68 / 80.76 → **80.78 ± 0.09%** vs JEDI-Linear 80.9% (single number). Matches within ~0.1; best seed ties it.
- 03:55 — pruned 305k seed0 final: 80.73% @ 169.4k LUT (9.8%), 15 stages, 50 ns (in envelope). Pruned form is no better than the per-slot winner at the same LUT.

## 2026-10-08 — the real JEDI-linear bar, and the JEDI backbone in our pipeline
- Read the official JEDI-linear release (github.com/calad0i/JEDI-linear: official_models + Vivado post-route reports).
  Besides the pT-sorted models we were targeting (N=64 80.92% @ 70.7k LUT; N=32 78.00% @ 45.3k), it publishes
  **permutation-invariant** models (data-lane bit-widths shared across particles) that are inside the envelope and
  much stronger: **N=64 81.81% @ 163.9k LUT (24 cycles, ~78 ns achieved)**, **N=32 79.04% @ 135.9k LUT**.
  (16-feature models are higher still — 82.35% N=64 — but use 16 inputs per particle; not like-for-like.)
  → our best in-envelope (80.78 ± 0.09%, 3 seeds) is ~1.0 pt below the real 3-feature bar.
- JEDI 'gnn' = per-particle MLP(3→64) → s = MLP(φ), d = MLP(mean φ), h = s + d → MLP → sum-pool → MLP 64-32-16-5,
  BN folded (QEinsumDenseBatchnorm), weights SAT_SYM. Width 64 vs our d=16 is the obvious difference.
- Reference benchmarking: scripts/prep_hls4ml_raw.py streams the raw hls4ml 150p dataset (Zenodo 3602260) and keeps
  the 16 features of the leading 64 constituents; scripts/eval_jedi_official.py scores each official checkpoint on the
  260k test set with our metric code (acc, AUC, W/Z/t rejection @80% TPR) and traces it with our da4ml convention.
  Official models + summaries copied to /j-jepa-vol/phat-jet-aaron/jedi_official/. Leaderboard reference rows now
  carry post-route LUT/FF/latency + (once evaluated) AUC/rejection + our da4ml estimate.
- Added `--arch jedi` (phat_variants.build_jedi): the JEDI backbone in our pipeline, optional `--gmp grid` hybrid
  (PHAT-JeT GMP message from 16 channels added before the second per-particle MLP), `--jedi_per_slot_bits`.
  Launched **q22** (5000 ep, pT order, ensemble distillation unless 'nokd'): jd64 t230k (+ nokd control), jd64
  per-slot bits t230k, JEDI+GMP N=64 t230k/t200k, jd32 / JEDI+GMP N=32 t200k.
- 3-block (pruned) compressed traces: 80.51% @ 208.7k LUT (25 stg, 83 ns); 80.79% @ 242k; 80.43% @ 228k (q21 225k) —
  ~0.9–1.0 LUT/EBOP, so depth on the PHAT form cannot fit the envelope at useful accuracy. Remaining 3-block runs
  cancelled (checkpoints kept) to free GPUs for the JEDI-backbone line (q22).
- **Official JEDI-linear models benchmarked on our 260k test set** (our metric code; reproduce published acc within 0.02):
  3-feature N=64 perm-inv 81.81% / AUC 0.9620 / rej 80.4 (W 122 Z 101 t 18.6), post-route 163.9k LUT, 78 ns (da4ml ours: 158k, 12 stg);
  N=64 pT-sorted 80.94% / 0.9589 / 66.0, 70.7k LUT; N=32 perm-inv 79.04% / 0.9519 / 36.8, 135.9k; N=32 pT-sorted 78.00% / 0.9472 / 30.7, 45.3k.
  16-feature: N=64 perm-inv 82.35% / 83.2 rej (192k LUT post-route, out of envelope), N=64 81.78% @ 84k.
  Our best in-envelope (80.89%, rej 59.3) trails the perm-inv N=64 model by 0.9 pt accuracy and ~21 in rejection (W/Z mostly).

## 2026-10-08 — envelope loosened to latency only (user decision)
- Source check: the 10%-of-LUTs / 173k cap came only from the NeurIPS rebuttal ("Approximately 10% of the LUTs are
  available because Correlator Layer 2 must also perform jet clustering"); no CMS document found confirming it.
  JEDI-linear paper: "sub-100 ns latency and sub-10 ns initiation intervals", resources "no more than one SLR … or even
  less". CMS DeepSets L1T tagger (arXiv:2509.24371): 13% of VU13P LUTs, 234 ns tagging inside CTL2's 1 µs @ 360 MHz.
- **New envelope: latency < 100 ns (< 30 stages @ 300 MHz), II = 1; no LUT cap** (LUT reported as cost). Target:
  beat 3-feature JEDI-linear perm-inv 81.81% (N=64) / 79.04% (N=32). make_leaderboard/trace_run/loop_status/page updated.
- Added `--global_mode jedi` (JEDI interaction inside the PHAT block: x = relu(Ws x) + relu(Wd mean x)) and
  `--shared_bits` (perm-inv data-lane quantization for any model).
- Launched **f11** float screen (2 seeds): native Linformer d16/32/64, 2–3 blocks; Linformer inside the PHAT block
  (+GMP, mix / JEDI global, d16–64); PHAT without GMP + JEDI global (patch attention or none, d32/64); PHAT + GMP + JEDI
  global; full PHAT patch sizes 4/8/16/32 and d32; JEDI and JEDI+GMP float references.
- Launched **q24**: Linformer QAT with attention protection (+GMP d16, + shared bits, native) — first Linformer QAT
  since the protection fixes (earlier q9 Linformer QAT collapsed to 69.5%).

## 2026-10-08 — NRP compliance fix
- Audit against nrp.ai usage policy: helper was a Job running `sleep infinity` (bannable) and idle (0 of 16 CPU,
  0.2 of 48 GiB); every GPU job used ~0.5–0.8 of 4 CPUs and 2–4 of 24 GiB (<20%), GPU util often 0–5% (<40%).
- Fixed: helper → Deployment `anrunw-helper` (1 CPU/2 GiB, exempt); heavy CPU work → `cpu_job.sh` Jobs.
  make_var_jobs: up to 5 trainings run concurrently per GPU, requests == limits sized to use (1 CPU + 5 GiB per run),
  GPU list without A100/H100 (no access), T4 or 2080 Ti (11 GB). q22/q23/q24/f11 cancelled (q22/q23 ~2 h in) and
  relaunched as 9 GPU jobs (was 27); partial dirs moved to runs/_aborted/1008-0552 (not deleted).

## Backlog (launch as GPUs free up; always pack ≤5 runs per GPU — see NRP rules)
1. f11 winners → QAT (5000 ep, ensemble KD, latency-only targets 400k–1M EBOPs), N=64 and N=32, 2 seeds each.
2. PHAT patch-size / head-count QAT sweep around the best f11 PHAT form (p4/8/16, heads 1/2/4).
3. JEDI-backbone (q22/q23) winners: 2nd seed, width 96/128, N=32, higher EBOPs (no LUT cap), + per-slot bits.
4. Linformer: if q24 learns → k 4/8/16, width 32/64, 2 heads (user: 2 heads best), Linformer + PHAT block QAT.
5. Depth with the latency headroom: JEDI with 2 interaction rounds; PHAT-JEDI-global 2–3 blocks.
6. Stronger teacher: re-build the distillation ensemble from the best wide f11 float models.
7. Every new best: da4ml trace (cpu_job.sh) → Verilog + Verilator bit-exact → leaderboard.
- 07:55 — f11 first results (float, test): **full PHAT d32 p8 2 heads 81.65 / 81.68%** (best float of the project;
  d16 was 81.3–81.4), JEDI float 81.64%, native Linformer d16 80.10/80.06 (d32 ~81.4 val, d64 ~81.2 val mid-run),
  Linformer+PHAT d32 mix 80.68/80.85. Queued **q25** (1 packed job): QAT of PHAT d32 p8 h2 (full and attention-on-32,
  1M EBOPs), native Linformer d32 (600k) / d64 (1M), JEDI at 600k — latency is the test (full PHAT attention was 130 ns at d16).
- 08:30 — f11 more: **JEDI+GMP 81.80%** (81.82/81.77, rej 71.2; JEDI alone 81.62), **PHAT+GMP+JEDI-global attention-free d32 81.71%** (6.3k params, rej 70.3), PHAT no-GMP JEDI-global d64 81.53, native Linformer d32 81.35. Queued **q26**: QAT of PHAT+GMP+JEDI-global d32 (400k/800k, +shared bits) and JEDI+GMP (400k/800k).
- 09:00 — f11: Linformer+PHAT block + GMP + JEDI global d32 **81.78%** (81.88/81.68, rej 70.9); PHAT+GMP+JEDI-global with patch attention d32 81.63; Linformer+PHAT d16 mix 81.07. Pattern: GMP + JEDI global interaction is what matters (81.7–81.8 with or without Linformer/attention); attention-free version (q26) is the latency-friendly QAT candidate; Linformer+GMP+JEDI-global QAT goes in the next batch if q26 confirms.
- 09:35 — **Linformer QAT collapsed again (q24)**, all 3 arms → chance (20.3%) as EBOPs fell below ~1–2M (from scratch,
  bits 7/7/0/0, attention protection on); val was 0.78–0.79 at 2–3M EBOPs. q24 cancelled (dirs kept). Rebuilt q25
  (not started) as **float-init QAT** from the f11 float checkpoints (wide initial bits 10/10/3/8): PHAT d32 p8 h2
  (full / attention-on-32), native Linformer d32, Linformer+GMP+JEDI-global d32, + JEDI 600k (scratch).
- f11 more: full PHAT d32 p8 (4 heads) **81.82%** (seed0); patch 16 81.45, patch 32 81.37 (d16); native Linformer d64
  ~81.65 val mid-run; 2–3-block native Linformer ~81 val mid-run.
- 10:40 — q23/q25/q26 all landed on k8s-3090-01.usd.edu and failed UnexpectedAdmissionError before starting (bad GPU node, no work lost). Node excluded; resubmitted as q23-00-r2 / q25-00-r2 / q26-00-r2.
- 13:30 — **Linformer QAT root cause (diag_minmax.py, CPU job).** Float-init Linformer QAT also collapsed (q25 nl-d32:
  81% → 80.5% down to ~3.8M EBOPs, then 72.6% @ 2.2M, chance @ 2.0M). Re-fitting activation integer ranges with
  hgq trace_minmax makes compressed checkpoints *worse* (→ chance below ~4M EBOPs) and reloaded checkpoints disagree
  with their training-time val (e.g. saved 80.53% → reload 33.3%; 78.4% → 49.3%): the compressed Linformer relies on
  **wrap-around overflow** of activation quantizers and is numerically fragile (GPU TF32 vs CPU float32 flips wraps).
  Fix under test: `--datalane_overflow SAT` (saturating activations). Killed only the dead nl-d32 process in
  q25-00-r3 (PHAT attention runs there are healthy, 80.3–80.6% val).
  Caution for every attention model: check reload/test == training val and Verilator bit-exactness before trusting.
- 14:10 — SAT build/trace check passed. Launched **q27** (1 job, 4 runs): float-init QAT with saturating activations — native Linformer d32 (600k) / d64 (1M), Linformer+GMP+JEDI-global d32 (800k), PHAT d32 p8 h2 (1M; SAT check vs the wrap-mode q25 run).
- 14:45 — Wrap-overflow collapse is broader than Linformer: PHAT attention d32 float-init (q25) 43.6% @ 779k and
  attention-on-32 20.3% @ 624k; Linformer+GMP+JEDI-global (wrap) 59% @ 1.9M; attention-free PHAT+GMP+JEDI-global with
  *shared* bits 42.3% @ 720k (per-element-bit version healthy 80.2–80.4% @ 0.7–0.9M). Cancelled q25-00-r3 and stopped
  the two collapsing processes in q25-01-r3 / q26-00-r3 (dirs kept). SAT versions are queued in q27.
  Healthy compressed QAT: **JEDI+GMP 81.18% val @ 509k EBOPs** (q26), JEDI 81.03% @ 514k (q25), JEDI+GMP 80.72–80.74%
  @ 288–360k, JEDI no-KD 80.86% @ 224k (KD run drifting down to 78.3%).
- 15:40 — Reload check (diag_minmax, CPU job): JEDI-family Pareto ckpts reload within +0.05–0.10 of saved val and are unchanged by trace_minmax → robust (the wrap fragility is specific to attention / Linformer / shared-bit models). Best: JEDI per-particle bits 81.41–81.47% val @ 372k EBOPs, 81.31–81.39% @ 226–261k; JEDI 600k 81.40–81.45% @ 509–512k. Tracing them (cpu_job trace-jedi).
- 16:30 — **Synthesized JEDI-family QAT (our pipeline, test):** JEDI per-particle bits 81.04–81.09% @ 142–181k LUT (12 stg, 40 ns), 81.26% @ 275k (47 ns); JEDI 600k 81.24–81.37% @ 325–344k (43 ns); JEDI+GMP 81.22% @ 235–238k (43 ns). Plateau ~81.3% vs official JEDI perm-inv 81.81% @ 164k — and our float JEDI is only 81.62% → suspect the input pipeline. Added `--data hls4ml` (JEDI's exact inputs; standardized eta/phi span ±3.8) to train/eval/verilog; launched **f12** float test (JEDI, JEDI+GMP, PHAT+GMP+JEDI-global d32, PHAT d32 p8; 2 seeds).
- 17:00 — JEDI+GMP traced too: 81.22–81.29% @ 235–285k LUT (43–47 ns), 81.26% @ 401k. Leaderboard builder taught the new options (JEDI backbone, JEDI global, JEDI inputs, SAT, shared bits, float-init); published 228 models.
- 18:00 — **Input pipeline is the gap (f12, float, mid-run val):** on JEDI's exact inputs JEDI reaches 82.24 / 81.87%
  val at epoch ~160 (our inputs: 81.62% test), PHAT+GMP+JEDI-global d32 82.24% at epoch ~210 (ours 81.71%), JEDI+GMP
  82.06% at epoch ~100 (ours 81.80%). ~+0.4–0.6 pt → explains our ~81.3% QAT plateau vs official 81.81%. f12 first
  attempt OOM'd (loader cast 16 features to float32) → fixed (select 3 cols first), relaunched as f12-00-r2.
  If test confirms: move all QAT to `--data hls4ml` (no distillation until teacher logits exist for that order).
- Stopped 3 fully collapsed QAT processes (0 EBOPs at epoch ~3270: q22 jd64 ens/nokd, jd32); their earlier Pareto
  ckpts are kept and already traced where good.
- q27 (SAT) early: Linformer+GMP+JEDI 80.62% @ 3.2M, Linformer d64 80.67% @ 4.3M, PHAT d32 80.42% @ 3.3M; Linformer
  d32 69.8% @ 2.4M (degrading even with SAT — watching).
- 18:40 — **f12 test (float, JEDI's exact inputs): JEDI+GMP 82.59 / 82.68%, JEDI 82.39 / 82.33%, PHAT+GMP+JEDI-global
  d32 82.30%** (same archs on our inputs: 81.80 / 81.62 / 81.71). Input pipeline = +0.6–0.9 pt; GMP adds +0.25 over JEDI.
  → switched QAT to `--data hls4ml`: cancelled the 6 old-input QAT jobs (ckpts kept), launched **q28** (2 jobs, 8 runs,
  float-init from f12): JEDI+GMP 230k/400k/700k, JEDI 230k (vs official 81.81% @ 227k), JEDI per-slot bits 400k,
  PHAT+GMP+JEDI-global d32 600k, N=32 JEDI+GMP 200k / JEDI 170k (scratch). Building a JEDI-input ensemble teacher
  (f12 JEDI+GMP ×2 + JEDI ×2 → runs/_seeds/ensemble4_hl_train_logits.npy) for a distilled follow-up stage.
- 19:10 — Teacher ensemble4_hl built (f12 JEDI+GMP 82.59/82.68 + JEDI 82.39/82.33, test). Launched **q29** (1 job, 5 runs): distilled versions of the main q28 arms (JEDI+GMP 230k/400k, JEDI 230k, PHAT+GMP+JEDI-global d32 600k, N=32 JEDI+GMP 200k).
- 19:45 — **Float-init failed for every JEDI-arch run in q28/q29** ("transfer collapsed"): float JEDI keeps BN as
  separate `*_bn` layers while the Q model folds BN into QEinsumDenseBatchnorm, so the name-matched transfer drops BN
  params. PHAT-arch float-init works (PHAT+GMP+JEDI-global d32 81.10% val @ 1.05M; distilled 81.79% early).
  Relaunched the 8 JEDI-arch arms from scratch as **q30** (JEDI+GMP 230k/400k/700k, JEDI 230k, per-slot 400k, + KD
  JEDI+GMP 230k/400k, JEDI 230k). TODO: fold float BN into the QEinsumDenseBatchnorm transfer.
- 20:30 — Fixed float-init for the JEDI backbone: transfer_weights folds float BN into QEinsumDenseBatchnorm (kernel=W, bias=β, γ, mean=μ−b, var=var+1e-3−keras ε); verified on f12 JEDI+GMP: 81.19% right after transfer (float 82.37%, threshold 90%). q30 (scratch) had already started → kept; added **q31** (float-init JEDI+GMP 230k/400k, + KD 230k, JEDI KD 230k) for a scratch-vs-float-init comparison.
- 21:10 — **Linformer does not survive QAT even with saturating activations** (q27): Linformer+GMP+JEDI-global SAT
  66% @ 2.0M, native Linformer d64 SAT 41% @ 2.6M, d32 25% (stopped). PHAT patch attention with SAT is fine
  (79.98% @ 1.84M) — so the failure is specific to Linformer's softmax over sequence-projected keys. Linformer stays a
  float-only result (best float: native d64 81.58%, Linformer+GMP+JEDI-global d32 81.78% on our inputs).
- JEDI-input QAT early (val): JEDI 230k distilled 81.71% @ 745k EBOPs (ep 281, scratch); JEDI float-init distilled
  81.68% @ 2.4M; JEDI per-slot 81.58% @ 853k; JEDI+GMP 400k distilled 81.20% @ 1.3M; still compressing.
- 22:50 (18:50 EDT 14:50) — **First designs above JEDI-linear inside the latency envelope (JEDI's inputs, synthesized):**
  JEDI backbone distilled (q30 jd64-hl-t230k-kd ep223) **82.25% test @ 610.7k LUT, 14 stg, 46.6 ns**; JEDI+GMP
  float-init (q31 jdg64-hl-fi2-t400k ep226) 81.86% @ 637.9k, 15 stg, 50 ns; JEDI distilled ep536 81.67% @ 360.8k, 43 ns.
  (JEDI-linear perm-inv 81.81% @ 163.9k, 78 ns.) Verilog + Verilator bit-exact running for the 82.25% design.
  Page: "in envelope" is now purely synthesized latency < 100 ns; "beats JEDI-linear" is a separate badge.
  Attention collapse: diag_attn.py running on collapsed PHAT / Linformer ckpts (score/softmax/AV lanes had no floor).
  User: publish the page at 15:15 EDT with everything benchmarked by then (trace-hl1, trace-hl2, verilog jobs).
- 23:10 — **Attention-collapse diagnosis (diag_attn.py).** PHAT d32 (q25, wrap, attention-protected): healthy ckpt
  81.95%, compressed 63%: Q/K/V weights pruned to ~0–1 bits, Q/K data lanes at f = −8 (values in steps of 256),
  QK scores f = −11, local softmax output ≈ 0 everywhere (mean max weight 0.003 vs uniform 0.125) → attention output
  constant (≈0); patch-token softmax exactly uniform (0.119 vs 0.125). Linformer d64 (q27, SAT): attention locked onto
  one key (mean max 0.85), F-projection outputs exploding (std 34) — also input-independent.
  Root cause: `--attn_floor` set a MinMax on the *output* quantizer of the Q/K/V/O Dense, but HGQ quantizes layer
  *inputs*, so the floor never bound (Q lane reached f = −8 under a floor of 3). Fix: floors on the input quantizers of
  every attention sub-layer (projections, QK einsum, softmax incl. exp/inverse tables, AV einsum, Linformer E/F) +
  a minimum on attention weight bits.
- 23:15 — **Verilator bit-exact verified** for the 82.25% design (q30 jd64-hl-t230k-kd ep223): 2000 test jets, max|err| 0.0, argmax 100%; Verilog in verilog/q30_jd64_hl_t230k_kd_seed0__…epoch_223…; test 82.25%, avg rej 93.3 (JEDI-linear perm-inv 80.4), 610.7k LUT, 14 stages, 46.6 ns, II=1. First verified design above JEDI-linear inside the latency envelope. Also: JEDI per-slot bits (no KD) 82.10% @ 474k, 47 ns; JEDI+GMP float-init 81.86% @ 638k, 50 ns.
