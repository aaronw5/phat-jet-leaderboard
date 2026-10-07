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
