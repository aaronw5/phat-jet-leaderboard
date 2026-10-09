"""Load a QAT checkpoint (.keras, HGQ2) from the July campaign, re-evaluate its accuracy on the
held-out set, and optionally trace it with da4ml at the CTL2 convention (300 MHz, hard_dc=2).

  env/bin/python eval_ckpt.py <ckpt.keras> [--trace] [--out results.json]

Data is chosen from the checkpoint's input length: N=128 -> jets_128x3_aaron_ourunits_kt.npz,
N<=150 -> jets_150x3_kt.npz sliced to the first N (kt-sorted head), both in paper units.
Reads only; writes to --out (default under /j-jepa-vol/phat-jet-aaron/results).
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("KERAS_BACKEND", "jax")
import numpy as np
import keras
import hgq.layers  # noqa: F401  (registers Q* classes so load_model can deserialize campaign/own checkpoints)

FPGA = "/j-jepa-vol/phatjet-fpga"
CLOCK_NS, LATENCY_CUTOFF, PART = 3.33, 4.0, "xcvu13p-flga2577-2-e"

p = argparse.ArgumentParser()
p.add_argument("ckpt")
p.add_argument("--trace", action="store_true")
p.add_argument("--kif", default="1,8,16", help="input fixed-point (keep,int,frac); campaign used 1,8,16")
p.add_argument("--n_eval", type=int, default=0, help="subsample the val set (0 = all 260k)")
p.add_argument("--out", default="/j-jepa-vol/phat-jet-aaron/results/eval_ckpt.json")
a = p.parse_args()

t0 = time.time()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phat_variants import build_q_from_args, build_variant, kw_from_args, load_q_model

run_dir = os.path.dirname(a.ckpt) if os.path.basename(os.path.dirname(a.ckpt)) != "pareto" else os.path.dirname(os.path.dirname(a.ckpt))
cfg_path = os.path.join(run_dir, "config.json")
if os.path.basename(run_dir) == "_traced":  # snapshot naming: <run_tag>__<ckpt>.keras with <run_tag>__config.json
    cfg_path = os.path.join(run_dir, os.path.basename(a.ckpt).rsplit("__", 1)[0] + "__config.json")
cfg = json.load(open(cfg_path))  # lambda GMP indicators: rebuild, then load_weights
if cfg.get("mode") == "qat":
    m = load_q_model(cfg, a.ckpt)
else:
    m = build_variant(quantized=False, **kw_from_args(cfg))
    m.load_weights(a.ckpt)
N = m.inputs[0].shape[1]
from phat_variants import load_split
f = (f"{cfg.get('sort', 'kt')}-sorted hls4ml150p (JEDI-linear inputs)" if cfg.get("data") == "hls4ml"
     else f"{cfg.get('sort', 'kt')}-sorted jets_150x3_kt.npz")
from phat_variants import use_data
use_data(cfg)
x, y = load_split("val", N, cfg.get("sort", "kt"))
if a.n_eval:
    x, y = x[: a.n_eval], y[: a.n_eval]
print(f"loaded {os.path.basename(a.ckpt)}: N={N}, params={m.count_params()}, data={os.path.basename(f)} ({time.time()-t0:.0f}s)")

pr = m.predict(x, batch_size=4096, verbose=0)
from phat_variants import evaluate
ev = evaluate(pr, y)
acc = ev["test_acc"]
res = dict(ckpt=a.ckpt, N=int(N), params=int(m.count_params()), n_eval=int(len(x)), acc=acc, **ev)
try:
    from hgq.utils.sugar import FreeEBOPs  # noqa: F401
    res["ebops"] = float(sum(float(l.ebops) for l in m.layers if hasattr(l, "ebops")))
except Exception as e:  # pragma: no cover
    res["ebops_err"] = str(e)
print(f"accuracy {acc*100:.2f}%  ebops {res.get('ebops')}")

if a.trace:
    from da4ml.converter import trace_model
    from da4ml.trace import comb_trace, HWConfig, to_pipeline

    kif = tuple(int(v) for v in a.kif.split(","))
    t1 = time.time()
    inp, out = trace_model(m, solver_options={"hard_dc": 2}, hwconf=HWConfig(1, -1, -1), inputs_kif=kif)
    sol = comb_trace(inp, out)
    psol = to_pipeline(sol, latency_cutoff=LATENCY_CUTOFF)
    res.update(lut_est=float(psol.cost), ff_est=int(psol.reg_bits), stages=len(psol.solutions),
               latency_ns=len(psol.solutions) * CLOCK_NS, lut_pct_vu13p=100 * float(psol.cost) / 1_728_000,
               comb_lut=float(sol.cost), kif=kif, trace_s=time.time() - t1)
    print(f"LUT {res['lut_est']:,.0f} ({res['lut_pct_vu13p']:.1f}% VU13P)  FF {res['ff_est']:,}  "
          f"stages {res['stages']}  latency {res['latency_ns']:.1f} ns  ({res['trace_s']:.0f}s)")

os.makedirs(os.path.dirname(a.out), exist_ok=True)
rows = json.load(open(a.out)) if os.path.exists(a.out) else []
rows = [r for r in rows if r.get("ckpt") != a.ckpt] + [res]
json.dump(rows, open(a.out, "w"), indent=1)
