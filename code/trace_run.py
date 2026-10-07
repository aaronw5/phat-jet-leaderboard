"""Trace the Pareto checkpoints of a QAT run with da4ml (CPU) and record LUT/FF/stages/latency.

  env/bin/python trace_run.py <run_dir> [--max 6] [--min_acc 0.78]

Picks, from <run_dir>/pareto/, the checkpoints with val_acc >= --min_acc, then the --max of them spread
evenly in EBOPs (always including the cheapest and the most accurate), evaluates each on the full held-out
set and traces it at the CTL2 convention (300 MHz, hard_dc=2, inputs kif 1,8,16).
Appends rows to <run_dir>/trace.json and to /j-jepa-vol/phat-jet-aaron/results/eval_ckpt.json.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("run_dir")
p.add_argument("--max", type=int, default=6)
p.add_argument("--min_acc", type=float, default=0.0)
p.add_argument("--n_eval", type=int, default=0)
a = p.parse_args()

cks = []
for f in glob.glob(os.path.join(a.run_dir, "pareto", "*.keras")):
    m = re.search(r"epoch=(\d+)-val_acc=([0-9.]+)-ebops=([0-9]+(?:\.[0-9]+)?)", os.path.basename(f))
    if m and float(m.group(2)) >= a.min_acc:
        cks.append((float(m.group(3)), float(m.group(2)), f))
cks.sort()
if not cks:
    print("no checkpoints above min_acc"); sys.exit(0)
done = set()
tj = os.path.join(a.run_dir, "trace.json")
if os.path.exists(tj):
    done = {r["ckpt"] for r in json.load(open(tj)) if r.get("lut_est")}
if len(cks) > a.max:
    idx = sorted(set([0, len(cks) - 1] + [round(i * (len(cks) - 1) / (a.max - 1)) for i in range(a.max)]))
    cks = [cks[i] for i in idx]
py = sys.executable
here = os.path.dirname(os.path.abspath(__file__))
for eb, acc, f in cks:
    if f in done:
        print("skip (traced)", os.path.basename(f)); continue
    print(f"== tracing {os.path.basename(f)} (ebops {eb:.3g}, val {acc:.4f})", flush=True)
    cmd = [py, os.path.join(here, "eval_ckpt.py"), f, "--trace", "--out", tj]
    if a.n_eval:
        cmd += ["--n_eval", str(a.n_eval)]
    subprocess.run(cmd, check=False)
# merge into the global results file
rows = json.load(open(tj)) if os.path.exists(tj) else []
g = "/j-jepa-vol/phat-jet-aaron/results/eval_ckpt.json"
allrows = json.load(open(g)) if os.path.exists(g) else []
allrows = [r for r in allrows if r.get("ckpt") not in {x["ckpt"] for x in rows}] + rows
json.dump(allrows, open(g, "w"), indent=1)
for r in sorted(rows, key=lambda r: r.get("lut_est", 1e12)):
    if r.get("lut_est"):
        ok = r["lut_est"] <= 172800 and r["stages"] < 30
        print(f"{os.path.basename(r['ckpt'])[:55]:55s} acc {100*r['acc']:.2f}  LUT {r['lut_est']:>11,.0f} ({r['lut_pct_vu13p']:.1f}%)  stages {r['stages']:3d}  {r['latency_ns']:.0f} ns  {'IN ENVELOPE' if ok else ''}")
