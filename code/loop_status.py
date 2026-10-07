"""Research-loop status: aggregate every runs/<stage>/<config>/seed*/ into a table + plots.

  env/bin/python loop_status.py            -> plots/loop/status.md, float_screen.png, qat_fronts.png
Float runs: best/final val acc per config (mean ± std over seeds), params.
QAT runs:   accuracy-vs-EBOPs Pareto fronts from pareto/ filenames + latest log.csv row.
"""
import csv
import glob
import json
import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/j-jepa-vol/phat-jet-aaron"
OUT = f"{ROOT}/plots/loop"
os.makedirs(OUT, exist_ok=True)
lines = []


def last_csv_row(p):
    try:
        rows = list(csv.DictReader(open(p)))
        return rows[-1] if rows else None, len(rows)
    except Exception:
        return None, 0


# ---------------------------------------------------------------- float stages
flo = {}
for cfgdir in sorted(glob.glob(f"{ROOT}/runs/f*/*/")):
    stage, name = cfgdir.rstrip("/").split("/")[-2:]
    seeds = []
    for sd in sorted(glob.glob(cfgdir + "seed*/")):
        r = os.path.join(sd, "result.json")
        if os.path.exists(r):
            seeds.append(json.load(open(r)) | {"done": True})
        else:
            row, n = last_csv_row(os.path.join(sd, "log.csv"))
            if row:
                va = [float(x) for x in [row.get("val_accuracy") or row.get("val_acc_metric") or 0]]
                seeds.append(dict(best_val=np.nan, final_val=va[0], params=None, done=False, epoch=n))
    if seeds:
        flo[f"{stage}/{name}"] = seeds
if flo:
    lines.append("## Float screen\n\n| config | seeds | best val acc (%) | running/last val (%) | params |\n|---|---|---|---|---|")
    pts = []
    for k, seeds in sorted(flo.items(), key=lambda kv: -np.nanmean([s["best_val"] for s in kv[1]] or [0])):
        b = np.array([s["best_val"] for s in seeds if s["done"]], float)
        f = np.array([s["final_val"] for s in seeds], float)
        done = sum(s["done"] for s in seeds)
        bs = f"{100*b.mean():.2f} ± {100*b.std():.2f}" if len(b) else "–"
        lines.append(f"| {k} | {done}/{len(seeds)} | {bs} | {100*np.nanmean(f):.2f} | {seeds[0].get('params')} |")
        if len(b):
            pts.append((k, 100 * b.mean(), 100 * b.std(), seeds[0].get("params") or 0))
    if pts:
        fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(pts))))
        pts.sort(key=lambda p: p[1])
        ax.barh([p[0] for p in pts], [p[1] for p in pts], xerr=[p[2] for p in pts], color="C0", alpha=0.8)
        ax.set_xlim(min(p[1] for p in pts) - 1, max(p[1] for p in pts) + 0.6); ax.set_xlabel("best val accuracy (%)")
        ax.axvline(80.9, c="g", ls="--", lw=1); ax.text(80.9, -0.6, " JEDI N=64", color="g", fontsize=8)
        ax.axvline(78.0, c="g", ls=":", lw=1); ax.text(78.0, -0.6, " JEDI N=32", color="g", fontsize=8)
        ax.grid(alpha=0.3, axis="x"); ax.set_title("Float screen (mean ± std over seeds)")
        plt.tight_layout(); plt.savefig(f"{OUT}/float_screen.png", dpi=130); plt.close()

# ---------------------------------------------------------------- qat stages
qat = {}
for cfgdir in sorted(glob.glob(f"{ROOT}/runs/q*/*/")):
    stage, name = cfgdir.rstrip("/").split("/")[-2:]
    for sd in sorted(glob.glob(cfgdir + "seed*/")):
        front = []
        for f in glob.glob(os.path.join(sd, "pareto", "*.keras")):
            m = re.search(r"epoch=(\d+)-val_acc=([0-9.]+)-ebops=([0-9]+(?:\.[0-9]+)?)", os.path.basename(f))
            if m:
                front.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
        row, n = last_csv_row(os.path.join(sd, "log.csv"))
        qat[f"{stage}/{name}/{os.path.basename(sd.rstrip('/'))}"] = dict(front=sorted(front, key=lambda t: t[2]), last=row, epochs=n)
if qat:
    lines.append("\n## QAT runs\n\n| run | epochs | last val acc (%) | last EBOPs | beta | best acc on front (ebops) | cheapest on front ≥79% | ≥80% | ≥80.9% |\n|---|---|---|---|---|---|---|---|---|")
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (k, v) in enumerate(sorted(qat.items())):
        fr = v["front"]; row = v["last"] or {}
        acc_key = next((c for c in ("val_accuracy", "val_acc_metric") if c in row), None)
        la = f"{100*float(row[acc_key]):.2f}" if acc_key else "–"
        eb = f"{float(row['ebops']):.3g}" if row.get("ebops") else "–"
        bt = f"{float(row['beta']):.2g}" if row.get("beta") else "–"
        def cheapest(th):
            c = [t for t in fr if t[1] >= th]
            return f"{min(c, key=lambda t: t[2])[2]:.3g}" if c else "–"
        best = max(fr, key=lambda t: t[1]) if fr else None
        lines.append(f"| {k} | {v['epochs']} | {la} | {eb} | {bt} | {100*best[1]:.2f} ({best[2]:.3g}) | {cheapest(0.79)} | {cheapest(0.80)} | {cheapest(0.809)} |" if best else f"| {k} | {v['epochs']} | {la} | {eb} | {bt} | – | – | – | – |")
        if fr:
            ax.plot([t[2] for t in fr], [100 * t[1] for t in fr], "o-", ms=3, lw=1, label=k, color=f"C{i%10}")
    ax.set_xscale("log"); ax.set_xlabel("EBOPs"); ax.set_ylabel("val accuracy (%)"); ax.axhline(80.9, c="g", ls="--", lw=1); ax.axhline(78.0, c="g", ls=":", lw=1)
    ax.grid(alpha=0.3); ax.legend(fontsize=7); ax.set_title("QAT Pareto fronts (val acc vs EBOPs)")
    plt.tight_layout(); plt.savefig(f"{OUT}/qat_fronts.png", dpi=130); plt.close()

# ---------------------------------------------------------------- traced rows (ours)
tr = f"{ROOT}/results/eval_ckpt.json"
if os.path.exists(tr):
    rows = [r for r in json.load(open(tr)) if r.get("lut_est")]
    if rows:
        lines.append("\n## Traced (da4ml, 300 MHz, hard_dc=2)\n\n| ckpt | N | acc (%) | LUT | % VU13P | stages | latency ns | in envelope |\n|---|---|---|---|---|---|---|---|")
        for r in sorted(rows, key=lambda r: r["lut_est"]):
            ok = r["lut_est"] <= 172800 and r["stages"] < 30
            lines.append(f"| {os.path.basename(r['ckpt'])[:60]} | {r['N']} | {100*r['acc']:.2f} | {r['lut_est']:,.0f} | {r['lut_pct_vu13p']:.1f} | {r['stages']} | {r['latency_ns']:.1f} | {'YES' if ok else 'no'} |")

open(f"{OUT}/status.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
