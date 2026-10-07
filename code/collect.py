"""Aggregate runs/<stage>/<config>/seed*/result.json -> table (md + csv) and plots."""
import glob
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/j-jepa-vol/phat-jet-aaron"
stage = sys.argv[1] if len(sys.argv) > 1 else "s1"
out = f"{ROOT}/plots/{stage}"
os.makedirs(out, exist_ok=True)

rows = []
for d in sorted(glob.glob(f"{ROOT}/runs/{stage}/*/")):
    name = os.path.basename(d.rstrip("/"))
    res = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/seed*/result.json"))]
    hists = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/seed*/history.json"))]
    if not res and not hists:
        continue
    r0 = res[0] if res else {}
    G = None if not r0.get("edges") else len(r0["edges"]) - 1
    if r0.get("gmp") == "dynamic":
        G = "dyn"
    # static-GMP op count as in the FPGA port (one-hot scatter + gather: 2*N*G^2*C, dw conv: k^2*G^2*C),
    # N=64 (HW operating point), C=16; relative to core7 (G=7, k=3)
    k_ = r0.get("cpe_k") or 3
    rel = ((2 * 64 + k_ ** 2) * G ** 2) / ((2 * 64 + 9) * 49) if isinstance(G, int) else None
    row = dict(name=name, n=len(res), G=G, rel_cost=rel, k=r0.get("cpe_k"), params=r0.get("params"),
               best_val_acc=[max(h["val_accuracy"]) for h in hists], epochs=[len(h["loss"]) for h in hists])
    for key in ["test_acc", "test_auc", "avg_bg_rej"]:
        v = np.array([r[key] for r in res])
        row[key] = (float(v.mean()), float(v.std())) if len(v) else (np.nan, np.nan)
    rows.append(row)

rows.sort(key=lambda r: -np.nan_to_num(r["test_acc"][0], nan=-1))
lines = ["| config | G×G cells | GMP cost vs core7 | k | params | seeds done | test acc (%) | AUC | avg bg rej @0.8 | best val acc (running) | epochs |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    a, s = r["test_acc"]
    rc = "-" if r["rel_cost"] is None else f"{r['rel_cost']:.1f}x"
    lines.append(f"| {r['name']} | {r['G']} | {rc} | {r['k']} | {r['params']} | {r['n']} | {100*a:.2f} ± {100*s:.2f} | "
                 f"{r['test_auc'][0]:.4f} ± {r['test_auc'][1]:.4f} | {r['avg_bg_rej'][0]:.1f} ± {r['avg_bg_rej'][1]:.1f} | "
                 f"{', '.join('%.4f' % v for v in r['best_val_acc'])} | {r['epochs']} |")
open(f"{out}/table.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

done = [r for r in rows if r["n"] and isinstance(r["G"], int)]
if done:
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for r in done:
        cells = r["G"] ** 2
        for i, key in enumerate(["test_acc", "avg_bg_rej"]):
            m, s = r[key]
            ax[i].errorbar(cells, m * (100 if i == 0 else 1), yerr=s * (100 if i == 0 else 1), fmt="o", capsize=3)
            ax[i].annotate(r["name"], (cells, m * (100 if i == 0 else 1)), fontsize=7, xytext=(4, 2), textcoords="offset points")
    for r in rows:
        if r["G"] in ("dyn", None) and r["n"]:
            for i, key in enumerate(["test_acc", "avg_bg_rej"]):
                m, s = r[key]
                sc = 100 if i == 0 else 1
                ax[i].axhspan((m - s) * sc, (m + s) * sc, alpha=0.15, color="C3" if r["G"] == "dyn" else "gray")
                ax[i].axhline(m * sc, ls="--", lw=0.8, color="C3" if r["G"] == "dyn" else "gray")
                ax[i].text(1.05 * 4, m * sc, r["name"], fontsize=7, va="bottom")
    ax[0].set_ylabel("test accuracy (%)"); ax[1].set_ylabel("avg background rejection @ 80% TPR")
    for a_ in ax:
        a_.set_xscale("log"); a_.set_xlabel("static grid cells G² (GMP hardware cost ∝ G²)"); a_.grid(alpha=0.3)
    fig.suptitle(f"PHAT-JeT static GMP grids ({stage}); bands = dynamic grid / no-GMP references")
    plt.tight_layout(); plt.savefig(f"{out}/acc_vs_cells.png", dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(9, 5))
for i, r in enumerate(rows):
    for j, f in enumerate(sorted(glob.glob(f"{ROOT}/runs/{stage}/{r['name']}/seed*/history.json"))):
        h = json.load(open(f))
        ax.plot(h["val_accuracy"], color=f"C{i % 10}", lw=0.8, label=r["name"] if j == 0 else None)
ax.set_xlabel("epoch"); ax.set_ylabel("val accuracy"); ax.set_ylim(0.78, 0.82); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{out}/val_curves.png", dpi=130); plt.close()
