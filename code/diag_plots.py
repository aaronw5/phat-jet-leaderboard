"""Diagnostics that need no Keras: (1) hardware frontier of all traced designs, (2) pT-rank physics."""
import glob
import json
import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/j-jepa-vol/phat-jet-aaron/plots/diag"
os.makedirs(OUT, exist_ok=True)
HW = "/j-jepa-vol/phatjet-fpga/hw"

# ------------------------------------------------------------------ 1. frontier
rows = []
for f in glob.glob(f"{HW}/*/hw_results.json"):
    d = json.load(open(f))
    items = d if isinstance(d, list) else [d]
    for r in items:
        if not isinstance(r, dict) or r.get("lut_est") is None:
            continue
        ck = str(r.get("ckpt", ""))
        m = re.search(r"val_acc[=_]([0-9.]+)", ck)
        acc = r.get("ckpt_val_acc") or (float(m.group(1)) if m else None)
        n = re.search(r"/(n(\d+)|runs/[a-z-]*?)(?:-|/)", ck)
        N = 128 if "n128" in ck else 64 if ("n64" in ck or "fit-n64" in ck or "best-n64" in ck) else 32 if "n32" in ck else 16 if "n16" in ck else 150 if "150" in ck else None
        if "jedi" in ck:
            fam = "JEDI-Linear (Zihan re-trace)"
        else:
            fam = f"PHAT-JeT N={N}" if N else "PHAT-JeT"
        rows.append(dict(acc=acc, lut=r["lut_est"], stages=r.get("stages"), fam=fam, ck=os.path.basename(ck), dir=os.path.basename(os.path.dirname(f))))
# rebuttal / GitHub-repo rows (da4ml est., same convention)
rows += [dict(acc=0.6745, lut=87206, stages=29, fam="PHAT-JeT rebuttal (aaron)", ck="N=16"),
         dict(acc=0.7136, lut=98069, stages=29, fam="PHAT-JeT rebuttal (aaron)", ck="N=32"),
         dict(acc=0.7458, lut=167304, stages=29, fam="PHAT-JeT rebuttal (aaron)", ck="N=64"),
         dict(acc=0.7864, lut=1185319, stages=38, fam="PHAT-JeT rebuttal (aaron)", ck="N=128")]
jedi_pub = [(16, 0.719, 44000, 54), (32, 0.780, 45000, 63), (64, 0.809, 71000, 61), (128, 0.809, 98000, 82)]
jedi_pi = [(32, 0.790, 136000, 80), (64, 0.818, 164000, 78), (128, 0.816, 296000, 138)]

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
fams = sorted(set(r["fam"] for r in rows))
for i, fam in enumerate(fams):
    rr = [r for r in rows if r["fam"] == fam and r["acc"]]
    ax[0].scatter([r["lut"] for r in rr], [100 * r["acc"] for r in rr], s=22, label=fam, alpha=0.8, color=f"C{i}")
    ax[1].scatter([r["stages"] * 3.33 for r in rr if r["stages"]], [100 * r["acc"] for r in rr if r["stages"]], s=22, alpha=0.8, color=f"C{i}")
ax[0].scatter([j[2] for j in jedi_pub], [100 * j[1] for j in jedi_pub], marker="*", s=160, color="k", label="JEDI-Linear published (pT-sorted), post-route")
ax[0].scatter([j[2] for j in jedi_pi], [100 * j[1] for j in jedi_pi], marker="*", s=160, color="gray", label="JEDI-Linear published (perm-inv)")
ax[1].scatter([j[3] for j in jedi_pub], [100 * j[1] for j in jedi_pub], marker="*", s=160, color="k")
ax[1].scatter([j[3] for j in jedi_pi], [100 * j[1] for j in jedi_pi], marker="*", s=160, color="gray")
for j in jedi_pub:
    ax[0].annotate(f"N={j[0]}", (j[2], 100 * j[1]), fontsize=7, xytext=(4, -9), textcoords="offset points")
ax[0].axvspan(0, 172800, color="green", alpha=0.07); ax[0].axhline(80.9, ls="--", c="green", lw=1)
ax[0].text(20000, 81.1, "envelope: ≤10% VU13P (173k LUT) and ≥ JEDI N=64 (80.9%)", fontsize=8, color="green")
ax[0].set_xscale("log"); ax[0].set_xlabel("LUT estimate (da4ml, 300 MHz, hard_dc=2)"); ax[0].set_ylabel("val accuracy (%)")
ax[0].set_ylim(64, 83); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=7, loc="lower right")
ax[1].axvspan(0, 100, color="green", alpha=0.07); ax[1].axhline(80.9, ls="--", c="green", lw=1)
ax[1].set_xlabel("latency (ns) = stages × 3.33"); ax[1].set_ylabel("val accuracy (%)"); ax[1].set_ylim(64, 83); ax[1].grid(alpha=0.3)
fig.suptitle(f"Every traced design on the volume ({len(rows)} rows) vs the CTL2 envelope")
plt.tight_layout(); plt.savefig(f"{OUT}/frontier.png", dpi=130); plt.close()

inb = sorted([r for r in rows if r["lut"] <= 172800 and (r["stages"] or 99) < 30 and r["acc"]], key=lambda r: -r["acc"])
print("inside envelope (LUT<=173k, stages<30):", [(r["fam"], r["ck"][:40], r["acc"], int(r["lut"]), r["stages"]) for r in inb[:8]])
best = {}
for r in rows:
    if r["acc"]:
        best.setdefault(r["fam"], []).append(r)
for fam, rr in best.items():
    rr.sort(key=lambda r: r["lut"])
    print(fam, "cheapest:", rr[0]["ck"][:50], int(rr[0]["lut"]), rr[0]["acc"], "| best acc:", max(rr, key=lambda r: r["acc"])["acc"])

# ------------------------------------------------------------------ 2. what the leading-K particles carry
d = np.load("/j-jepa-vol/phatjet-fpga/jets_150x3_kt.npz")
x, y = d["x_val"][:40000], d["y_val"][:40000].argmax(1)
pt = x[:, :, 0]
real = pt > 0
CL = ["q", "g", "W", "Z", "t"]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
ranks = np.arange(1, 151)
for c in range(5):
    m = y == c
    cum = np.cumsum(pt[m], 1) / pt[m].sum(1, keepdims=True)
    ax[0].plot(ranks, cum.mean(0), label=CL[c])
    ax[1].plot(ranks, (real[m].sum(1)[:, None] > ranks[None]).mean(0), label=CL[c])
ax[0].set_xlabel("particle rank (kT-ordered)"); ax[0].set_ylabel("cumulative pT fraction (mean)"); ax[0].set_xscale("log"); ax[0].grid(alpha=0.3); ax[0].legend()
for K in (16, 32, 64):
    ax[0].axvline(K, c="gray", ls=":"); ax[1].axvline(K, c="gray", ls=":")
ax[1].set_xlabel("K"); ax[1].set_ylabel("fraction of jets with > K real particles"); ax[1].set_xscale("log"); ax[1].grid(alpha=0.3)
# kT-ordered: pT of the k-th particle relative to the leading one
rel = pt / np.maximum(pt[:, :1], 1e-6)
ax[2].semilogy(ranks, np.median(rel, 0), label="median pT_k / pT_1")
ax[2].semilogy(ranks, np.percentile(rel, 90, 0), label="90th pct")
ax[2].set_xlabel("particle rank (kT-ordered)"); ax[2].set_ylabel("relative pT"); ax[2].grid(alpha=0.3); ax[2].legend(); ax[2].set_xscale("log")
fig.suptitle("What the leading-K constituents carry (val set, 40k jets)")
plt.tight_layout(); plt.savefig(f"{OUT}/leading_k_physics.png", dpi=130); plt.close()
for K in (16, 32, 64, 128):
    print(f"K={K}: mean pT fraction in top-K = {np.mean(np.cumsum(pt,1)[:,K-1]/pt.sum(1)):.3f}; jets with >K particles: {(real.sum(1)>K).mean():.3f}")
