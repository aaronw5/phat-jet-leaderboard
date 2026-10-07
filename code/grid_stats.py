"""Grid statistics for PHAT-JeT GMP: dynamic (paper, per-jet min-shifted) vs static grids.

All coordinates are in the paper's robust-scaled units (eta/IQR, IQR=0.1211 for eta,
0.1211 for phi, median 0), i.e. the units the paper's `grid_size` (delta) is quoted in.

Outputs plots (PNG) + a JSON of the summary numbers into OUT_DIR.
"""
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "/j-jepa-vol/l1-jet-id/data/jetid/processed"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/j-jepa-vol/phat-jet-aaron/plots/grid_stats"
NJ = int(os.environ.get("NJ", 50000))
NPAIR = int(os.environ.get("NPAIR", 5000))
IQR = 0.1211  # eta/phi IQR of the paper's robust scaling (raw = scaled * IQR)
CLASSES = ["q", "g", "W", "Z", "t"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
os.makedirs(OUT, exist_ok=True)

x = np.load(f"{DATA}/x_train_robust_150const_ptetaphi.npy", mmap_mode="r")
y = np.load(f"{DATA}/y_train_robust_150const_ptetaphi.npy", mmap_mode="r")
rng = np.random.default_rng(0)
idx = np.sort(rng.choice(x.shape[0], NJ, replace=False))
X = np.asarray(x[idx], dtype=np.float32)
lab = np.argmax(np.asarray(y[idx]), 1)
real = np.abs(X[:, :, 0]) > 0
pt, eta, phi = X[:, :, 0], X[:, :, 1], X[:, :, 2]
nreal = real.sum(1)
summary = {"n_jets": NJ, "iqr": IQR}


def wq(v, w, q):
    o = np.argsort(v)
    c = np.cumsum(w[o]) / w.sum()
    return v[o][np.searchsorted(c, q)]


# ---------------------------------------------------------------- 1. coordinate extent
r = np.sqrt(eta**2 + phi**2)
ext = np.maximum(np.abs(eta), np.abs(phi))
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
bins = np.linspace(-4, 4, 161)
for c in range(5):
    m = real & (lab[:, None] == c)
    ax[0].hist(eta[m], bins, weights=pt[m], histtype="step", density=True, color=COLORS[c], label=CLASSES[c])
ax[0].set_yscale("log"); ax[0].set_xlabel(r"$\eta_{rel}$ (scaled units)"); ax[0].set_title(r"$p_T$-weighted $\eta$ profile")
ax[0].legend()
Ls = np.linspace(0.25, 4.0, 60)
fpart = [(ext[real] > L).mean() for L in Ls]
fpt = [pt[real][ext[real] > L].sum() / pt[real].sum() for L in Ls]
ax[1].semilogy(Ls, fpart, label="particles outside")
ax[1].semilogy(Ls, fpt, label=r"$p_T$ outside")
ax[1].set_xlabel(r"static half-extent $L$ (scaled); max(|$\eta$|,|$\phi$|) > L clamped")
ax[1].set_title("Clipping of a static grid of half-extent L")
ax[1].axvline(0.4 / IQR, ls="--", c="k", lw=0.8); ax[1].text(0.4 / IQR, 0.3, " HW ±0.4 raw", fontsize=8)
ax[1].grid(alpha=0.3); ax[1].legend()
for c in range(5):
    m = real & (lab[:, None] == c)
    ax[2].hist(r[m], np.linspace(0, 4, 81), weights=pt[m], histtype="step", density=True, cumulative=True, color=COLORS[c], label=CLASSES[c])
ax[2].set_xlabel(r"$\Delta R$ from jet axis (scaled)"); ax[2].set_title(r"cumulative $p_T$ containment"); ax[2].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/01_extent.png", dpi=130); plt.close()
summary["extent"] = {f"L={L:.2f}": {"part_out": float(a), "pt_out": float(b)} for L, a, b in zip(Ls[::6], fpart[::6], fpt[::6])}
summary["pt_containment_r"] = {str(q): float(wq(r[real], pt[real], q)) for q in [0.5, 0.9, 0.95, 0.99, 0.999]}
summary["abs_coord_quantiles"] = {str(q): float(np.quantile(ext[real], q)) for q in [0.9, 0.99, 0.999, 0.9999]}

# ---------------------------------------------------------------- 2. 2D density per class
fig, ax = plt.subplots(1, 5, figsize=(22, 4.4))
b2 = np.linspace(-3, 3, 121)
for c in range(5):
    m = real & (lab[:, None] == c)
    H, _, _ = np.histogram2d(eta[m], phi[m], [b2, b2], weights=pt[m])
    ax[c].imshow(np.log10(H.T + 1e-3), origin="lower", extent=[-3, 3, -3, 3], cmap="viridis")
    ax[c].set_title(f"{CLASSES[c]}: log10 Σ$p_T$"); ax[c].set_xlabel(r"$\eta$"); ax[c].set_ylabel(r"$\phi$")
    for e in np.array([-0.40, -0.20, -0.12, -0.04, 0.04, 0.12, 0.20, 0.40]) / IQR:
        ax[c].axvline(e, c="w", lw=0.5, alpha=0.6); ax[c].axhline(e, c="w", lw=0.5, alpha=0.6)
fig.suptitle("pT density per class (white lines: HW 'core7' static grid, converted to scaled units)")
plt.tight_layout(); plt.savefig(f"{OUT}/02_density_core7_overlay.png", dpi=110); plt.close()

# ---------------------------------------------------------------- grid assignment helpers
def assign(e, p, delta, scheme, L=None):
    """Return integer cell coords (ie, ip) per particle. Padding is ignored downstream."""
    if scheme == "dynamic":  # paper: per-jet min over REAL particles, floor((x-min)/delta)
        emin = np.where(real, e, np.inf).min(1, keepdims=True)
        pmin = np.where(real, p, np.inf).min(1, keepdims=True)
        return np.floor((e - emin) / delta).astype(np.int64), np.floor((p - pmin) / delta).astype(np.int64)
    off = 0.0 if scheme == "static_even" else 0.5  # odd: a cell is centred on the jet axis
    ie, ip = np.floor(e / delta + off).astype(np.int64), np.floor(p / delta + off).astype(np.int64)
    if L is not None:
        k = int(np.floor(L / delta + off))
        lo = -k if off else -k
        ie, ip = np.clip(ie, lo, k - (0 if off else 1)), np.clip(ip, lo, k - (0 if off else 1))
    return ie, ip


def cell_stats(ie, ip):
    """per jet: hottest-cell pT share, collision fraction, occupied cells."""
    key = (ie - ie.min()) * 100003 + (ip - ip.min())
    key = np.where(real, key, -1)
    hot, coll, occ = np.zeros(NJ), np.zeros(NJ), np.zeros(NJ)
    for j in range(NJ):
        m = real[j]
        k = key[j, m]
        u, inv, cnt = np.unique(k, return_inverse=True, return_counts=True)
        s = np.bincount(inv, weights=pt[j, m])
        hot[j] = s.max() / s.sum()
        coll[j] = (cnt[inv] > 1).mean()
        occ[j] = len(u)
    return hot, coll, occ


deltas = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.3, 1.65]
schemes = ["dynamic", "static_even", "static_odd"]
res = {s: {"hot": [], "coll": [], "occ": [], "ncell": []} for s in schemes}
for d in deltas:
    for s in schemes:
        ie, ip = assign(eta, phi, d, s)
        h, c, o = cell_stats(ie, ip)
        res[s]["hot"].append(np.median(h)); res[s]["coll"].append(c.mean()); res[s]["occ"].append(o.mean())
        if s == "dynamic":
            w = np.where(real, ie, -1).max(1) + 1
            hh = np.where(real, ip, -1).max(1) + 1
            res[s]["ncell"].append(float(np.mean(w * hh)))
        else:
            res[s]["ncell"].append(None)
    print("delta", d, {s: (round(res[s]["hot"][-1], 3), round(res[s]["coll"][-1], 3)) for s in schemes}, flush=True)
summary["cell_stats"] = {"deltas": deltas, **res}

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for s, ls in zip(schemes, ["-", "--", ":"]):
    ax[0].plot(deltas, res[s]["hot"], ls, marker="o", label=s)
    ax[1].plot(deltas, res[s]["coll"], ls, marker="o", label=s)
    ax[2].plot(deltas, np.array(res[s]["occ"]) / nreal.mean(), ls, marker="o", label=s)
ax[0].set_ylabel("median hottest-cell pT share"); ax[1].set_ylabel("fraction of particles sharing a cell")
ax[2].set_ylabel("occupied cells / real particles")
for a in ax:
    a.set_xscale("log"); a.set_xlabel(r"cell size $\delta$ (scaled; raw = ×0.121)"); a.grid(alpha=0.3); a.legend()
    for d0, t in [(0.2, "paper δ"), (0.08 / IQR, "core7 core"), (0.2 / IQR, "core7 outer")]:
        a.axvline(d0, c="gray", lw=0.7, ls="-."); a.text(d0, a.get_ylim()[0], t, rotation=90, fontsize=7, va="bottom")
fig.suptitle("Cell statistics: dynamic (per-jet min-shift) vs static grids (no extent clamp)")
plt.tight_layout(); plt.savefig(f"{OUT}/03_cell_stats_vs_delta.png", dpi=130); plt.close()

# ---------------------------------------------------------------- 4. dynamic grid size / phase
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for d in [0.1, 0.2, 0.3, 0.8]:
    ie, ip = assign(eta, phi, d, "dynamic")
    w = np.where(real, ie, -1).max(1) + 1
    ax[0].hist(w, np.arange(0, 80), histtype="step", density=True, label=f"δ={d}")
ax[0].set_xlabel("dynamic grid width (cells, η)"); ax[0].set_title("per-jet dynamic grid size"); ax[0].legend()
emin = np.where(real, eta, np.inf).min(1)
for d in [0.2, 0.8]:
    phase = np.mod(-emin, d) / d  # where the jet axis (0) sits inside its cell
    ax[1].hist(phase, 50, histtype="step", density=True, label=f"δ={d}")
ax[1].set_xlabel("position of jet axis inside its cell (0=edge, 0.5=centre)")
ax[1].set_title("dynamic grid: axis-to-cell phase (≈ random)"); ax[1].legend()
ax[2].hist(emin, 100, histtype="step", density=True, label=r"$\eta_{min}$")
ax[2].set_xlabel(r"per-jet $\eta_{min}$ (scaled)"); ax[2].set_title("dynamic grid origin per jet")
plt.tight_layout(); plt.savefig(f"{OUT}/04_dynamic_grid_size_phase.png", dpi=130); plt.close()

# ---------------------------------------------------------------- 5. message-passing graph agreement
# A depthwise conv with kernel k, padding='same' (TF: pad_before=(k-1)//2) lets particle j
# reach particle i iff cell_j - cell_i in [-(k-1)//2, k//2] on both axes.
sub = np.arange(min(NPAIR, NJ))


def neigh(ie, ip, k, j):
    m = real[j]
    de = ie[j, m][None, :] - ie[j, m][:, None]
    dp = ip[j, m][None, :] - ip[j, m][:, None]
    lo, hi = -((k - 1) // 2), k // 2
    return (de >= lo) & (de <= hi) & (dp >= lo) & (dp <= hi)


def pair_dist(j):
    m = real[j]
    return np.maximum(np.abs(eta[j, m][None] - eta[j, m][:, None]), np.abs(phi[j, m][None] - phi[j, m][:, None]))


configs = [  # (label, delta, k, scheme, L) -- HW config approximated as uniform core bins
    ("paper dyn δ=0.2 k=8", 0.2, 8, "dynamic", None),
    ("static-odd δ=0.2 k=8", 0.2, 8, "static_odd", None),
    ("static-even δ=0.2 k=8", 0.2, 8, "static_even", None),
    ("paper dyn δ=0.8 k=8", 0.8, 8, "dynamic", None),
    ("static-odd δ=0.8 k=8", 0.8, 8, "static_odd", None),
    ("static-odd δ=0.2 k=7", 0.2, 7, "static_odd", None),
    ("static-odd δ=0.4 k=3", 0.4, 3, "static_odd", None),
    ("static-odd δ=0.66 k=3 (≈core7)", 0.08 / IQR, 3, "static_odd", 0.4 / IQR),
]
assigned = {c[0]: assign(eta, phi, c[1], c[3], c[4]) for c in configs}
dbins = np.linspace(0, 6, 61)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
jac = {}
ref = configs[0][0]
for lab_, d, k, s, L in configs:
    num, den = np.zeros(len(dbins) - 1), np.zeros(len(dbins) - 1)
    jj = []
    for j in sub:
        A = neigh(*assigned[lab_], k, j)
        D = pair_dist(j)
        h1, _ = np.histogram(D[A], dbins); h2, _ = np.histogram(D, dbins)
        num += h1; den += h2
        B = neigh(*assigned[ref], 8, j)
        jj.append((A & B).sum() / max((A | B).sum(), 1))
    jac[lab_] = float(np.mean(jj))
    ax[0].plot(0.5 * (dbins[1:] + dbins[:-1]), num / np.maximum(den, 1), label=lab_)
ax[0].set_xlabel(r"pair separation max(|Δη|,|Δφ|) (scaled; raw ×0.121)")
ax[0].set_ylabel("P(pair exchanges a GMP message)")
ax[0].set_title("Effective GMP interaction kernel"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=7)
for a_, sep, t in [(ax[0], 0.16 / IQR, "W prongs"), (ax[0], 0.28 / IQR, "t prongs")]:
    a_.axvline(sep, c="gray", ls=":"); a_.text(sep, 0.5, t, rotation=90, fontsize=7)
names = list(jac)
ax[1].barh(names, [jac[n] for n in names]); ax[1].set_xlabel("Jaccard overlap of interacting pairs vs paper dyn δ=0.2 k=8")
ax[1].set_xlim(0, 1)
plt.tight_layout(); plt.savefig(f"{OUT}/05_interaction_kernel.png", dpi=130); plt.close()
summary["jaccard_vs_paper"] = jac

# ---------------------------------------------------------------- 6. per-class pair separations (what scale must be resolved)
fig, ax = plt.subplots(figsize=(7, 4.5))
for c in range(5):
    ds, ws = [], []
    for j in np.where(lab[:NPAIR] == c)[0][:800]:
        m = real[j]
        D = np.sqrt((eta[j, m][None] - eta[j, m][:, None]) ** 2 + (phi[j, m][None] - phi[j, m][:, None]) ** 2)
        W = pt[j, m][None] * pt[j, m][:, None]
        iu = np.triu_indices(m.sum(), 1)
        ds.append(D[iu]); ws.append(W[iu])
    ax.hist(np.concatenate(ds), np.linspace(0, 6, 121), weights=np.concatenate(ws), histtype="step", density=True, color=COLORS[c], label=CLASSES[c])
ax.set_xlabel(r"pair $\Delta R$ (scaled)"); ax.set_ylabel(r"$p_{T,i}p_{T,j}$-weighted density"); ax.legend()
ax.set_title("Energy-energy correlator: where class information lives")
for d0, t in [(0.2, "δ=0.2"), (0.8, "δ=0.8"), (1.6, "8×0.2")]:
    ax.axvline(d0, c="gray", ls=":"); ax.text(d0, ax.get_ylim()[1] * 0.8, t, rotation=90, fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/06_eec_per_class.png", dpi=130); plt.close()

json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
print(json.dumps(summary, indent=1, default=float)[:4000])
