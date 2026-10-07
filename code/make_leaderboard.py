"""Collect every model run into docs/data.json for the GitHub Pages leaderboard.

  python make_leaderboard.py [--out /j-jepa-vol/phat-jet-aaron/leaderboard/data.json]

One leaderboard entry per (stage, config). Float entries: mean ± std over seeds of test accuracy / AUC /
background rejection. QAT entries: every traced checkpoint (accuracy on the 260k test set, EBOPs, LUT, FF,
DSP, BRAM, stages, latency @300 MHz, II) plus the best untraced Pareto point. Reference rows (paper, JEDI-Linear)
are added from literature values. Also embeds the builder source so the page can show the implementation.
"""
import argparse
import csv
import glob
import json
import os
import re
import shlex
import time

ROOT = "/j-jepa-vol/phat-jet-aaron"
VU13P = 1_728_000


def mean_std(v):
    v = [x for x in v if x is not None]
    if not v:
        return None, None
    m = sum(v) / len(v)
    s = (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5
    return m, s


def describe(c):
    """Human description of a config (what the model does)."""
    n, p, d, h = c["n"], c["patch"], c["d_model"], c["heads"]
    parts = [f"Input: the leading {n} constituents per jet ({c.get('sort', 'kt')}-sorted; pT, η, φ in the paper's robust units)."]
    parts.append(f"Per-particle embedding: Dense({d}) with {c.get('pre_norm') or 'no'} pre-activation.")
    g = c["gmp"]
    if g == "grid":
        G = len(c["gmp_edges"].split(",")) - 1 if c.get("gmp_edges") else c["gmp_bins"]
        ch = c.get("gmp_channels") or d
        parts.append(f"Geometric Message Passing on a static {G}×{G} (η,φ) grid (±{c['gmp_bounds']} scaled): one-hot scatter-add of {ch} channels, "
                     f"{c['gmp_kernel']}×{c['gmp_kernel']} depthwise conv, gather back, pointwise Dense, residual.")
    elif g == "sep":
        G = len(c["gmp_edges"].split(",")) - 1 if c.get("gmp_edges") else c["gmp_bins"]
        parts.append(f"Separable GMP: two 1-D passes (η grid and φ grid, {G} bins each, {c['gmp_kernel']}-tap depthwise conv) summed — ~{G}× cheaper than the 2-D grid.")
    else:
        parts.append("No geometric message passing.")
    K = c.get("attn_particles") or n
    loc = c["local"]
    if loc == "mha":
        parts.append(f"Local attention: exact multi-head self-attention ({h} head{'s' if h > 1 else ''}) inside patches of {p} consecutive particles"
                     + (f", applied only to the leading {K} particles (soft tail bypasses it)" if K < n else "") + (", shared Q/K projection" if c.get("share_qk") else "") + ".")
    elif loc == "linf":
        parts.append(f"Linformer-style attention over {'the leading ' + str(K) if K < n else 'all'} particles: keys/values projected along the particle axis to k={c.get('linf_k', 4)} rows "
                     f"with constant matrices (QEinsumDense), {h} head{'s' if h > 1 else ''}; activation×activation cost 2·N·k·d.")
    else:
        parts.append("No local attention.")
    gm = c["global_mode"]
    if gm == "mha":
        parts.append(f"Global stage: mean token per patch ({n // p} tokens) → multi-head attention over tokens → Dense → broadcast back to the particles (residual).")
    elif gm == "mix":
        parts.append(f"Global stage: mean token per patch ({n // p} tokens) → constant-weight token mixer (Dense over the flattened tokens, ReLU, Dense) → broadcast (residual). No activation×activation multiply.")
    elif gm == "mean":
        parts.append("Global stage: single mean token → Dense → broadcast to all particles (residual).")
    else:
        parts.append("No global stage.")
    parts.append(f"FFN (×{c['ffn_mult']}, ReLU, residual) → mean over particles → " + ("Dense(d, ReLU) → " if not c.get("no_head1") else "") + "Dense(5) logits.")
    if c["mode"] == "qat":
        parts.append(f"Quantization-aware training (HGQ2): per-element bit widths learned under an EBOPs penalty steered to {c['ebops_target']:.3g} EBOPs"
                     + (f"; attention lanes floored at {c['attn_floor']} fractional bits, attention β×{c['attn_beta_scale']}" if c.get("attn_floor") else "")
                     + (f"; distilled from a float teacher (α={c['distill_alpha']}, T={c['distill_T']})" if c.get("distill_alpha") else "") + ".")
    return " ".join(parts)


def diagram(c):
    """Block list for the SVG diagram: (label, kind) with kind in {io, linear, geo, attn, mix, pool}."""
    n, p = c["n"], c["patch"]
    K = c.get("attn_particles") or n
    b = [(f"{n} particles × (pT, η, φ)", "io"), (f"Dense {c['d_model']} + {c.get('pre_norm') or 'linear'}", "linear")]
    if c["gmp"] == "grid":
        G = len(c["gmp_edges"].split(",")) - 1 if c.get("gmp_edges") else c["gmp_bins"]
        b.append((f"GMP {G}×{G} grid, {c['gmp_kernel']}×{c['gmp_kernel']} dw-conv", "geo"))
    elif c["gmp"] == "sep":
        b.append((f"GMP separable (η | φ), {c['gmp_bins']} bins", "geo"))
    if c["local"] == "mha":
        b.append((f"patch MHA P={p}, {c['heads']}h" + (f" on top {K}" if K < n else ""), "attn"))
    elif c["local"] == "linf":
        b.append((f"Linformer k={c.get('linf_k', 4)}, {c['heads']}h" + (f" on top {K}" if K < n else ""), "attn"))
    gm = c["global_mode"]
    if gm == "mha":
        b += [(f"mean tokens ({n // p})", "pool"), ("token MHA", "attn"), ("broadcast +", "mix")]
    elif gm == "mix":
        b += [(f"mean tokens ({n // p})", "pool"), ("token mixer (const)", "mix"), ("broadcast +", "mix")]
    elif gm == "mean":
        b += [("mean token", "pool"), ("Dense → broadcast +", "mix")]
    if c["ffn_mult"]:
        b.append((f"FFN ×{c['ffn_mult']}", "linear"))
    b.append(("mean over particles", "pool"))
    if not c.get("no_head1"):
        b.append((f"Dense {c['d_model']} ReLU", "linear"))
    b.append(("Dense 5 → logits", "io"))
    return b


def run_command(c):
    skip = {"out", "max_jets", "resume"}
    args = []
    for k, v in c.items():
        if k in skip or v in (None, False, 0, 0.0, "") and k not in ("i0_w", "i0_a"):
            continue
        if v is True:
            args.append(f"--{k}")
        else:
            args.append(f"--{k} {shlex.quote(str(v))}")
    return "python scripts/train_variant.py --out runs/<stage>/<name>/seed<s> " + " ".join(args)


def last_csv(p):
    try:
        rows = list(csv.DictReader(open(p)))
        return rows[-1] if rows else None, len(rows)
    except Exception:
        return None, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/leaderboard/data.json")
    a = ap.parse_args()
    entries = []
    for cfgdir in sorted(glob.glob(f"{ROOT}/runs/[fq]*/*/")):
        stage, name = cfgdir.rstrip("/").split("/")[-2:]
        if name.startswith("_"):
            continue
        seeds = sorted(glob.glob(cfgdir + "seed*/"))
        cfg = None
        for sd in seeds:
            if os.path.exists(sd + "config.json"):
                cfg = json.load(open(sd + "config.json")); break
        if cfg is None:
            continue
        e = dict(id=f"{stage}/{name}", stage=stage, name=name, mode=cfg["mode"], N=cfg["n"], sort=cfg.get("sort", "kt"),
                 config=cfg, description=describe(cfg), diagram=diagram(cfg), command=run_command(cfg), seeds=[])
        traced = []
        for sd in seeds:
            s = dict(seed=os.path.basename(sd.rstrip("/")))
            r = os.path.join(sd, "result.json")
            if os.path.exists(r):
                j = json.load(open(r)); s.update(done=True, **{k: j.get(k) for k in ("best_val", "final_val", "test_acc", "test_auc", "bg_rej", "avg_bg_rej", "params", "minutes")})
            else:
                row, n = last_csv(os.path.join(sd, "log.csv"))
                s.update(done=False, epochs=n)
                if row:
                    k = next((c for c in ("val_accuracy", "val_acc_metric") if c in row), None)
                    s["running_val"] = float(row[k]) if k else None
                    if row.get("ebops"):
                        s["running_ebops"] = float(row["ebops"])
            e["seeds"].append(s)
            tj = os.path.join(sd, "trace.json")
            if os.path.exists(tj):
                for t in json.load(open(tj)):
                    if t.get("lut_est"):
                        traced.append(dict(ckpt=os.path.basename(t["ckpt"]), seed=s["seed"], test_acc=t.get("acc"), test_auc=t.get("test_auc"),
                                           avg_bg_rej=t.get("avg_bg_rej"), bg_rej=t.get("bg_rej"), ebops=t.get("ebops"), lut=t["lut_est"],
                                           lut_pct=100 * t["lut_est"] / VU13P, ff=t.get("ff_est"), dsp=0, bram=0, stages=t["stages"],
                                           latency_ns=t["latency_ns"], ii=1, n_eval=t.get("n_eval"),
                                           in_envelope=bool(t["lut_est"] <= 172800 and t["stages"] < 30)))
            # pareto summary for QAT
            if cfg["mode"] == "qat":
                front = []
                for f in glob.glob(os.path.join(sd, "pareto", "*.keras")):
                    m = re.search(r"epoch=(\d+)-val_acc=([0-9.]+)-ebops=([0-9]+(?:\.[0-9]+)?)", os.path.basename(f))
                    if m:
                        front.append(dict(epoch=int(m.group(1)), val_acc=float(m.group(2)), ebops=float(m.group(3))))
                s["pareto"] = sorted(front, key=lambda t: t["ebops"])
        done = [s for s in e["seeds"] if s.get("done")]
        for k in ("test_acc", "test_auc", "avg_bg_rej", "best_val"):
            m, sdv = mean_std([s.get(k) for s in done])
            e[k], e[k + "_std"] = m, sdv
        e["params"] = next((s.get("params") for s in done), None)
        e["n_done"], e["n_seeds"] = len(done), len(e["seeds"])
        e["traced"] = sorted(traced, key=lambda t: t["lut"])
        if traced:
            best_in = [t for t in traced if t["in_envelope"]]
            e["best_in_envelope"] = max(best_in, key=lambda t: t["test_acc"]) if best_in else None
            e["cheapest"] = traced[0]
        entries.append(e)
    refs = [
        dict(id="ref/paper-phatjet", stage="reference", name="PHAT-JeT (paper, float, N=150)", mode="float", N=150, test_acc=0.8180, avg_bg_rej=71.6, params=6405, reference=True,
             description="Paper model (arXiv:2605.21789): kT-sorted 150 constituents, dynamic per-jet GMP grid δ=0.2, patch size 10, d=16, 4 heads, LayerNorm, FFN ×4."),
        dict(id="ref/jedi-n32", stage="reference", name="JEDI-Linear N=32 (pT-sorted, post-route)", mode="qat", N=32, test_acc=0.780, reference=True,
             traced=[dict(ckpt="published", lut=45000, lut_pct=2.6, ff=None, dsp=0, bram=0, stages=19, latency_ns=63, ii=1, test_acc=0.780, in_envelope=True)],
             description="Published JEDI-Linear (arXiv:2508.15468) FPGA result, pT-sorted inputs, post place-and-route on VU13P."),
        dict(id="ref/jedi-n64", stage="reference", name="JEDI-Linear N=64 (pT-sorted, post-route)", mode="qat", N=64, test_acc=0.809, reference=True,
             traced=[dict(ckpt="published", lut=71000, lut_pct=4.1, ff=None, dsp=0, bram=0, stages=18, latency_ns=61, ii=1, test_acc=0.809, in_envelope=True)],
             description="Published JEDI-Linear FPGA result (the envelope target for N=64)."),
        dict(id="ref/rebuttal-n64", stage="reference", name="PHAT-JeT rebuttal N=64 (da4ml est.)", mode="qat", N=64, test_acc=0.7458, reference=True,
             traced=[dict(ckpt="rebuttal", lut=167304, lut_pct=9.7, ff=None, dsp=0, bram=0, stages=29, latency_ns=96.6, ii=1, test_acc=0.7458, in_envelope=True)],
             description="The number quoted in the NeurIPS rebuttal: static core7 grid, 161–300 QAT epochs on CPU."),
    ]
    code = {}
    for f in ("phat_variants.py", "train_variant.py", "eval_ckpt.py"):
        p = os.path.join(ROOT, "scripts", f)
        if os.path.exists(p):
            code[f] = open(p).read()
    out = dict(generated=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), envelope=dict(lut=172800, stages=30, latency_ns=100, ii=1, acc_n64=0.809, acc_n32=0.780),
               entries=entries + refs, code=code)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"))
    print(f"{len(entries)} model entries + {len(refs)} references -> {a.out}")


if __name__ == "__main__":
    main()
