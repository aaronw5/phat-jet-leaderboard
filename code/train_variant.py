#!/usr/bin/env python
"""Float pretraining or HGQ2 QAT of a PHAT-JeT variant (see phat_variants.py), Keras 3 / JAX.

  --mode float : train the float model (cosine LR), save float.keras + float_hist.json
  --mode qat   : build the Q model, init from --float_ckpt (weights by layer name) or from
                 scratch, then train with the EBOPs penalty. beta is driven by a controller
                 that steers log(ebops) to log(--ebops_target) (as in the July campaign's PID runs);
                 with --ebops_target 0 the jsc150-style fixed ramp is used instead.
                 Pareto checkpoints (val_accuracy vs ebops) land in <out>/pareto/.

Distillation: --teacher_logits <npy> with pre-computed teacher logits for the TRAIN set
(same jet order); loss = (1-a)*CE(y) + a*T^2*KL(soft teacher || soft student).

Data: /j-jepa-vol/phatjet-fpga/jets_150x3_kt.npz (paper units, kt-sorted, pad at end),
sliced to --n particles. Read-only; outputs only under --out.
"""
import argparse
import json
import os
import sys
import time
from math import cos, pi

os.environ.setdefault("KERAS_BACKEND", "jax")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import keras
from keras import ops

from phat_variants import build_variant, evaluate, kw_from_args, load_split, transfer_weights

DATA = "/j-jepa-vol/phatjet-fpga/jets_150x3_kt.npz"
EXP_I_MAX = 5.0


# ----------------------------------------------------------------------------- callbacks
class ClampSoftmaxExpBits(keras.callbacks.Callback):
    """hgq 0.1.9 ignores the exp-table `ic` constraint in the WRAP training path; re-clamp per epoch."""

    def __init__(self, i_max=EXP_I_MAX):
        super().__init__(); self.i_max = float(i_max)

    def _qs(self):
        for l in self.model.layers:
            q = getattr(getattr(getattr(l, "exp_table", None), "iq", None), "quantizer", None)
            if q is not None and hasattr(q, "_i"):
                yield q

    def on_epoch_end(self, epoch, logs=None):
        for q in self._qs():
            iv = np.array(q._i)
            if iv.max() > self.i_max:
                q._i.assign(np.minimum(iv, self.i_max))


class AttnBetaPID:
    """hgq.utils.sugar.BetaPID (log-space PID on EBOPs, warm-up, damping) + a separate beta scale for *attn* layers."""

    def __new__(cls, target, init_beta, p, i, warmup, max_beta, damp, attn_scale=1.0):
        from hgq.utils.sugar.beta_pid import BetaPID

        class _PID(BetaPID):
            def set_beta(self, beta):
                for l in self.model._flatten_layers():
                    if getattr(l, "_beta", None) is not None:
                        v = beta * (attn_scale if "attn" in l.name else 1.0)
                        l._beta.assign(ops.convert_to_tensor(v, dtype=l._beta.dtype))

        return _PID(target, init_beta=init_beta, p=p, i=i, warmup=warmup, max_beta=max_beta, damp_beta_on_target=damp)


class StopFile(keras.callbacks.Callback):
    def __init__(self, out):
        super().__init__(); self.p = os.path.join(out, "STOP")

    def on_epoch_end(self, epoch, logs=None):
        if os.path.exists(self.p):
            print(f"[stop] STOP file found at epoch {epoch}", flush=True); self.model.stop_training = True


def cosine(lr0, epochs, alpha=1e-6, restarts=1):
    cyc = max(epochs // restarts, 1)
    return lambda e: alpha + 0.5 * (lr0 - alpha) * (1 + cos(pi * min((e % cyc) / cyc, 1.0)))


# ----------------------------------------------------------------------------- losses
def make_loss(alpha, T, label_smooth):
    ce = keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=label_smooth)
    if alpha <= 0:
        return ce

    def loss(y_true, y_pred):  # y_true = [onehot(5) | teacher logits(5)]
        y, t = y_true[:, :5], y_true[:, 5:]
        pt = ops.softmax(t / T, axis=-1)
        ls = ops.log_softmax(y_pred / T, axis=-1)
        kd = -ops.sum(pt * ls, axis=-1) + ops.sum(pt * ops.log(pt + 1e-9), axis=-1)
        return (1 - alpha) * ce(y, y_pred) + alpha * T * T * ops.mean(kd)

    return loss


def acc_metric(y_true, y_pred):
    return ops.mean(ops.cast(ops.argmax(y_true[:, :5], -1) == ops.argmax(y_pred, -1), "float32"))


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["float", "qat"], required=True)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--patch", type=int, default=8)
    p.add_argument("--d_model", type=int, default=16)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--gmp", default="grid", choices=["grid", "sep", "none"])
    p.add_argument("--gmp_bins", type=int, default=8)
    p.add_argument("--gmp_bounds", type=float, default=1.6)
    p.add_argument("--gmp_edges", default=None)
    p.add_argument("--gmp_kernel", type=int, default=3)
    p.add_argument("--gmp_channels", type=int, default=0)
    p.add_argument("--local", default="mha", choices=["mha", "linf", "none"])
    p.add_argument("--linf_k", type=int, default=4, help="Linformer projected length k")
    p.add_argument("--global_mode", default="mha", choices=["mha", "mix", "mean", "jedi", "none"])
    p.add_argument("--ffn_mult", type=int, default=1)
    p.add_argument("--ffn_hidden", type=int, default=0, help="FFN hidden width (overrides ffn_mult*d); 0 = use ffn_mult")
    p.add_argument("--mix_hidden", type=int, default=0, help="token-mixer hidden width (low-rank mixer); 0 = NP*d")
    p.add_argument("--n_blocks", type=int, default=1, help="repeat GMP/attention/global/FFN block (latency headroom)")
    p.add_argument("--lut_layers", default="", help="comma list of roles using HGQ learned-LUT dense (QDenseT): embed,ffn,mix,proj,head")
    p.add_argument("--lut_dhl", type=int, default=8, help="hidden width of each QDenseT edge function")
    p.add_argument("--pre_norm", default="tanh")
    p.add_argument("--attn_bits", type=int, default=None)
    p.add_argument("--parallel_attn", action="store_true")
    p.add_argument("--attn_particles", type=int, default=0, help="local attention over the leading K particles only")
    p.add_argument("--share_qk", action="store_true")
    p.add_argument("--no_head1", action="store_true")
    p.add_argument("--unshared", action="store_true", help="position-specific weights in embed/GMP-pointwise/FFN (JEDI-style)")
    p.add_argument("--data", default="ours", choices=["ours", "hls4ml"], help="input pipeline (hls4ml = JEDI-linear's exact inputs)")
    p.add_argument("--datalane_overflow", default="wrap", choices=["wrap", "SAT"], help="activation overflow: wrap (default) or saturate")
    p.add_argument("--shared_bits", action="store_true", help="data-lane bits shared across particles (JEDI perm-inv quantization)")
    p.add_argument("--arch", default="phat", choices=["phat", "jedi"], help="jedi: JEDI-linear gnn backbone (+ GMP if --gmp grid)")
    p.add_argument("--jedi_width", type=int, default=64)
    p.add_argument("--jedi_head", default="64,32,16")
    p.add_argument("--jedi_per_slot_bits", action="store_true", help="per-particle-slot data-lane bits (JEDI pT-sorted) instead of shared (perminv)")
    p.add_argument("--w_l1", type=float, default=0.0, help="L1 on kernels (sparsity; zero weights are free in RTL)")
    # training
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=2790)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--restarts", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label_smooth", type=float, default=0.0)
    p.add_argument("--teacher_logits", default=None)
    p.add_argument("--distill_alpha", type=float, default=0.0)
    p.add_argument("--distill_T", type=float, default=2.0)
    p.add_argument("--max_jets", type=int, default=0)
    p.add_argument("--sort", default="kt", choices=["kt", "pt"], help="constituent order: kt (stored) or pt-descending")
    # qat
    p.add_argument("--float_ckpt", default=None)
    # bit init: from scratch (jsc150) 7/7/0/0; from a float donor the repo-validated 10/10/3/8 (else wrap-collapse)
    p.add_argument("--bw_k", type=int, default=None)
    p.add_argument("--bw_a", type=int, default=None)
    p.add_argument("--i0_w", type=int, default=None)
    p.add_argument("--i0_a", type=int, default=None)
    p.add_argument("--floor_bits", type=float, default=0.0)
    p.add_argument("--l1_reg", type=float, default=1e-8)
    p.add_argument("--ebops_target", type=float, default=0.0)
    p.add_argument("--beta0", type=float, default=2e-8)
    p.add_argument("--beta_kp", type=float, default=0.1, help="BetaPID p (log-space); July campaign used 0.1")
    p.add_argument("--beta_ki", type=float, default=0.002, help="BetaPID i; campaign used 0.001")
    p.add_argument("--beta_warmup", type=int, default=10)
    p.add_argument("--beta_max", type=float, default=3e-5)
    p.add_argument("--beta_damp", type=float, default=0.02, help="beta *= 1-damp while below target")
    p.add_argument("--resume", default=None, help="resume from a Q checkpoint (.keras)")
    p.add_argument("--attn_floor", type=float, default=0.0, help="min fractional bits on attention Q/K/V/O lanes")
    p.add_argument("--attn_in_floor", type=float, default=None, help="min fractional bits on the INPUT lanes of every attention sub-layer (QK, softmax, AV, projections, Linformer E/F)")
    p.add_argument("--attn_ifloor", type=float, default=None, help="min integer bits on those attention input lanes (no clipping of scores)")
    p.add_argument("--attn_wbits", type=float, default=None, help="min bit width of attention projection weights")
    p.add_argument("--attn_beta_scale", type=float, default=1.0, help="EBOPs pressure multiplier on *attn* layers")
    a = p.parse_args()

    xfer = bool(a.float_ckpt)
    a.bw_k = a.bw_k if a.bw_k is not None else (10 if xfer else 7)
    a.bw_a = a.bw_a if a.bw_a is not None else (10 if xfer else 7)
    a.i0_w = a.i0_w if a.i0_w is not None else (3 if xfer else 0)
    a.i0_a = a.i0_a if a.i0_a is not None else (8 if xfer else 0)
    os.makedirs(a.out, exist_ok=True)
    keras.utils.set_random_seed(a.seed)
    json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"), indent=1)

    # Protocol: validation = fixed 10% split of the 620k training jets (used for monitoring and Pareto
    # checkpoint selection); the 260k "x_val" file is the paper's held-out TEST set, touched only at the end.
    from phat_variants import use_data
    use_data(a.data)
    if a.teacher_logits and (a.data == "hls4ml") != ("_hl" in os.path.basename(a.teacher_logits)):
        raise SystemExit("teacher logits must come from the same data file / jet order ('_hl' in the name <=> --data hls4ml)")
    xall, yall = load_split("train", a.n, a.sort)
    xte, yte = load_split("val", a.n, a.sort)
    perm = np.random.default_rng(1234).permutation(len(xall))
    nva = len(xall) // 10
    iva, itr = perm[:nva], perm[nva:]
    if a.teacher_logits and a.distill_alpha > 0:
        tl = np.load(a.teacher_logits).astype(np.float32)
        assert len(tl) == len(xall), (tl.shape, xall.shape)
        yall_fit = np.concatenate([yall, tl], 1)
    else:
        yall_fit = yall
    xtr, ytr_fit = xall[itr], yall_fit[itr]
    xva, yva_fit, yva = xall[iva], yall_fit[iva], yall[iva]
    del xall
    if a.max_jets:
        xtr, ytr_fit = xtr[: a.max_jets], ytr_fit[: a.max_jets]
        xva, yva_fit, yva = xva[: a.max_jets], yva_fit[: a.max_jets], yva[: a.max_jets]
        xte, yte = xte[: a.max_jets], yte[: a.max_jets]

    edges = [float(e) for e in a.gmp_edges.split(",")] if a.gmp_edges else None
    kw = kw_from_args(a)

    cbs = [keras.callbacks.LearningRateScheduler(cosine(a.lr, a.epochs, restarts=a.restarts)),
           keras.callbacks.CSVLogger(os.path.join(a.out, "log.csv"), append=bool(a.resume)), StopFile(a.out)]
    metrics = [acc_metric] if a.distill_alpha > 0 else ["accuracy"]
    mname = "acc_metric" if a.distill_alpha > 0 else "accuracy"

    if a.mode == "float":
        model = build_variant(quantized=False, **kw)
        cbs.append(keras.callbacks.ModelCheckpoint(os.path.join(a.out, "float.weights.h5"), monitor=f"val_{mname}",
                                                   save_best_only=True, save_weights_only=True))
    else:
        from hgq.utils.sugar import FreeEBOPs, ParetoFront
        from phat_variants import build_q_from_args
        model = build_q_from_args(a)
        if a.resume:
            model.load_weights(a.resume)
            print(f"resumed Q weights from {a.resume}", flush=True)
        elif a.float_ckpt:
            fm = build_variant(quantized=False, **kw)
            fm.load_weights(a.float_ckpt)
            n = transfer_weights(model, fm)
            lg = model.predict(xva[:50000], batch_size=4096, verbose=0)
            acc0 = float((lg.argmax(1) == yva[:50000].argmax(1)).mean())
            lf = fm.predict(xva[:50000], batch_size=4096, verbose=0)
            accf = float((lf.argmax(1) == yva[:50000].argmax(1)).mean())
            print(f"init from float: {n} layers, float val_acc={accf:.4f} post-transfer val_acc={acc0:.4f}", flush=True)
            assert acc0 > 0.9 * accf, "transfer collapsed: widen --bw_k/--bw_a/--i0_a"
        cbs = [ClampSoftmaxExpBits(), FreeEBOPs()] + cbs
        if a.ebops_target > 0:
            cbs.append(AttnBetaPID(a.ebops_target, a.beta0, a.beta_kp, a.beta_ki, a.beta_warmup, a.beta_max, a.beta_damp,
                                   attn_scale=a.attn_beta_scale))
        else:
            from hgq.utils.sugar import BetaScheduler, PieceWiseSchedule
            cbs.append(BetaScheduler(PieceWiseSchedule([(0, a.beta0, "linear"), (int(a.epochs * 2 / 7), 3e-7, "log"),
                                                        (a.epochs, 3e-6, "constant")])))
        os.makedirs(os.path.join(a.out, "pareto"), exist_ok=True)
        cbs.append(ParetoFront(os.path.join(a.out, "pareto"), [f"val_{mname}", "ebops"], [1, -1],
                               fname_format="epoch={epoch}-val_acc={val_%s:.4f}-ebops={ebops}.keras" % mname,
                               enable_if=lambda x: x[f"val_{mname}"] > 0.6))
        cbs.append(keras.callbacks.ModelCheckpoint(os.path.join(a.out, "latest.weights.h5"), save_weights_only=True))

    if a.w_l1 > 0:
        for l in model.layers:
            k = getattr(l, "kernel", None)
            if k is not None:
                model.add_loss(lambda k=k: a.w_l1 * ops.sum(ops.abs(k)))
    model.compile(optimizer=keras.optimizers.Adam(a.lr), loss=make_loss(a.distill_alpha, a.distill_T, a.label_smooth),
                  metrics=metrics, steps_per_execution=4)
    print(f"params {model.count_params()}  N={a.n}  train {xtr.shape}", flush=True)
    t0 = time.time()
    h = model.fit(xtr, ytr_fit, validation_data=(xva, yva_fit), batch_size=a.batch, epochs=a.epochs, verbose=2, callbacks=cbs)
    hist = {k: [float(v) for v in vs] for k, vs in h.history.items()}
    hist["minutes"] = (time.time() - t0) / 60
    json.dump(hist, open(os.path.join(a.out, "hist.json"), "w"))
    if a.mode == "float":
        model.load_weights(os.path.join(a.out, "float.weights.h5"))
        model.save(os.path.join(a.out, "float.keras"))
    else:
        model.save(os.path.join(a.out, "qat_final.keras"))
    lg = model.predict(xva, batch_size=4096, verbose=0)
    acc = float((lg.argmax(1) == yva.argmax(1)).mean())
    lt = model.predict(xte, batch_size=4096, verbose=0)
    ev = evaluate(lt, yte)
    best = max(hist[f"val_{mname}"])
    print(f"DONE best_val={best:.4f} final_val={acc:.4f} test_acc={ev['test_acc']:.4f} auc={ev['test_auc']:.4f} "
          f"rej={ev['avg_bg_rej']:.1f} params={model.count_params()} {hist['minutes']:.0f} min", flush=True)
    json.dump(dict(best_val=best, final_val=acc, params=model.count_params(), minutes=hist["minutes"],
                   protocol="val=10% of train (seed 1234); test=260k held-out", **ev),
              open(os.path.join(a.out, "result.json"), "w"))


if __name__ == "__main__":
    main()
