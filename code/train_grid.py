#!/usr/bin/env python
"""PHAT-JeT (paper 'small' preset) with a pluggable GMP grid: dynamic | static | none.

Model follows github.com/aaronw5/PHAT-JeT models/PHAT_JeT.py ('small': d=16, 1 block,
4 heads, patch 10, cpe_k=8, patch messages on, mean tokenizer, mean aggregation, GELU),
plus a padding mask (|pT|>0) as in the paper's training runs.

GMP grids (coordinates are the paper's robust-scaled eta/phi; raw = scaled * 0.1211):
  dynamic : paper. Per-jet min-shift over real particles, cell = floor((x - min)/delta).
  static  : fixed bin edges (same for eta and phi). Out-of-range particles clamp to the
            edge bins. Built from --edges, or uniform from --grid_size/--extent/--align.
  none    : GMP removed.
Static GMP is computed with a scatter onto a fixed [G,G] grid, which is mathematically
identical to the one-hot/einsum form used in the FPGA port.
"""
import argparse
import json
import logging
import math
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

DATA = "/j-jepa-vol/l1-jet-id/data/jetid/processed"


# ----------------------------------------------------------------------------- grids
def uniform_edges(delta, extent, align):
    """Ascending edges covering [-extent, extent]. odd: a cell centred on 0; even: an edge at 0."""
    if align == "odd":
        k = int(round(extent / delta - 0.5))
        return ((np.arange(-k, k + 2) - 0.5) * delta).tolist()
    k = int(round(extent / delta))
    return (np.arange(-k, k + 1) * delta).tolist()


class GMP(layers.Layer):
    def __init__(self, channels, mode, kernel_size, grid_size=0.2, edges=None, **kw):
        super().__init__(**kw)
        self.channels, self.mode, self.k, self.delta = channels, mode, kernel_size, grid_size
        self.edges = None if edges is None else np.asarray(edges, np.float32)
        self.conv = layers.Conv2D(channels, kernel_size, padding="same", groups=channels, use_bias=True)
        self.pointwise = layers.Dense(channels)
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def _cells(self, eta, phi, mask):
        if self.mode == "dynamic":
            big = tf.constant(1e9, eta.dtype)
            emin = tf.reduce_min(tf.where(mask, eta, big), axis=1, keepdims=True)
            pmin = tf.reduce_min(tf.where(mask, phi, big), axis=1, keepdims=True)
            ge = tf.cast(tf.floor((eta - emin) / self.delta), tf.int32)
            gp = tf.cast(tf.floor((phi - pmin) / self.delta), tf.int32)
            ge = tf.where(mask, ge, 0)
            gp = tf.where(mask, gp, 0)
            H = tf.reduce_max(ge) + 1
            W = tf.reduce_max(gp) + 1
            return ge, gp, H, W
        inner = tf.constant(self.edges[1:-1])
        ge = tf.reduce_sum(tf.cast(eta[..., None] >= inner, tf.int32), -1)
        gp = tf.reduce_sum(tf.cast(phi[..., None] >= inner, tf.int32), -1)
        G = len(self.edges) - 1
        return ge, gp, G, G

    def call(self, x, eta, phi, mask):
        B, N, C = tf.shape(x)[0], tf.shape(x)[1], self.channels
        ge, gp, H, W = self._cells(eta, phi, mask)
        bidx = tf.tile(tf.range(B)[:, None], [1, N])
        idx = tf.stack([bidx, ge, gp], -1)
        xm = x * tf.cast(mask, x.dtype)[..., None]  # padding contributes nothing
        grid = tf.scatter_nd(tf.reshape(idx, [-1, 3]), tf.reshape(xm, [-1, C]), tf.stack([B, H, W, C]))
        grid = tf.ensure_shape(grid, [None, None, None, C])
        out = tf.gather_nd(self.conv(grid), idx)
        out = self.norm(self.pointwise(out))
        return x + out


# ----------------------------------------------------------------------------- blocks
def split_heads(x, h):
    b, t, d = tf.shape(x)[0], tf.shape(x)[1], x.shape[-1]
    return tf.transpose(tf.reshape(x, [b, t, h, d // h]), [0, 2, 1, 3])


def merge_heads(x, d):
    b, t = tf.shape(x)[0], tf.shape(x)[2]
    return tf.reshape(tf.transpose(x, [0, 2, 1, 3]), [b, t, d])


class MHA(layers.Layer):
    def __init__(self, d, h, **kw):
        super().__init__(**kw)
        self.d, self.h = d, h
        self.wq, self.wk, self.wv, self.wo = [layers.Dense(d) for _ in range(4)]

    def call(self, x, kmask):
        q, k, v = (split_heads(f(x), self.h) for f in (self.wq, self.wk, self.wv))
        s = tf.einsum("bhtd,bhTd->bhtT", q, k) / math.sqrt(self.d // self.h)
        s += (1.0 - tf.cast(kmask, s.dtype))[:, None, None, :] * -1e9
        o = tf.einsum("bhtT,bhTd->bhtd", tf.nn.softmax(s, -1), v)
        return self.wo(tf.ensure_shape(merge_heads(o, self.d), [None, None, self.d]))


class PHATBlock(layers.Layer):
    def __init__(self, d, h, P, gmp_kw, use_patch_messages=True, **kw):
        super().__init__(**kw)
        self.d, self.P = d, P
        self.gmp = None if gmp_kw["mode"] == "none" else GMP(d, **gmp_kw)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.local = MHA(d, h)
        self.use_pm = use_patch_messages
        if use_patch_messages:
            self.patch_attn = MHA(d, h)
            self.proj = layers.Dense(d)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ff1, self.ff2 = layers.Dense(4 * d, activation="gelu"), layers.Dense(d)

    def call(self, x, coords, mask):
        B, T, d, P = tf.shape(x)[0], tf.shape(x)[1], self.d, self.P
        NP = T // P
        if self.gmp is not None:
            x = self.gmp(x, coords[..., 0], coords[..., 1], mask)
        # local attention inside patches
        xp = tf.reshape(self.norm1(x), [B * NP, P, d])
        mp = tf.reshape(mask, [B * NP, P])
        y = tf.reshape(self.local(xp, mp), [B, T, d])
        x = x + y
        # patch tokens -> patch attention -> broadcast
        if self.use_pm:
            xn = tf.reshape(self.norm1(x), [B, NP, P, d])
            mf = tf.cast(tf.reshape(mask, [B, NP, P]), x.dtype)
            tok = tf.reduce_sum(xn * mf[..., None], 2) / tf.maximum(tf.reduce_sum(mf, 2), 1.0)[..., None]
            pmask = tf.reduce_sum(mf, 2) > 0
            msg = self.proj(self.patch_attn(tok, pmask))
            x = x + tf.reshape(tf.tile(msg[:, :, None, :], [1, 1, P, 1]), [B, T, d])
        x = x + self.ff2(self.ff1(self.norm2(x)))
        return x


def build_model(N, n_out, gmp_kw, d=16, h=4, P=10, use_patch_messages=True):
    inp = layers.Input((N, 3), name="features")
    mask = tf.abs(inp[..., 0]) > 0
    coords = inp[..., 1:3]
    x = layers.Dense(d, activation="relu")(inp)
    x = PHATBlock(d, h, P, gmp_kw, use_patch_messages)(x, coords, mask)
    mf = tf.cast(mask, x.dtype)[..., None]
    x = tf.reduce_sum(x * mf, 1) / tf.maximum(tf.reduce_sum(mf, 1), 1.0)
    x = layers.Dense(d, activation="relu")(x)
    out = layers.Dense(n_out, activation="softmax")(x)
    return Model(inp, out)


# ----------------------------------------------------------------------------- data
def kt_sort(x):
    key = x[:, :, 0] * np.sqrt(x[:, :, 1] ** 2 + x[:, :, 2] ** 2)
    idx = np.argsort(key, axis=1)[:, ::-1]
    return np.take_along_axis(x, idx[:, :, None], axis=1)


def load(split):
    x = np.load(f"{DATA}/x_{split}_robust_150const_ptetaphi.npy").astype(np.float32)
    y = np.load(f"{DATA}/y_{split}_robust_150const_ptetaphi.npy").astype(np.float32)
    return kt_sort(x), y


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--gmp", choices=["dynamic", "static", "none"], required=True)
    p.add_argument("--grid_size", type=float, default=0.2)
    p.add_argument("--extent", type=float, default=3.3)
    p.add_argument("--align", choices=["odd", "even"], default="odd")
    p.add_argument("--edges", type=str, default=None, help="comma-separated ascending edges (scaled units)")
    p.add_argument("--cpe_k", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--schedule", type=str, default="128:200,256:200,512:200,1024:200,2048:200,4096:400")
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--no_patch_messages", action="store_true")
    p.add_argument("--max_jets", type=int, default=0, help="smoke test: subsample train/test")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    logging.basicConfig(filename=os.path.join(a.out, "train.log"), filemode="w", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    log = logging.getLogger()
    log.addHandler(logging.StreamHandler())
    tf.keras.utils.set_random_seed(a.seed)
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)

    edges = None
    if a.gmp == "static":
        edges = [float(e) for e in a.edges.split(",")] if a.edges else uniform_edges(a.grid_size, a.extent, a.align)
    gmp_kw = dict(mode=a.gmp, kernel_size=a.cpe_k, grid_size=a.grid_size, edges=edges)
    log.info("args %s", vars(a))
    log.info("edges %s (G=%s)", None if edges is None else [round(e, 4) for e in edges],
             None if edges is None else len(edges) - 1)

    x, y = load("train")
    if a.max_jets:
        x, y = x[: a.max_jets], y[: a.max_jets]
    xtr, xva, ytr, yva = train_test_split(x, y, test_size=0.2, random_state=42)
    del x, y
    model = build_model(150, 5, gmp_kw, use_patch_messages=not a.no_patch_messages)
    model.compile(tf.keras.optimizers.Adam(1e-3), "categorical_crossentropy", metrics=["accuracy"])
    n_params = model.count_params()
    log.info("params %d", n_params)

    ck = os.path.join(a.out, "best.weights.h5")
    hist, ep0, t0 = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": [], "bs": []}, 0, time.time()
    for stage in a.schedule.split(","):
        bs, ne = map(int, stage.split(":"))
        tf.keras.backend.set_value(model.optimizer.learning_rate, 1e-3)
        cbs = [tf.keras.callbacks.ModelCheckpoint(ck, monitor="val_loss", save_best_only=True, save_weights_only=True),
               tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=a.patience, restore_best_weights=True)]
        if os.path.exists(ck):
            cbs[0].best = min(hist["val_loss"])
        with tf.device("/CPU:0"):  # keep the dataset on the host; only batches go to the GPU
            dtr = tf.data.Dataset.from_tensor_slices((xtr, ytr)).shuffle(len(xtr), seed=a.seed + ep0,
                  reshuffle_each_iteration=True).batch(bs).prefetch(tf.data.AUTOTUNE)
            dva = tf.data.Dataset.from_tensor_slices((xva, yva)).batch(4096).prefetch(tf.data.AUTOTUNE)
        h = model.fit(dtr, validation_data=dva, initial_epoch=ep0, epochs=ep0 + ne, callbacks=cbs, verbose=2)
        for kk in ["loss", "val_loss", "accuracy", "val_accuracy"]:
            hist[kk] += [float(v) for v in h.history[kk]]
        hist["bs"] += [bs] * len(h.history["loss"])
        ep0 += len(h.history["loss"])
        log.info("stage bs=%d done: %d epochs, best val_loss so far %.5f, %.1f min", bs, len(h.history["loss"]),
                 min(hist["val_loss"]), (time.time() - t0) / 60)
        json.dump(hist, open(os.path.join(a.out, "history.json"), "w"))
    model.load_weights(ck)

    xte, yte = load("val")  # the paper's held-out test set (260k jets)
    if a.max_jets:
        xte, yte = xte[: a.max_jets], yte[: a.max_jets]
    with tf.device("/CPU:0"):
        dte = tf.data.Dataset.from_tensor_slices(xte).batch(4096)
    pr = model.predict(dte, verbose=0)
    acc = float((pr.argmax(1) == yte.argmax(1)).mean())
    auc = float(roc_auc_score(yte, pr, average="macro", multi_class="ovo"))
    rej = {}
    for i, lab in enumerate(["q", "g", "W", "Z", "t"]):
        if i < 2:
            continue
        m = (yte[:, 0] == 1) | (yte[:, 1] == 1) | (yte[:, i] == 1)
        fpr, tpr, _ = roc_curve(yte[m, i], pr[m, i])
        f = fpr[np.argmin(np.abs(tpr - 0.8))]
        rej[lab] = float(1 / f) if f > 0 else float("inf")
    res = dict(vars(a), edges=edges, params=n_params, test_acc=acc, test_auc=auc, bg_rej=rej,
               avg_bg_rej=float(np.mean(list(rej.values()))), epochs=ep0, minutes=(time.time() - t0) / 60)
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    log.info("RESULT %s", json.dumps(res))


if __name__ == "__main__":
    main()
