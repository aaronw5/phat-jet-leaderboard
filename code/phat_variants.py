"""PHAT-JeT trigger-scale variants, Keras 3 / HGQ2, synthesizable through da4ml.

Derived from github.com/aaronw5/phat-jet-hw scripts/phat_jet_k3.py (one PHAT block, static
GMP grid as one-hot scatter/gather, no LayerNorm, logits output). Adds the knobs needed to
search the CTL2 envelope (<=173k LUT, <30 stages @ 300 MHz):

  global_mode : "mha"   exact MHA over patch tokens (paper / July campaign)
                "mix"   token mixing with a constant-weight Dense over the NP*D flattened
                        tokens -> D (no activation x activation multiply)
                "mean"  single mean token broadcast through a Dense (cheapest)
                "none"
  local_mode  : "mha" | "none"
  gmp_mode    : "grid"  2-D one-hot grid (G^2 cells)      cost ~ 2*N*G^2*C
                "sep"   separable: eta-grid and phi-grid 1-D passes  cost ~ 4*N*G*C
                "none"
  gmp_channels: channels that pass through GMP (rest skip it); <= d_model
  pre_norm    : "tanh" bounded pre-activation on the embedding (July campaign) | None
  ffn_mult    : FFN hidden width multiplier (campaign used 1, paper 4)
  attn_bits   : if set, caps the fractional bits of Q and K data lanes (cheaper QK multiplies)

The same function builds the float model (quantized=False) and the HGQ model, so float
weights load 1:1 into the Q model by layer name.
"""
import numpy as _np
import keras
from keras import layers, ops


def _dense(units, q, activation=None, name=None):
    if q:
        from hgq.layers import QDense
        return QDense(units, activation=activation, name=name)
    return layers.Dense(units, activation=activation, name=name)


UNSHARED = {"on": False}
LUT = {"roles": set(), "d_hl": 8, "table": (6, 5)}  # roles: embed, ffn, mix, proj, head


def _ldense(role, units, q, activation=None, name=None, per_particle=False):
    """Dense, or HGQ learned-LUT dense (QDenseT: every input->output edge is a learned 1-D function stored as a
    lookup table) when `role` is enabled and the model is quantized."""
    if q and role in LUT["roles"]:
        from hgq.layers.table.dense import QDenseT
        return QDenseT(units, n_hl=1, d_hl=LUT["d_hl"], activation=activation, table_spec=LUT["table"], name=name)
    return (_pdense if per_particle else _dense)(units, q, activation=activation, name=name)


def _pdense(units, q, activation=None, name=None):
    """Per-particle Dense: shared weights (default) or position-specific weights when UNSHARED is on.
    Returns a callable taking the [B,N,F] tensor (N must be static for the per-slot kernel)."""
    if not UNSHARED["on"]:
        return _dense(units, q, activation=activation, name=name)

    def apply(x):
        N = x.shape[1]
        if q:
            from hgq.layers import QEinsumDense
            L = QEinsumDense("bnf,nfd->bnd", output_shape=(N, units), activation=activation, bias_axes="nd", name=name)
        else:
            L = layers.EinsumDense("bnf,nfd->bnd", output_shape=(N, units), activation=activation, bias_axes="nd", name=name)
        return L(x)

    return apply


def _einsum(eq, xs, q, name=None):
    if q:
        from hgq.layers import QEinsum
        return QEinsum(eq, name=name)(xs)
    return ops.einsum(eq, *xs)


def _softmax(x, q, name=None, exp_i_max=5):
    if q:
        from hgq.config import QuantizerConfig
        from hgq.constraints import MinMax
        from hgq.layers import QSoftmax
        exp_iq = QuantizerConfig("default", "datalane")
        exp_iq.config["ic"] = MinMax(-16, exp_i_max)
        return QSoftmax(axis=-1, name=name, exp_iq_conf=exp_iq)(x)
    return layers.Softmax(axis=-1, name=name)(x)


ATTN_FLOOR = {"f": None}  # set by train_variant: min fractional bits on attention data lanes


def _qdense_bits(units, q, fbits, name):
    """Dense whose OUTPUT data lane has fractional bits in [ATTN_FLOOR, fbits] (Q/K/V/O lanes).
    The floor stops the EBOPs penalty from collapsing the attention path to ~1 bit."""
    lo, hi = ATTN_FLOOR["f"], fbits
    if q and (lo is not None or hi is not None):
        from hgq.config import QuantizerConfig
        from hgq.constraints import MinMax
        from hgq.layers import QDense
        oq = QuantizerConfig("default", "datalane")
        oq.config["fc"] = MinMax(lo if lo is not None else -16, hi if hi is not None else 16)
        return QDense(units, name=name, oq_conf=oq)
    return _dense(units, q, name=name)


def _bin_indicator(lo, hi, first, last):
    if first:
        return lambda x: ops.cast(x < hi, x.dtype)
    if last:
        return lambda x: ops.cast(x >= lo, x.dtype)
    return lambda x: ops.cast((x >= lo) & (x < hi), x.dtype)


def onehot_bins(coord, edges, q, tag):
    edges = list(edges)
    nb = len(edges) - 1
    n = coord.shape[1]
    cols = []
    for g in range(nb):
        f = _bin_indicator(edges[g], edges[g + 1], g == 0, g == nb - 1)
        if q:
            from hgq.layers import QUnaryFunctionLUT
            c = QUnaryFunctionLUT(f, name=f"oh_{tag}_{g}")(coord)
        else:
            c = layers.Lambda(f, name=f"oh_{tag}_{g}")(coord)
        cols.append(ops.reshape(c, (-1, n, 1)))
    return layers.Concatenate(axis=-1, name=f"oh_{tag}")(cols)


def _dwconv(ch, k, q, name, rank=2):
    if q:
        from hgq.layers import QConv1D, QConv2D
        L = QConv2D if rank == 2 else QConv1D
    else:
        L = layers.Conv2D if rank == 2 else layers.Conv1D
    return L(ch, k, padding="same", groups=ch, name=name)


def gmp_grid(x, eta, phi, ch, q, edges, k, name="gmp"):
    eo = onehot_bins(eta, edges, q, f"{name}_eta")
    po = onehot_bins(phi, edges, q, f"{name}_phi")
    cell = _einsum("bne,bnp->bnep", [eo, po], q, name=f"{name}_cell")
    grid = _einsum("bnep,bnc->bepc", [cell, x], q, name=f"{name}_scatter")
    conv = _dwconv(ch, k, q, f"{name}_dwconv")(grid)
    return _einsum("bnep,bepc->bnc", [cell, conv], q, name=f"{name}_gather")


def gmp_sep(x, eta, phi, ch, q, edges, k, name="gmp"):
    """Separable GMP: 1-D scatter/conv/gather along eta, then along phi, summed."""
    out = []
    for coord, tag in ((eta, "eta"), (phi, "phi")):
        oh = onehot_bins(coord, edges, q, f"{name}_{tag}")
        line = _einsum("bne,bnc->bec", [oh, x], q, name=f"{name}_{tag}_scatter")
        conv = _dwconv(ch, k, q, f"{name}_{tag}_dwconv", rank=1)(line)
        out.append(_einsum("bne,bec->bnc", [oh, conv], q, name=f"{name}_{tag}_gather"))
    return out[0] + out[1]


def patch_mha(x4, d, nh, q, name, attn_bits=None, share_qk=False):
    NP, P = x4.shape[1], x4.shape[2]
    dh = d // nh
    qv = _qdense_bits(d, q, attn_bits, f"{name}_wq")(x4)
    kv = qv if share_qk else _qdense_bits(d, q, attn_bits, f"{name}_wk")(x4)
    vv = _qdense_bits(d, q, None, f"{name}_wv")(x4)
    qv = ops.reshape(qv, (-1, NP, P, nh, dh)); kv = ops.reshape(kv, (-1, NP, P, nh, dh)); vv = ops.reshape(vv, (-1, NP, P, nh, dh))
    s = _einsum("bnphd,bnqhd->bnhpq", [qv, kv], q, name=f"{name}_qk")
    w = _softmax(s, q, name=f"{name}_softmax")
    o = _einsum("bnhpq,bnqhd->bnphd", [w, vv], q, name=f"{name}_av")
    o = ops.reshape(o, (-1, NP, P, d))
    return _qdense_bits(d, q, None, f"{name}_wo")(o)


def token_mha(x, d, nh, q, name, attn_bits=None, share_qk=False):
    T = x.shape[1]
    dh = d // nh
    qv = _qdense_bits(d, q, attn_bits, f"{name}_wq")(x)
    kv = qv if share_qk else _qdense_bits(d, q, attn_bits, f"{name}_wk")(x)
    vv = _qdense_bits(d, q, None, f"{name}_wv")(x)
    qv = ops.reshape(qv, (-1, T, nh, dh)); kv = ops.reshape(kv, (-1, T, nh, dh)); vv = ops.reshape(vv, (-1, T, nh, dh))
    s = _einsum("bthd,bshd->bhts", [qv, kv], q, name=f"{name}_qk")
    w = _softmax(s, q, name=f"{name}_softmax")
    o = _einsum("bhts,bshd->bthd", [w, vv], q, name=f"{name}_av")
    return _qdense_bits(d, q, None, f"{name}_wo")(ops.reshape(o, (-1, T, d)))


def _seq_proj(x, k, q, name):
    """Constant-weight projection along the particle axis: [B,N,D] -> [B,k,D] (Linformer E/F)."""
    N, D = x.shape[1], x.shape[2]
    if q:
        from hgq.layers import QEinsumDense
        return QEinsumDense("bnd,nk->bkd", output_shape=(k, D), name=name)(x)
    return layers.EinsumDense("bnd,nk->bkd", output_shape=(k, D), name=name)(x)


def linformer_attn(x, d, nh, k, q, name, attn_bits=None, share_qk=False, share_ef=True):
    """Single-stage Linformer attention: Q from all N particles, K/V projected to k rows with constant
    E/F matrices (shared E=F by default). Activation x activation cost: 2*N*k*d (k << N)."""
    N = x.shape[1]
    dh = d // nh
    qv = _qdense_bits(d, q, attn_bits, f"{name}_wq")(x)
    kv = qv if share_qk else _qdense_bits(d, q, attn_bits, f"{name}_wk")(x)
    vv = _qdense_bits(d, q, None, f"{name}_wv")(x)
    kp = _seq_proj(kv, k, q, f"{name}_E")
    vp = kp if (share_qk and share_ef) else _seq_proj(vv, k, q, f"{name}_F")
    if share_qk and share_ef:  # K==V rows share one projection; V needs its own values though
        vp = _seq_proj(vv, k, q, f"{name}_F")
    qv = ops.reshape(qv, (-1, N, nh, dh)); kp = ops.reshape(kp, (-1, k, nh, dh)); vp = ops.reshape(vp, (-1, k, nh, dh))
    s_ = _einsum("bnhd,bkhd->bhnk", [qv, kp], q, name=f"{name}_qk")
    w = _softmax(s_, q, name=f"{name}_softmax")
    o = _einsum("bhnk,bkhd->bnhd", [w, vp], q, name=f"{name}_av")
    return _qdense_bits(d, q, None, f"{name}_wo")(ops.reshape(o, (-1, N, d)))


def build_variant(
    num_particles=64, d_model=16, num_heads=4, patch_size=8, n_classes=5, quantized=False,
    gmp_mode="grid", gmp_edges=None, gmp_bins=8, gmp_bounds=1.6, gmp_kernel=3, gmp_channels=None,
    local_mode="mha", global_mode="mha", ffn_mult=1, pre_norm="tanh", attn_bits=None,
    parallel_attn=False, attn_particles=None, share_qk=False, use_head1=True, linf_k=4, unshared=False,
    ffn_hidden=None, mix_hidden=None, n_blocks=1, lut_layers=None, lut_dhl=8, name=None,
):
    UNSHARED["on"] = bool(unshared)
    LUT["roles"] = set((lut_layers or "").split(",")) - {""}
    LUT["d_hl"] = lut_dhl
    """attn_particles: if set (< num_particles), local attention runs only over the leading
    `attn_particles` (pT/kT-ordered head); the soft tail bypasses it (physics-informed pruning)."""
    q = quantized
    NP = num_particles // patch_size
    assert NP * patch_size == num_particles
    gmp_channels = gmp_channels or d_model
    if gmp_edges is None:
        gmp_edges = _np.linspace(-gmp_bounds, gmp_bounds, gmp_bins + 1).tolist()

    feats = keras.Input((num_particles, 3), name="features")
    eta, phi = feats[..., 1], feats[..., 2]
    # da4ml allows only ReLU inline; any other activation must be its own unary-LUT layer.
    if pre_norm in (None, "relu"):
        x = _pdense(d_model, q, activation=pre_norm, name="embed")(feats)
    else:
        x = _pdense(d_model, q, name="embed")(feats)
        if q:
            from hgq.layers import QUnaryFunctionLUT
            x = QUnaryFunctionLUT(getattr(ops, pre_norm), name="embed_act")(x)
        else:
            x = layers.Activation(pre_norm, name="embed_act")(x)

    for bi in range(n_blocks):
        S = "" if bi == 0 else f"_b{bi}"
        if gmp_mode != "none":
            xin = x if gmp_channels == d_model else x[..., :gmp_channels]
            fn = gmp_grid if gmp_mode == "grid" else gmp_sep
            msg = fn(xin, eta, phi, gmp_channels, q, gmp_edges, gmp_kernel, name="gmp" + S)
            msg = _ldense("proj", d_model, q, name="gmp_pointwise" + S, per_particle=True)(msg)
            x = x + msg

        x0 = x
        if local_mode == "linf":  # Linformer over the leading K particles (global, no patches)
            K = attn_particles or num_particles
            xa = x if K == num_particles else x[..., :K, :]
            a = linformer_attn(xa, d_model, num_heads, linf_k, q, "local_attn" + S, attn_bits, share_qk)
            if K < num_particles:  # residual on the leading K only; the soft tail passes through (no zeros_like: not traceable)
                x = layers.Concatenate(axis=-2, name="attn_pad" + S)([x[..., :K, :] + a, x[..., K:, :]])
            else:
                x = x + a
        if local_mode == "mha":
            K = attn_particles or num_particles
            assert K % patch_size == 0 and K <= num_particles
            xa = x if K == num_particles else x[..., :K, :]
            h4 = ops.reshape(xa, (-1, K // patch_size, patch_size, d_model))
            a = patch_mha(h4, d_model, num_heads, q, "local_attn" + S, attn_bits, share_qk)
            a = ops.reshape(a, (-1, K, d_model))
            if K < num_particles:  # residual on the leading K only; the soft tail passes through (no zeros_like: not traceable)
                x = layers.Concatenate(axis=-2, name="attn_pad" + S)([x[..., :K, :] + a, x[..., K:, :]])
            else:
                x = x + a

        if global_mode != "none":
            src = x0 if parallel_attn else x
            h4 = ops.reshape(src, (-1, NP, patch_size, d_model))
            if q:
                from hgq.layers import QSum
                tok = QSum(axes=2, scale=1.0 / patch_size, name="patch_tokens" + S)(h4)
            else:
                tok = ops.sum(h4, axis=2) * (1.0 / patch_size)
            if global_mode == "mha":
                tok = token_mha(tok, d_model, num_heads, q, "patch_attn" + S, attn_bits, share_qk)
                tok = _dense(d_model, q, name="patch_msg_proj" + S)(tok)
                msg = ops.repeat(tok, patch_size, axis=1)
            elif global_mode == "mix":
                flat = ops.reshape(tok, (-1, NP * d_model))
                mixed = _ldense("mix", mix_hidden or NP * d_model, q, activation="relu", name="patch_mix" + S)(flat)
                if mix_hidden and mix_hidden != NP * d_model:  # low-rank mixer: expand back to NP*d before the broadcast
                    mixed = _dense(NP * d_model, q, name="patch_mix2" + S)(mixed)
                mixed = ops.reshape(mixed, (-1, NP, d_model))
                tok = _dense(d_model, q, name="patch_msg_proj" + S)(mixed)
                msg = ops.repeat(tok, patch_size, axis=1)
            elif global_mode == "mean":
                if q:
                    from hgq.layers import QSum
                    g = QSum(axes=1, scale=1.0 / NP, name="global_token" + S)(tok)
                else:
                    g = ops.sum(tok, axis=1) * (1.0 / NP)
                g = _ldense("mix", d_model, q, activation="relu", name="patch_mix" + S)(g)
                g = _dense(d_model, q, name="patch_msg_proj" + S)(g)
                msg = ops.repeat(ops.reshape(g, (-1, 1, d_model)), num_particles, axis=1)
            else:
                raise ValueError(global_mode)
            x = x + msg

        if ffn_hidden or ffn_mult:
            h = _ldense("ffn", ffn_hidden or ffn_mult * d_model, q, activation="relu", name="ffn1" + S, per_particle=True)(x)
            x = x + _ldense("ffn", d_model, q, name="ffn2" + S, per_particle=True)(h)

    if q:
        from hgq.layers import QSum
        pooled = QSum(axes=1, scale=1.0 / num_particles, name="agg_mean")(x)
    else:
        pooled = layers.GlobalAveragePooling1D(name="agg_mean")(x)
    hd = _ldense("head", d_model, q, activation="relu", name="head1")(pooled) if use_head1 else pooled
    logits = _ldense("head", n_classes, q, name="head_out")(hd)
    return keras.Model(feats, logits, name=name or ("phat_var" + ("_q" if q else "")))


def transfer_weights(qmodel, fmodel):
    """Copy float weights into same-named Q layers (Q layers carry extra quantizer vars)."""
    fw = {l.name: l.get_weights() for l in fmodel.layers if l.get_weights()}
    n = 0
    for layer in qmodel.layers:
        if layer.name not in fw:
            continue
        src, dst = fw[layer.name], layer.get_weights()
        rep, si = list(dst), 0
        for di, w in enumerate(dst):
            if si < len(src) and src[si].shape == w.shape:
                rep[di] = src[si]; si += 1
        if si == len(src):
            layer.set_weights(rep); n += 1
    return n


# ----------------------------------------------------------------------------- build from a run config
def kw_from_args(a):
    """Model kwargs from a train_variant argparse namespace / config.json dict."""
    g = a.get if isinstance(a, dict) else (lambda k, d=None: getattr(a, k, d))
    edges = g("gmp_edges")
    if isinstance(edges, str):
        edges = [float(e) for e in edges.split(",")]
    return dict(num_particles=g("n"), d_model=g("d_model"), num_heads=g("heads"), patch_size=g("patch"),
                gmp_mode=g("gmp"), gmp_edges=edges, gmp_bins=g("gmp_bins"), gmp_bounds=g("gmp_bounds"),
                gmp_kernel=g("gmp_kernel"), gmp_channels=g("gmp_channels") or None, local_mode=g("local"),
                global_mode=g("global_mode"), ffn_mult=g("ffn_mult"), pre_norm=g("pre_norm") or None,
                attn_bits=g("attn_bits"), parallel_attn=bool(g("parallel_attn")),
                attn_particles=g("attn_particles") or None, share_qk=bool(g("share_qk")),
                use_head1=not g("no_head1"), linf_k=g("linf_k") or 4, unshared=bool(g("unshared")),
                ffn_hidden=g("ffn_hidden") or None, mix_hidden=g("mix_hidden") or None,
                n_blocks=g("n_blocks") or 1, lut_layers=g("lut_layers") or None, lut_dhl=g("lut_dhl") or 8)


def build_q_from_args(a):
    """Quantized model with the exact HGQ2 quantizer scopes train_variant used (needed to load_weights
    from a .keras file: lambda-based GMP indicators cannot be deserialized by load_model)."""
    from hgq.config import LayerConfigScope, QuantizerConfigScope
    from hgq.constraints import MinMax
    from hgq.regularizers import MonoL1
    g = a.get if isinstance(a, dict) else (lambda k, d=None: getattr(a, k, d))
    floor = g("floor_bits") or 0.0
    fc = MinMax(floor, 16) if floor > 0 else None
    l1 = g("l1_reg") if g("l1_reg") is not None else 1e-8
    s0 = QuantizerConfigScope(default_q_type="kbi", b0=g("bw_k"), overflow_mode="wrap", i0=g("i0_w"),
                              fr=MonoL1(l1), ir=MonoL1(l1), i_decay_speed=1e-3, **({"fc": fc} if fc else {}))
    s1 = QuantizerConfigScope(default_q_type="kif", place="datalane", overflow_mode="wrap", f0=g("bw_a"),
                              i0=g("i0_a"), fr=MonoL1(l1), ic=MinMax(0, 12), **({"fc": fc} if fc else {}))
    ATTN_FLOOR["f"] = g("attn_floor") if (g("attn_floor") or 0) > 0 else None
    with s0, s1, LayerConfigScope(beta0=0):
        return build_variant(quantized=True, **kw_from_args(a))


# ----------------------------------------------------------------------------- data
DATA = "/j-jepa-vol/phatjet-fpga/jets_150x3_kt.npz"  # paper units, kT-sorted (pT*dR desc), padding at the end


def sort_jets(x, sort):
    """Re-order constituents per jet. 'kt' = as stored; 'pt' = descending pT (the order CTL2 delivers, no sorter)."""
    if sort in (None, "kt"):
        return x
    if sort == "pt":
        idx = _np.argsort(-x[:, :, 0], axis=1, kind="stable")  # padding has pT=0 -> stays at the end
        return _np.take_along_axis(x, idx[:, :, None], axis=1)
    raise ValueError(sort)


def load_split(split, n, sort="kt"):
    """x_train/y_train (620k) or x_val/y_val (260k held-out test) sliced to the leading n particles."""
    d = _np.load(DATA)
    x = sort_jets(d[f"x_{split}"].astype(_np.float32), sort)[:, :n]
    return x, d[f"y_{split}"].astype(_np.float32)


# ----------------------------------------------------------------------------- metrics (paper protocol)
CLASSES = ["q", "g", "W", "Z", "t"]


def evaluate(logits, y):
    """Accuracy, macro OvO AUC, and background rejection at 80% signal efficiency for W/Z/t
    (background = q,g jets, as in the paper's test script). logits/y: [M,5]."""
    from sklearn.metrics import roc_auc_score, roc_curve
    p = _np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    yi = y.argmax(1)
    out = dict(test_acc=float((p.argmax(1) == yi).mean()),
               test_auc=float(roc_auc_score(y, p, average="macro", multi_class="ovo")))
    rej = {}
    for i in (2, 3, 4):
        m = (yi == 0) | (yi == 1) | (yi == i)
        fpr, tpr, _ = roc_curve((yi[m] == i).astype(int), p[m, i])
        f = fpr[_np.argmin(_np.abs(tpr - 0.8))]
        rej[CLASSES[i]] = float(1.0 / f) if f > 0 else float("inf")
    out["bg_rej"] = rej
    out["avg_bg_rej"] = float(_np.mean(list(rej.values())))
    return out
