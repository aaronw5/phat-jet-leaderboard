"""Dump a float run's logits on the TRAIN set (for distillation) and its val accuracy.

  env/bin/python teacher_logits.py <float_run_dir> [--n_student N]
Writes <float_run_dir>/train_logits.npy (float32, [620000, 5]). The student must use the same data file
(jets_150x3_kt.npz) and jet order; N may differ (the teacher sees its own N).
"""
import argparse
import json
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "jax")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from phat_variants import build_variant, kw_from_args, load_split

p = argparse.ArgumentParser()
p.add_argument("run_dir")
a = p.parse_args()
cfg = json.load(open(os.path.join(a.run_dir, "config.json")))
m = build_variant(quantized=False, **kw_from_args(cfg))
m.load_weights(os.path.join(a.run_dir, "float.weights.h5"))
N, sort = cfg["n"], cfg.get("sort", "kt")
xv, yv = load_split("val", N, sort)
lv = m.predict(xv, batch_size=4096, verbose=0)
acc = float((lv.argmax(1) == yv.argmax(1)).mean())
xt, _ = load_split("train", N, sort)
lt = m.predict(xt, batch_size=4096, verbose=0).astype(np.float32)
np.save(os.path.join(a.run_dir, "train_logits.npy"), lt)
print(f"teacher {a.run_dir}: val acc {100*acc:.2f}%, wrote train_logits.npy {lt.shape}")
