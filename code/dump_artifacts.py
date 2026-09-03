# -*- coding: utf-8 -*-
"""Dump the checkpoint-derived auxiliary artifacts the manuscript quotes.

    python code/dump_artifacts.py            # writes into <workspace>/data/

  masks.csv         normalized squared masks m_u^2(i) of the variable-load
                    N=4 model (||m_u||^2 = L), one row per user  -> Fig. 2,
                    Fig. 3 (via sinr_model.py), Sec. V-C overlaps, the
                    appendix deviation (gamma_concentration.py), the
                    Monte Carlo check (sinr_mc_check.py)
  mux_weights.csv   trained per-user superposition weights w_u    -> Sec. V-C
  param_counts.csv  parameter counts of the three chains          -> Sec. VI-A

These used to be produced by one-off commands on the GPU host, which is the
"discarded script" the writing standard (2.2) forbids: after the receiver
fix of 2026-09-02 every one of them had to be regenerated and no script
existed to do it. Reads only the stored checkpoints; no GPU needed.
"""
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_main import MAIN, wpath   # the ONE configuration (standard 7.9)
from swinsc import Config, Transmitter, Receiver
from swinsc.swin import SwinEncoder, SwinDecoder

CK = wpath("checkpoints")
OUT = wpath("data")
os.makedirs(OUT, exist_ok=True)
N = MAIN["N"][-1]
VAR = "swinsc_ov_u%dvar_learned" % N


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def load_pair(name):
    d = os.path.join(CK, name)
    cfg = Config.load(os.path.join(d, "config.json"))
    tx = Transmitter(cfg); tx.load_state_dict(torch.load(os.path.join(d, "tx.pt"), map_location="cpu"))
    rx = Receiver(cfg); rx.load_state_dict(torch.load(os.path.join(d, "rx.pt"), map_location="cpu"))
    return cfg, tx, rx


# ---- masks and weights of the variable-load model ---------------------------
cfg, tx, rx = load_pair(VAR)
W = tx.masks.W.detach()                                   # (U, L) raw masks
U, L = W.shape
M = W / W.norm(dim=1, keepdim=True) * L ** 0.5            # ||m_u||^2 = L
M2 = (M ** 2)
with open(os.path.join(OUT, "masks.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["user"] + ["m2_%d" % i for i in range(L)])
    for u in range(U):
        w.writerow([u] + ["%.6g" % v for v in M2[u].tolist()])   # 6 significant digits: a 1e-4 entry must not round to 0
print("masks.csv: U=%d L=%d  beta diag %s" % (U, L, [round(float((M2[u] * M2[u]).sum() / L), 3) for u in range(U)]))

wts = tx.mux.w.detach() if getattr(tx.mux, "w", None) is not None else torch.ones(U)
with open(os.path.join(OUT, "mux_weights.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["user", "w"])
    for u in range(U):
        w.writerow([u + 1, "%.5f" % float(wts[u])])
print("mux_weights.csv:", ["%.4f" % float(v) for v in wts])

# ---- parameter counts of the three chains -----------------------------------
rows = []
for name, chain in (("swinsc_ov_u%d_learned" % N, "masked_fixed_u%d" % N),
                    ("swinsc_ov_u%dvar_learned" % N, "masked_var_u%d" % N),
                    ("swinsc_ov_u%d_oma" % N, "oma_u%d" % N),
                    ("swinsc_ov_u%d_deepma" % N, "deepma_u%d" % N)):
    if not os.path.exists(os.path.join(CK, name, "tx.pt")):
        print("param_counts: no checkpoint for %s, skipped" % name); continue
    c, t, r = load_pair(name)
    rows.append([chain, "tx_total", n_params(t)])
    rows.append([chain, "rx_total_allusers", n_params(r)])
td = os.path.join(CK, "todma_v256")
st = torch.load(os.path.join(td, "model.pt"), map_location="cpu")
cfgT = Config.load(os.path.join(td, "config.json"))
V, D = st["v"], st["l_s"]
enc = SwinEncoder(cfgT.in_ch, cfgT.patch, cfgT.dims, cfgT.depths, cfgT.heads, cfgT.window, D)
dec = SwinDecoder(cfgT.in_ch, cfgT.patch, tuple(reversed(cfgT.dims)), tuple(reversed(cfgT.depths)),
                  tuple(reversed(cfgT.heads)), cfgT.window, D)
rows += [["todma_v%d" % V, "enc", n_params(enc)], ["todma_v%d" % V, "dec", n_params(dec)],
         ["todma_v%d" % V, "codebook", int(st["vq"]["codebook"].numel())]]
with open(os.path.join(OUT, "param_counts.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["chain", "side", "params"]); w.writerows(rows)
for r in rows:
    print("param_counts.csv: %-18s %-18s %d" % tuple(r))
