"""Coherence of the ToDMA signature dictionary quoted in Remark 3.

The dictionary is regenerated exactly as code/tsp_eval.py builds it (seed 2026,
V Gaussian rows normalized to squared norm D), so the number in the manuscript
has a stored artifact (writing standard 7.6).

    python code/todma_coherence.py
"""
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_main import MAIN   # the ONE configuration (standard 7.9)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "todma_coherence.csv")

V = MAIN["VOCAB"]
rows = []
for D in (MAIN["L"], MAIN["L_RICH"]):
    g = torch.Generator(device="cpu").manual_seed(MAIN["SIGNATURE_SEED"])
    S = torch.randn(V, D, generator=g)
    S = S / S.norm(dim=1, keepdim=True)          # unit-norm atoms
    G = (S @ S.t()).abs()
    G.fill_diagonal_(0.0)
    mu = G.max().item()
    off = G[~torch.eye(V, dtype=torch.bool)]
    mean_mu = off.mean().item()
    welch = ((V - D) / (D * (V - 1))) ** 0.5     # Welch bound for V atoms in D dims
    rows.append([D, V, round(mu, 4), round(mean_mu, 4), round(welch, 4)])
    print("D=%2d  max coherence %.4f   mean |corr| %.4f   Welch bound %.4f"
          % (D, mu, mean_mu, welch))

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dims", "vocab", "max_coherence", "mean_abs_corr", "welch_bound"])
    w.writerows(rows)
print("saved", OUT)
