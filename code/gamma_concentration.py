# -*- coding: utf-8 -*-
"""Concentration of the frame normalization, discussed in Remark 1.

Under the appendix assumptions (zero-mean, uncorrelated, unit-variance,
independent across tokens, Gaussian fourth moments),

    Var(Y_t) = (2/L^2) * sum_i ( sum_{u in K} m_u^2(i) )^2 ,
    Var(gamma^2) = Var(Y_t) / T ,

so the relative root-mean-square deviation of gamma^2 from its mean K is
sqrt(Var(Y_t)/T)/K. Reported per load and per active subset, because the
bound depends on WHICH users are active, not only on how many.

    python code/gamma_concentration.py
"""
import csv
import itertools
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_main import MAIN                      # noqa: E402

OUT = os.path.join(ROOT, "data", "gamma_concentration.csv")
MASKS = os.path.join(ROOT, "data", "masks.csv")

rows_m = list(csv.DictReader(open(MASKS)))
U = len(rows_m)
L = len([k for k in rows_m[0] if k.startswith("m2_")])
M2 = [[float(rows_m[u]["m2_%d" % i]) for i in range(L)] for u in range(U)]
T = 1024                                          # tokens per image, Table IV

out, worst = [], 0.0
for K in range(1, U + 1):
    for act in itertools.combinations(range(U), K):
        var_y = 2.0 / L ** 2 * sum(
            sum(M2[u][i] for u in act) ** 2 for i in range(L))
        rms = (var_y / T) ** 0.5 / K              # relative to E[gamma^2] = K
        worst = max(worst, rms)
        out.append([K, "+".join(str(u + 1) for u in act),
                    round(var_y, 6), round(100 * rms, 4)])

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["active", "subset", "var_Y_t", "rms_dev_percent"])
    w.writerows(out)

for K in range(1, U + 1):
    sel = [r[3] for r in out if r[0] == K]
    print("K=%d  rms deviation %.3f%% to %.3f%%" % (K, min(sel), max(sel)))
print("worst case over all subsets and loads: %.3f%%" % (100 * worst))
print("saved", OUT)
