"""Monte Carlo verification of the closed-form SINR (Prop. 2, eq. (9)).

Simulates the masked chain under the proposition's assumptions (white
unit-variance embeddings, w_u = 1, exact per-frame normalization) with
the TRAINED masks from data/masks.csv, measures the per-user
demultiplexed SINR, and compares against the closed form with
gamma^2 = K. Writes data/sinr_mc_check.csv with the worst-case
disagreement in dB.

    python code/sinr_mc_check.py
"""
import csv
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "masks.csv"))))
U = len(rows)
L = len([k for k in rows[0] if k.startswith("m2_")])
M2 = np.array([[float(rows[u][f"m2_{i}"]) for i in range(L)] for u in range(U)])
M = np.sqrt(M2)                       # signs irrelevant for power statistics
beta = (M2 @ M2.T) / L

rng = np.random.default_rng(0)
T = 1024
FRAMES = 2000
out = [["active", "snr_db", "user", "model_db", "mc_db", "abs_err_db"]]
worst = 0.0
for K in range(1, U + 1):
    act = list(range(K))
    for snr in (0, 10, 20):
        rho = 10 ** (snr / 10)
        sig_p = np.zeros(K); int_p = np.zeros(K); noi_p = np.zeros(K)
        for _ in range(FRAMES):
            S = rng.standard_normal((K, T, L))          # white embeddings
            E = S * M[act][:, None, :]                  # masked
            F = E.sum(0)
            g2 = (F ** 2).mean()                        # exact per-frame norm
            noise = rng.standard_normal((T, L)) / np.sqrt(rho)
            for j, u in enumerate(act):
                mu = M[u][None, :]
                sig = (E[j] / np.sqrt(g2)) * mu
                itf = ((F - E[j]) / np.sqrt(g2)) * mu
                sig_p[j] += (sig ** 2).mean()
                int_p[j] += (itf ** 2).mean()
                noi_p[j] += ((noise * mu) ** 2).mean()
        for j, u in enumerate(act):
            mc = sig_p[j] / (int_p[j] + noi_p[j])
            sig_m = beta[u, u] / K
            int_m = sum(beta[u, v] for v in act if v != u) / K
            model = sig_m / (int_m + 1.0 / rho)         # |h|^2 = 1 (conditional form)
            e = abs(10 * np.log10(mc) - 10 * np.log10(model))
            worst = max(worst, e)
            out.append([K, snr, u + 1, round(10 * np.log10(model), 4),
                        round(10 * np.log10(mc), 4), round(e, 4)])
with open(os.path.join(ROOT, "data", "sinr_mc_check.csv"), "w", newline="") as f:
    csv.writer(f).writerows(out)
print(f"worst |model - MC| = {worst:.4f} dB over {len(out)-1} points")
