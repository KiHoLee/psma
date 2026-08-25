"""Evaluate the matched-filter SINR model (Prop. 2) with the ACTUAL trained masks
of the variable-load model, averaged over Rayleigh fading by Monte Carlo.

    ~/tr_env/bin/python scripts/sinr_model.py --out ~/ViT/logs/sinr_model.csv
"""
import argparse, os, sys, csv
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default=os.path.expanduser("~/ViT/checkpoints/swinsc_ov_u4var_learned"))
p.add_argument("--snrs", type=float, nargs="+", default=[0, 10, 20])
p.add_argument("--mc", type=int, default=200000)
p.add_argument("--out", default=os.path.expanduser("~/ViT/logs/sinr_model.csv"))
a = p.parse_args()

sd = torch.load(os.path.join(a.ckpt, "tx.pt"), map_location="cpu")
W = sd["masks.W"]                                     # (U, L) learned masks
U, L = W.shape
M = W / W.norm(dim=1, keepdim=True) * L ** 0.5        # normalized masks, ||m||^2 = L
M2 = M ** 2
beta = (M2 @ M2.t()) / L                              # beta_{uv}
print("beta matrix:\n", beta.round(decimals=3))

g = torch.Generator().manual_seed(0)
h2 = ((torch.randn(a.mc, generator=g) ** 2 + torch.randn(a.mc, generator=g) ** 2) / 2)

rows = []
for K in range(1, U + 1):
    act = list(range(K))
    for snr in a.snrs:
        rho = 10 ** (snr / 10)
        vals = []
        for u in act:
            sig = beta[u, u] / K
            intf = sum(beta[u, v] for v in act if v != u) / K
            # E over h of SINR in dB (finite for every realization)
            sinr = sig / (intf + 1.0 / (rho * h2))
            vals.append((10 * torch.log10(sinr)).mean().item())
        rows.append([K, int(snr), round(sum(vals) / len(vals), 3)])
        print(K, snr, rows[-1][-1], flush=True)

with open(a.out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["active", "snr", "model_sinr_db"])
    w.writerows(rows)
print("saved", a.out)
