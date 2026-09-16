"""Provenance check of data/masks.csv (audit, 2026-09-08): which checkpoint did it come from?

Loads the mask tensor of the variable-load learned-mask checkpoints
(state_dict key `masks.W`, an nn.Parameter of shape (users, L_e) holding the
RAW masks; the mask used in the chain is W[u] / ||W[u]|| * sqrt(L_e), and
masks.csv stores its SQUARE, so that every row sums to L_e), recomputes the
masks.csv rows with the formula of dump_artifacts.py and reports the maximum
absolute difference to the stored file for each candidate checkpoint.

    PYTHONPATH=~/ViT python scripts/verify_masks.py
"""
import csv, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_main import MAIN, wpath

CK = wpath("checkpoints")
stored = list(csv.reader(open(wpath("data", "masks.csv"))))
hdr, body = stored[0], stored[1:]
S = torch.tensor([[float(v) for v in r[1:]] for r in body], dtype=torch.float64)
print("stored data/masks.csv: %d users x %d dims, header %s" % (S.shape[0], S.shape[1], hdr))
print("  row sums (should equal L_e):", ["%.4f" % v for v in S.sum(1).tolist()])
for name in ("swinsc_ov_u4var_learned", "swinsc_ov_u8var_learned", "swinsc_ov_u4_learned", "swinsc_ov_u2var_learned"):
    f = os.path.join(CK, name, "tx.pt")
    if not os.path.exists(f):
        print("%s: absent" % name); continue
    sd = torch.load(f, map_location="cpu")
    keys = [k for k in sd if k.startswith("masks.")]
    W = sd["masks.W"].double()
    U, L = W.shape
    M2 = (W / W.norm(dim=1, keepdim=True) * L ** 0.5) ** 2
    print("%s: keys %s, masks.W shape %s dtype %s" % (name, keys, tuple(W.shape), sd["masks.W"].dtype))
    if M2.shape == S.shape:
        print("  max |recomputed - stored| = %.3g   (stored file has 6 significant digits)" % (M2 - S).abs().max().item())
        print("  recomputed row 0:", ["%.6g" % v for v in M2[0].tolist()])
    else:
        print("  shape differs from stored file -> cannot be its source")
        print("  recomputed row 0:", ["%.6g" % v for v in M2[0].tolist()])
