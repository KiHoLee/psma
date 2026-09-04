"""Quality of the PSMA progressive code versus prefix length (manuscript Fig. 8).

One user alone on the frame transmits the first b symbols of its code on the
first b rows of Q, b = 1..L, at UNIT power per symbol (the power a user
holding b codes has at load K = L/b under the normalization of (8)), passes
the Rayleigh block-fading channel, and is decoded from the zero-padded prefix
with the prefix-length conditioning b. This is the successive-refinement
curve D(b) of the trained code, measured for the N=4 and N=8 PSMA models on
the same 200 validation images as ser_eval.py, and it is directly comparable
with the stand-alone heads of dynamic WH-OMA at K = L/b.

Rows: designed, prefix, snr, psnr
    python code/prefix_eval.py --out data/prefix_eval.csv
"""
import argparse, os, sys, glob, random, csv
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_main import MAIN, device as MAIN_DEVICE, wpath
from swinsc import Config, Transmitter, Receiver, Channel
from swinsc.mask_mux import power_normalize
from deepsc_ri.metrics import psnr

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=MAIN["VAL_IMAGES"])
p.add_argument("--reps", type=int, default=MAIN["EVAL_REPS"],
               help="independent fading draws per image (common across prefix lengths and models)")
p.add_argument("--snrs", type=float, nargs="+", default=list(MAIN["SNR_GRID"]))
p.add_argument("--crop", type=int, default=MAIN["CROP"])
p.add_argument("--out", default=wpath("data", "prefix_eval.csv"))
a = p.parse_args()
dev = MAIN_DEVICE()
random.seed(MAIN["SEED"]); torch.manual_seed(MAIN["SEED"])


def batches():
    return [(rep, i) for rep in range(a.reps) for i in range(0, a.n, 25)]


def seed_fading(K, s, rep, b):
    """Same common-random-number rule as ser_eval.py (prefix length in the
    role of the load), so every prefix and both models see one fading sequence."""
    torch.manual_seed(MAIN["SEED"] * 1_000_003 + K * 10_007 + (int(s) + 5) * 101 + rep * 13 + b // 25)
CK = wpath("checkpoints")
val = sorted(glob.glob(wpath("data", "imagenette160", "val", "*", "*.png")))
random.shuffle(val)


def load(path):
    im = Image.open(path).convert("RGB"); w, h = im.size; s = a.crop
    l, t = (w - s) // 2, (h - s) // 2
    return torch.from_numpy(__import__("numpy").array(im.crop((l, t, l + s, t + s)))).permute(2, 0, 1).float() / 255


imgs = torch.stack([load(val[i]) for i in range(a.n)])      # user 0's images, as in ser_eval.py
rows = []
# Ablation of the per-prefix term (author request, 2026-09-04): the PSMA
# models trained with the term over every length b = 1..L (canonical names),
# the earlier models whose term visited the powers of two only (parked under
# checkpoints_psma_pow2_20260904/ by run_psma_all.sh), and the first
# prototype trained without the term (scheme prog_v1). Missing checkpoints
# are skipped, so the script runs before and after the retraining.
POW2 = os.path.join(os.path.dirname(CK), "checkpoints_psma_pow2_20260904")
MODELS = [("psma", 4, os.path.join(CK, "swinsc_ov_u4_psma")),
          ("psma", 8, os.path.join(CK, "swinsc_ov_u8_psma")),
          ("psma_pow2", 4, os.path.join(POW2, "swinsc_ov_u4_psma")),
          ("psma_pow2", 8, os.path.join(POW2, "swinsc_ov_u8_psma")),
          ("prog_v1", 8, os.path.join(CK, "swinsc_ov_u8_prog"))]
for model, N, d in MODELS:
    if not os.path.exists(os.path.join(d, "tx.pt")):
        print("skip (no checkpoint):", d); continue
    cfg = Config.load(os.path.join(d, "config.json"))
    tx = Transmitter(cfg).to(dev).eval(); tx.load_state_dict(torch.load(os.path.join(d, "tx.pt"), map_location=dev))
    rx = Receiver(cfg).to(dev).eval(); rx.load_state_dict(torch.load(os.path.join(d, "rx.pt"), map_location=dev))
    L = cfg.l_e; Q = tx.spreader.Q
    for b in range(1, L + 1):
        for s in a.snrs:
            ch = Channel("rayleigh", s); ps = []
            with torch.no_grad():
                for rep, i in batches():
                    x = imgs[i:i + 25].to(dev)
                    seed_fading(b, s, rep, i)
                    e, hw = tx.encode_user(x, 0)                       # (B, T, L) ordered symbols
                    # prefix b on the first b rows at UNIT power per symbol, the
                    # power a user holding b codes has at load K = L/b in (8);
                    # power_normalize alone would hand the lone prefix the whole
                    # frame power (L/b per symbol), which no load delivers
                    f = power_normalize(e[..., :b] @ Q[:b]) * (b / L) ** 0.5
                    z = ch(f)
                    y = torch.nn.functional.pad(z @ Q[:b].T, (0, L - b))
                    dec = rx.decoders[0]
                    out = dec(rx.reduce[0](y), hw, **({"K": b} if dec.cond is not None else {}))
                    ps.append(psnr(out, x).cpu())
            rows.append([model, N, b, int(s), round(torch.cat(ps).mean().item(), 3)])
            print("%s N=%d prefix=%d snr=%d psnr=%.2f" % (model, N, b, s, rows[-1][4]), flush=True)

with open(a.out, "w", newline="") as fo:
    w = csv.writer(fo); w.writerow(["model", "designed", "prefix", "snr", "psnr"]); w.writerows(rows)
print("saved", a.out, len(rows), "rows")
