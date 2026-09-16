"""Prefix sweep D(b) of ONE given PSMA checkpoint (audit, 2026-09-08).

A copy of scripts/prefix_eval.py whose hard-coded MODELS list is replaced by
the command-line pair --ckpt / --users (and a --model label); the tensor path,
the image set (user 0's 200 validation images of ser_eval.py), the batching
and the common-random-number seeding `seed_fading(b, s, rep, i)` are those of
prefix_eval.py line for line, so the rows are directly comparable with the
`psma` rows of data/prefix_eval.csv. prefix_eval.py itself is left untouched.

Rows: model, designed, prefix, snr, psnr
    cd ~/ViT; PYTHONPATH=~/ViT python scripts/verify_prefix_e40.py \
        --ckpt checkpoints/swinsc_ov_u4_psma_e40 --users 4 --snrs 10 --out data/verify_prefix_e40.csv
"""
import argparse, os, sys, glob, random, csv, time
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_main import MAIN, device as MAIN_DEVICE, wpath
from swinsc import Config, Transmitter, Receiver, Channel
from swinsc.mask_mux import power_normalize
from deepsc_ri.metrics import psnr

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True, help="PSMA checkpoint dir (path, or name under checkpoints/)")
p.add_argument("--users", type=int, required=True, help="provisioned N of the checkpoint (the `designed` column)")
p.add_argument("--model", default="psma", help="label written in the `model` column")
p.add_argument("--n", type=int, default=MAIN["VAL_IMAGES"])
p.add_argument("--reps", type=int, default=MAIN["EVAL_REPS"],
               help="independent fading draws per image (common across prefix lengths and models)")
p.add_argument("--snrs", type=float, nargs="+", default=list(MAIN["SNR_GRID"]))
p.add_argument("--crop", type=int, default=MAIN["CROP"])
p.add_argument("--out", default=wpath("data", "verify_prefix_e40.csv"))
a = p.parse_args()
dev = MAIN_DEVICE()
random.seed(MAIN["SEED"]); torch.manual_seed(MAIN["SEED"])
T0 = time.time()


def batches():
    return [(rep, i) for rep in range(a.reps) for i in range(0, a.n, 25)]


def seed_fading(K, s, rep, b):
    """Same common-random-number rule as ser_eval.py / prefix_eval.py."""
    torch.manual_seed(MAIN["SEED"] * 1_000_003 + K * 10_007 + (int(s) + 5) * 101 + rep * 13 + b // 25)
CK = wpath("checkpoints")
val = sorted(glob.glob(wpath("data", "imagenette160", "val", "*", "*.png")))
random.shuffle(val)


def load(path):
    im = Image.open(path).convert("RGB"); w, h = im.size; s = a.crop
    l, t = (w - s) // 2, (h - s) // 2
    return torch.from_numpy(__import__("numpy").array(im.crop((l, t, l + s, t + s)))).permute(2, 0, 1).float() / 255


imgs = torch.stack([load(val[i]) for i in range(a.n)])      # user 0's images, as in ser_eval.py
print("images loaded: %d, first path: %s" % (a.n, val[0]), flush=True)
rows = []
d = a.ckpt if os.path.exists(os.path.join(a.ckpt, "tx.pt")) else os.path.join(CK, a.ckpt)
if not os.path.exists(os.path.join(d, "tx.pt")):
    raise FileNotFoundError(d)
model, N = a.model, a.users
cfg = Config.load(os.path.join(d, "config.json"))
tx = Transmitter(cfg).to(dev).eval(); tx.load_state_dict(torch.load(os.path.join(d, "tx.pt"), map_location=dev))
rx = Receiver(cfg).to(dev).eval(); rx.load_state_dict(torch.load(os.path.join(d, "rx.pt"), map_location=dev))
L = cfg.l_e; Q = tx.spreader.Q
print("checkpoint %s: L=%d, %d decoders" % (d, L, len(rx.decoders)), flush=True)
for b in range(1, L + 1):
    for s in a.snrs:
        ch = Channel("rayleigh", s); ps = []
        with torch.no_grad():
            for rep, i in batches():
                x = imgs[i:i + 25].to(dev)
                seed_fading(b, s, rep, i)
                e, hw = tx.encode_user(x, 0)                       # (B, T, L) ordered symbols
                # prefix b on the first b rows at UNIT power per symbol (prefix_eval.py)
                f = power_normalize(e[..., :b] @ Q[:b]) * (b / L) ** 0.5
                z = ch(f)
                y = torch.nn.functional.pad(z @ Q[:b].T, (0, L - b))
                dec = rx.decoders[0]
                out = dec(rx.reduce[0](y), hw, **({"K": b} if dec.cond is not None else {}))
                ps.append(psnr(out, x).cpu())
        rows.append([model, N, b, int(s), round(torch.cat(ps).mean().item(), 3)])
        print("%s N=%d prefix=%d snr=%d psnr=%.3f  [%.0fs]" % (model, N, b, s, rows[-1][4], time.time() - T0), flush=True)

with open(a.out, "w", newline="") as fo:
    w = csv.writer(fo); w.writerow(["model", "designed", "prefix", "snr", "psnr"]); w.writerows(rows)
print("saved", a.out, len(rows), "rows; %.0fs" % (time.time() - T0))
