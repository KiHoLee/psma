"""Noiseless PSNR of the token-signature (ToDMA-style) VQ autoencoder on the
evaluation population (audit, 2026-09-08; writing standard 12.16).

The manuscript quotes the "noiseless PSNR of 25.9 dB" of the quantized
autoencoder. That figure was read off the last line of logs/todma_train.log,
where todma_train.py evaluates the FIRST FOUR 64-image batches of the
torchvision ImageFolder validation loader (256 images, folder order) after the
final epoch. This script recomputes the quantity from the stored checkpoint
`checkpoints/todma_v256` on the SAME 200 validation images that ser_eval.py
assigns to user 0 (random.seed(SEED) + sorted glob + shuffle, 128x128 center
crop), through encode -> nearest codeword -> decode with NO channel, i.e. the
`dec(Cb[idx], hw)` path of the todma section of ser_eval.py with the true
index in place of the detected one. Per-image PSNRs are stored so that the
mean (in the dB domain, as every PSNR in the paper) has a raw artifact.

Outputs
  --out          per-image CSV: image_index, path, psnr           (user 0, 200 rows)
  --out_summary  summary CSV: protocol, n_images, mean_psnr_db, std_psnr_db, se_psnr_db
                 protocols: user0_200        the 200 images of user 0 (the quoted quantity)
                            all_users_1600   users 0..7 of ser_eval.py (8 x 200)
                            trainlog_4x64    the todma_train.py protocol (4 x 64 folder-order images),
                                             reproduces the training-log line
                            trainlog_4x64_st the same with the straight-through tensor s + (q - s)
                                             that todma_train.py decodes (numerically ~ q)

    cd ~/ViT; PYTHONPATH=~/ViT python scripts/verify_todma_clean.py
"""
import argparse, os, sys, glob, random, csv, time
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_main import MAIN, device as MAIN_DEVICE, wpath
from swinsc import Config
from swinsc.swin import SwinEncoder, SwinDecoder
from deepsc_ri.metrics import psnr

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="todma_v256", help="checkpoint dir (name under checkpoints/ or a path)")
p.add_argument("--n", type=int, default=MAIN["VAL_IMAGES"])
p.add_argument("--nmax", type=int, default=8)
p.add_argument("--crop", type=int, default=MAIN["CROP"])
p.add_argument("--out", default=wpath("data", "verify_todma_clean.csv"))
p.add_argument("--out_summary", default=wpath("data", "verify_todma_clean_summary.csv"))
a = p.parse_args()
dev = MAIN_DEVICE()
random.seed(MAIN["SEED"]); torch.manual_seed(MAIN["SEED"])
T0 = time.time()

# ---- data: identical to ser_eval.py --------------------------------------------------
CK = wpath("checkpoints")
val = sorted(glob.glob(wpath("data", "imagenette160", "val", "*", "*.png")))
random.shuffle(val)


def load(path):
    im = Image.open(path).convert("RGB"); w, h = im.size; s = a.crop
    l, t = (w - s) // 2, (h - s) // 2
    return torch.from_numpy(__import__("numpy").array(im.crop((l, t, l + s, t + s)))).permute(2, 0, 1).float() / 255


paths = [[val[(u * a.n + i) % len(val)] for i in range(a.n)] for u in range(a.nmax)]
imgs = [torch.stack([load(q) for q in paths[u]]) for u in range(a.nmax)]
print("images loaded: %d users x %d, first path of user 0: %s" % (a.nmax, a.n, paths[0][0]), flush=True)

# ---- model: identical to the todma section of ser_eval.py ------------------------------
d = a.ckpt if os.path.exists(os.path.join(a.ckpt, "model.pt")) else os.path.join(CK, a.ckpt)
st = torch.load(os.path.join(d, "model.pt"), map_location=dev)
cfgT = Config.load(os.path.join(d, "config.json"))
V, D = st["v"], st["l_s"]
enc = SwinEncoder(cfgT.in_ch, cfgT.patch, cfgT.dims, cfgT.depths, cfgT.heads, cfgT.window, D).to(dev).eval()
dec = SwinDecoder(cfgT.in_ch, cfgT.patch, tuple(reversed(cfgT.dims)), tuple(reversed(cfgT.depths)),
                  tuple(reversed(cfgT.heads)), cfgT.window, D).to(dev).eval()
enc.load_state_dict(st["enc"]); dec.load_state_dict(st["dec"])
Cb = st["vq"]["codebook"].to(dev)
print("checkpoint %s: V=%d D=%d" % (d, V, D), flush=True)


@torch.no_grad()
def clean_pass(x, straight_through=False):
    """encode -> nearest codeword -> decode, no channel. Returns per-image PSNR and indices."""
    ssem, hw = enc(x)
    d2 = ssem.pow(2).sum(-1, keepdim=True) - 2 * ssem @ Cb.t() + Cb.pow(2).sum(-1)
    qi = d2.argmin(-1)
    q = Cb[qi]
    if straight_through:                       # todma_train.py decodes s + (q - s).detach()
        q = ssem + (q - ssem)
    return psnr(dec(q, hw), x).cpu(), qi.cpu()


def summarize(name, ps):
    ps = torch.cat(ps).double()
    n = ps.numel(); m = ps.mean().item(); sd = ps.std(unbiased=True).item() if n > 1 else 0.0
    print("%-18s n=%4d mean=%.4f dB std=%.4f se=%.4f  [%.0fs]" % (name, n, m, sd, sd / n ** 0.5, time.time() - T0), flush=True)
    return [name, n, "%.4f" % m, "%.4f" % sd, "%.4f" % (sd / n ** 0.5)]


rows, summary, used = [], [], set()
# (1) user 0's 200 images, 25-image batches as in ser_eval.py
ps0 = []
for b in range(0, a.n, 25):
    pj, qi = clean_pass(imgs[0][b:b + 25].to(dev)); ps0.append(pj); used |= set(qi.flatten().tolist())
    for i in range(pj.shape[0]):
        rows.append([b + i, os.path.relpath(paths[0][b + i], wpath("data")), "%.4f" % float(pj[i])])
summary.append(summarize("user0_200", ps0))
print("codewords used by user 0's images: %d/%d" % (len(used), V), flush=True)
# (2) all eight users of ser_eval.py (1600 images)
psa = list(ps0)
for u in range(1, a.nmax):
    for b in range(0, a.n, 25):
        psa.append(clean_pass(imgs[u][b:b + 25].to(dev))[0])
summary.append(summarize("all_users_%d" % (a.nmax * a.n), psa))
# (3) the todma_train.py protocol: first four 64-image batches of the ImageFolder val loader
try:
    from swinsc.data import get_loader
    te = get_loader("imagenette", False, 64, cfgT.img_size, workers=0)
    pl, pl_st = [], []
    for i, (x, _) in enumerate(te):
        if i >= 4: break
        x = x.to(dev)
        pl.append(clean_pass(x)[0]); pl_st.append(clean_pass(x, straight_through=True)[0])
    summary.append(summarize("trainlog_4x64", pl))
    summary.append(summarize("trainlog_4x64_st", pl_st))
except Exception as e:                                       # the loader is a cross-check only
    print("trainlog protocol skipped:", repr(e), flush=True)

with open(a.out, "w", newline="") as fo:
    w = csv.writer(fo); w.writerow(["image_index", "path", "psnr"]); w.writerows(rows)
with open(a.out_summary, "w", newline="") as fo:
    w = csv.writer(fo); w.writerow(["protocol", "n_images", "mean_psnr_db", "std_psnr_db", "se_psnr_db"]); w.writerows(summary)
print("saved", a.out, len(rows), "rows;", a.out_summary, len(summary), "rows; %.0fs" % (time.time() - T0))
