"""Reproducibility spot check of ser_eval.py with per-image output (audit, 2026-09-08).

Re-runs the EXACT evaluation loop of ser_eval.py (same image order, same
25-image batches, same common-random-number seeding `seed_fading`, same
ResNet-50 semantic-error rule) for a given checkpoint evaluated as a PSMA
chain and, optionally, for the N=4 comparison chains of the main frame, and
writes two files:

  --out          summary CSV in ser_eval.csv format
                 (scheme, designed, active, snr, psnr, ser, n_images)
  --out_perimage per-image CSV
                 (scheme, designed, active, snr, rep, user, image_index, psnr, err)
                 so that paired statistics (same image, same fading draw) can be
                 computed offline.

Schemes (all follow ser_eval.py byte for byte in the tensor operations):
  psma            checkpoint --ckpt evaluated at K = 1..--users (--active)
  oma_static      swinsc_ov_u4_oma at N = K = 4, replicated to K = 1..3 with the
                  SAME per-image values, exactly as ser_eval.py does (Prop. 1);
                  seeds are those of load K = 4
  oma_static_k    the same static allocation evaluated PHYSICALLY at load K:
                  all four blocks are filled (users 0..3), the frame is
                  normalized over the full frame, only users 0..K-1 are decoded,
                  and the fading is drawn with the load-K seed, so it is PAIRED
                  with the psma / masking_var rows of the same K. Under scalar
                  block fading with ZF the other blocks do not enter a user's
                  received block, so this is the static allocation of Prop. 1
                  under the load-K channel draws (a Monte Carlo re-draw of
                  oma_static, not a different scheme).
  masking_var     swinsc_ov_u4var_learned at K = 1..4
  deepma_offload  swinsc_ov_u4_deepma at K = 1..4

    PYTHONPATH=~/ViT python scripts/verify_eval.py --ckpt checkpoints/swinsc_ov_u4_psma --users 4 \
        --snrs 10 --frame4 --out data/verify_psma_10db.csv --out_perimage data/verify_psma_10db_perimage.csv
"""
import argparse, os, sys, glob, random, csv, time
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_main import MAIN, device as MAIN_DEVICE, wpath
from swinsc import Config, Transmitter, Receiver, Channel
from swinsc.mask_mux import power_normalize
from deepsc_ri.metrics import psnr

CLASSES = ["tench", "English_springer", "cassette_player", "chain_saw", "church",
           "French_horn", "garbage_truck", "gas_pump", "golf_ball", "parachute"]
IMAGENET_IDS = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

p = argparse.ArgumentParser()
p.add_argument("--ckpt", action="append", default=[], help="PSMA checkpoint dir (repeatable)")
p.add_argument("--users", type=int, action="append", default=[], help="provisioned N of each --ckpt")
p.add_argument("--active", type=int, nargs="*", default=None, help="loads K to evaluate (default 1..N)")
p.add_argument("--frame4", action="store_true", help="also evaluate oma_static, oma_static_k, masking_var, deepma_offload of the N=4 frame")
p.add_argument("--n", type=int, default=MAIN["VAL_IMAGES"])
p.add_argument("--reps", type=int, default=MAIN["EVAL_REPS"])
p.add_argument("--snrs", type=float, nargs="+", default=[MAIN["SNR_OP"]])
p.add_argument("--crop", type=int, default=MAIN["CROP"])
p.add_argument("--nmax", type=int, default=8)
p.add_argument("--out", default=wpath("data", "verify_psma_10db.csv"))
p.add_argument("--out_perimage", default=wpath("data", "verify_psma_10db_perimage.csv"))
a = p.parse_args()
assert len(a.ckpt) == len(a.users), "--ckpt and --users must be given in pairs"
dev = MAIN_DEVICE()
random.seed(MAIN["SEED"]); torch.manual_seed(MAIN["SEED"])
T0 = time.time()


def batches():
    return [(rep, b) for rep in range(a.reps) for b in range(0, a.n, 25)]


def seed_fading(K, s, rep, b):
    """Identical to ser_eval.py."""
    torch.manual_seed(MAIN["SEED"] * 1_000_003 + K * 10_007 + (int(s) + 5) * 101 + rep * 13 + b // 25)


CK = wpath("checkpoints")
val = sorted(glob.glob(wpath("data", "imagenette160", "val", "*", "*.png")))
random.shuffle(val)


def label_of(path):
    return CLASSES.index(os.path.basename(os.path.dirname(path)))


def load(path):
    im = Image.open(path).convert("RGB"); w, h = im.size; s = a.crop
    l, t = (w - s) // 2, (h - s) // 2
    return torch.from_numpy(__import__("numpy").array(im.crop((l, t, l + s, t + s)))).permute(2, 0, 1).float() / 255


from torchvision.models import resnet50, ResNet50_Weights
clf = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(dev).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


@torch.no_grad()
def predict(x):
    x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
    logits = clf((x - MEAN) / STD)[:, IMAGENET_IDS]
    return logits.argmax(1)


paths = [[val[(u * a.n + i) % len(val)] for i in range(a.n)] for u in range(a.nmax)]
imgs = [torch.stack([load(q) for q in paths[u]]) for u in range(a.nmax)]
labels = [torch.tensor([label_of(q) for q in paths[u]]) for u in range(a.nmax)]
print("images loaded: %d users x %d, first path of user 0: %s" % (a.nmax, a.n, paths[0][0]), flush=True)

rows, per = [], []


def record(scheme, designed, K, s, ps, errs):
    rows.append([scheme, designed, K, int(s), round(torch.cat(ps).mean().item(), 3),
                 round(torch.cat(errs).float().mean().item(), 4), len(torch.cat(errs))])
    print("%-14s N=%d K=%d snr=%g psnr=%.3f ser=%.4f n=%d  [%.0fs]" % (scheme, designed, K, s, rows[-1][4], rows[-1][5], rows[-1][6], time.time() - T0), flush=True)


def add_per(scheme, designed, K, s, rep, u, b, ps, er):
    for i in range(ps.shape[0]):
        per.append([scheme, designed, K, int(s), rep, u, b + i, "%.4f" % float(ps[i]), int(er[i])])


def pair(d):
    if not os.path.isabs(d) and not os.path.exists(os.path.join(d, "tx.pt")):
        d = os.path.join(CK, d)
    cfg = Config.load(os.path.join(d, "config.json"))
    tx = Transmitter(cfg).to(dev).eval(); tx.load_state_dict(torch.load(os.path.join(d, "tx.pt"), map_location=dev))
    rx = Receiver(cfg).to(dev).eval(); rx.load_state_dict(torch.load(os.path.join(d, "rx.pt"), map_location=dev))
    return cfg, tx, rx


def eval_chain(d, scheme, designed, actives):
    """Verbatim tensor path of ser_eval.eval_chain, plus per-image logging."""
    cfg, tx, rx = pair(d)
    for K in actives:
        act = list(range(K))
        for s in a.snrs:
            ch = Channel("rayleigh", s)
            ps, errs = [], []
            with torch.no_grad():
                for rep, b in batches():
                    xs = [imgs[u][b:b + 25].to(dev) for u in act]
                    seed_fading(K, s, rep, b)
                    f, hw = tx(xs, active=act)
                    z = ch(f)
                    for j, u in enumerate(act):
                        out = rx(z, u, hw, K=len(act))
                        pj = psnr(out, xs[j]).cpu(); ej = (predict(out).cpu() != labels[u][b:b + 25])
                        ps.append(pj); errs.append(ej); add_per(scheme, designed, K, s, rep, u, b, pj, ej)
            record(scheme, designed, K, s, ps, errs)


def eval_oma_static_k(d, N, actives):
    """Static OMA of Prop. 1 evaluated at load K under the load-K seeds: every
    block is filled (users 0..N-1 transmit as at full load), users 0..K-1 decoded."""
    cfg, tx, rx = pair(d)
    full = list(range(N))
    for K in actives:
        act = list(range(K))
        for s in a.snrs:
            ch = Channel("rayleigh", s)
            ps, errs = [], []
            with torch.no_grad():
                for rep, b in batches():
                    xs = [imgs[u][b:b + 25].to(dev) for u in full]
                    seed_fading(K, s, rep, b)
                    f, hw = tx(xs, active=full)
                    z = ch(f)
                    for j, u in enumerate(act):
                        out = rx(z, u, hw, K=N)
                        pj = psnr(out, xs[j]).cpu(); ej = (predict(out).cpu() != labels[u][b:b + 25])
                        ps.append(pj); errs.append(ej); add_per("oma_static_k", N, K, s, rep, u, b, pj, ej)
            record("oma_static_k", N, K, s, ps, errs)


for d, N in zip(a.ckpt, a.users):
    actives = a.active if a.active else list(range(1, N + 1))
    actives = [K for K in actives if K <= N]
    eval_chain(d, "psma", N, actives)

if a.frame4:
    N0 = MAIN["N"][0]
    act4 = [K for K in (a.active if a.active else range(1, N0 + 1)) if K <= N0]
    # oma N=K=4 measured once; replicated as oma_static K=1..4 (ser_eval.py rule)
    n_before = len(per)
    eval_chain("swinsc_ov_u%d_oma" % N0, "oma", N0, [N0])
    oma_rows = [r for r in rows if r[0] == "oma"]
    oma_per = per[n_before:]
    for K in act4:
        for r in oma_rows:
            rows.append(["oma_static", N0, K] + r[3:])
        for q in oma_per:
            if q[5] < K:                                   # only users present at load K
                per.append(["oma_static", N0, K] + q[3:])
    eval_oma_static_k("swinsc_ov_u%d_oma" % N0, N0, act4)
    eval_chain("swinsc_ov_u%dvar_learned" % N0, "masking_var", N0, act4)
    eval_chain("swinsc_ov_u%d_deepma" % N0, "deepma_offload", N0, act4)

with open(a.out, "w", newline="") as fo:
    w = csv.writer(fo)
    w.writerow(["scheme", "designed", "active", "snr", "psnr", "ser", "n_images"]); w.writerows(rows)
with open(a.out_perimage, "w", newline="") as fo:
    w = csv.writer(fo)
    w.writerow(["scheme", "designed", "active", "snr", "rep", "user", "image_index", "psnr", "err"]); w.writerows(per)
print("saved", a.out, len(rows), "rows;", a.out_perimage, len(per), "rows; %.0fs" % (time.time() - T0))
