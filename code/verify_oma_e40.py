"""Evaluate a GIVEN Static-OMA checkpoint at N = K (audit, 2026-09-09).

Written for the 40-epoch N=4 OMA head (checkpoints/swinsc_ov_u4_oma_e40) so
that its row is comparable with the `oma,4,4,<snr>` rows of data/ser_eval.csv
and with the 40-epoch PSMA rows of data/verify_psma_e40.csv. The evaluation
loop is that of ser_eval.py / verify_eval.py, line for line:

  * images: random.seed(MAIN["SEED"]) + sorted glob of
    data/imagenette160/val/*/*.png + shuffle; user u reads images
    u*n .. u*n+n-1 (n = MAIN["VAL_IMAGES"] = 200), MAIN["CROP"] center crop;
  * batches of 25 images, reps = MAIN["EVAL_REPS"] = 5 fading draws;
  * seed_fading(K, s, rep, b) before every channel call with K = N (the load
    of the `oma` chain), so the fading sequence is the one every N=K=4 row of
    ser_eval.csv was measured on;
  * tx(xs, active=range(N)), Channel("rayleigh", s), rx(z, u, hw, K=N);
  * PSNR per image (deepsc_ri.metrics.psnr, mean in dB), semantic error by the
    fixed ResNet-50 IMAGENET1K_V2 classifier restricted to the ten classes.

Outputs (ser_eval.csv format; the scheme label is --scheme, default oma_e40):
  --out           scheme, designed, active, snr, psnr, ser, n_images
                  one row per SNR at K = N, plus (unless --no_static) the
                  Prop. 1 replication rows `<scheme>_static` for K = 1..N,
                  copied from the N=K row exactly as ser_eval.py builds
                  oma_static from oma.
  --out_perimage  scheme, designed, active, snr, rep, user, image_index, psnr, err
                  (K = N rows only; the replication adds no new measurement).

    PYTHONPATH=~/ViT python scripts/verify_oma_e40.py --ckpt checkpoints/swinsc_ov_u4_oma_e40 --users 4 \
        --snrs -5 0 5 10 15 20 --out data/verify_oma_e40.csv --out_perimage data/verify_oma_e40_perimage.csv
Smoke test (must give 21.541 dB / SER 0.376 at 10 dB, the ser_eval.csv oma,4,4,10 row):
    PYTHONPATH=~/ViT python scripts/verify_oma_e40.py --ckpt checkpoints/swinsc_ov_u4_oma --users 4 \
        --snrs 10 --scheme oma --out data/verify_smoke_oma_u4_10db.csv --out_perimage data/verify_smoke_oma_u4_10db_perimage.csv
"""
import argparse, os, sys, glob, random, csv, time
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_main import MAIN, device as MAIN_DEVICE, wpath
from swinsc import Config, Transmitter, Receiver, Channel
from deepsc_ri.metrics import psnr

CLASSES = ["tench", "English_springer", "cassette_player", "chain_saw", "church",
           "French_horn", "garbage_truck", "gas_pump", "golf_ball", "parachute"]
IMAGENET_IDS = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True, help="OMA checkpoint dir (tx.pt, rx.pt, config.json)")
p.add_argument("--users", type=int, required=True, help="provisioned N of the checkpoint; evaluated at K = N")
p.add_argument("--scheme", default="oma_e40", help="scheme label written to the CSV")
p.add_argument("--no_static", action="store_true", help="do not add the replicated <scheme>_static rows for K = 1..N")
p.add_argument("--n", type=int, default=MAIN["VAL_IMAGES"])
p.add_argument("--reps", type=int, default=MAIN["EVAL_REPS"])
p.add_argument("--snrs", type=float, nargs="+", default=list(MAIN["SNR_GRID"]))
p.add_argument("--crop", type=int, default=MAIN["CROP"])
p.add_argument("--nmax", type=int, default=8)
p.add_argument("--out", default=wpath("data", "verify_oma_e40.csv"))
p.add_argument("--out_perimage", default=wpath("data", "verify_oma_e40_perimage.csv"))
a = p.parse_args()
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
    print("checkpoint %s: users=%s mask=%s" % (d, getattr(cfg, "users", "?"), getattr(cfg, "mask", "?")), flush=True)
    return cfg, tx, rx


def eval_chain(d, scheme, designed, actives):
    """Verbatim tensor path of ser_eval.eval_chain (the `oma` chain is this with actives = [N])."""
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


N = a.users
eval_chain(a.ckpt, a.scheme, N, [N])

if not a.no_static:
    # Static OMA (Prop. 1): replicated from the N=K row for K = 1..N, exactly as
    # ser_eval.py builds oma_static from oma.
    for r in [r for r in rows if r[0] == a.scheme and r[1] == N and r[2] == N]:
        for K in range(1, N + 1):
            rows.append([a.scheme + "_static", N, K] + r[3:])

with open(a.out, "w", newline="") as fo:
    w = csv.writer(fo)
    w.writerow(["scheme", "designed", "active", "snr", "psnr", "ser", "n_images"]); w.writerows(rows)
with open(a.out_perimage, "w", newline="") as fo:
    w = csv.writer(fo)
    w.writerow(["scheme", "designed", "active", "snr", "rep", "user", "image_index", "psnr", "err"]); w.writerows(per)
print("saved", a.out, len(rows), "rows;", a.out_perimage, len(per), "rows; %.0fs" % (time.time() - T0))
