"""Channel-estimation-error robustness of the N=4 frame at 10 dB (audit, 2026-09-15).

Same evaluation loop as ser_eval.py / verify_eval.py (same image order, 25-image
batches, common-random-number seeding `seed_fading`, ResNet-50 semantic-error
rule), but the zero-forcing step of swinsc/channel.py

    r = r / h                     # perfect-CSI ZF

is replaced, WITHOUT editing the library, by a Channel subclass that divides by
an imperfect estimate

    h_hat = h + sqrt(eps) * w,    w ~ CN(0, 1) per frame,

so eps is the normalised estimation MSE E|h_hat - h|^2 / E|h|^2 (E|h|^2 = 1).
The subclass draws the noise and h in exactly the order of Channel.forward and
draws w AFTERWARDS from the same seeded generator, so

  * eps = 0 is bit-identical to the perfect-CSI path (h + 0*w == h) and must
    reproduce the ser_eval.csv rows;
  * every eps and every scheme sees the same (h, noise, w) per frame, i.e. the
    comparison across eps and across schemes is paired.

Schemes (checkpoints of the N=4 main frame), loads K given by --active:
  psma            checkpoints/swinsc_ov_u4_psma
  oma             checkpoints/swinsc_ov_u4_oma at N = K = 4, replicated as
                  oma_static for every K in --active (ser_eval.py rule, Prop. 1)
  masking_var     checkpoints/swinsc_ov_u4var_learned
  deepma_offload  checkpoints/swinsc_ov_u4_deepma

Output (--out): scheme, designed, active, snr, eps, psnr, ser, n_images.

    PYTHONPATH=~/ViT python scripts/verify_csi.py --snrs 10 --active 1 4 --eps 0 0.01 0.03 0.1 \
        --out data/verify_csi.csv
"""
import argparse, os, sys, glob, random, csv, time, math
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
p.add_argument("--designed", type=int, default=MAIN["N"][0], help="provisioned N of the frame (checkpoint names swinsc_ov_u<N>_*)")
p.add_argument("--active", type=int, nargs="+", default=[1, 4], help="loads K to evaluate")
p.add_argument("--eps", type=float, nargs="+", default=[0.0, 0.01, 0.03, 0.1], help="normalised CSI estimation MSE values")
p.add_argument("--schemes", nargs="+", default=["psma", "oma", "masking_var", "deepma_offload"])
p.add_argument("--n", type=int, default=MAIN["VAL_IMAGES"])
p.add_argument("--reps", type=int, default=MAIN["EVAL_REPS"])
p.add_argument("--snrs", type=float, nargs="+", default=[MAIN["SNR_OP"]])
p.add_argument("--crop", type=int, default=MAIN["CROP"])
p.add_argument("--nmax", type=int, default=8)
p.add_argument("--out", default=wpath("data", "verify_csi.csv"))
p.add_argument("--out_perimage", default=None, help="optional per-image CSV (scheme, designed, active, snr, eps, rep, user, image_index, psnr, err)")
a = p.parse_args()
dev = MAIN_DEVICE()
random.seed(MAIN["SEED"]); torch.manual_seed(MAIN["SEED"])
T0 = time.time()


class CSIChannel(Channel):
    """Channel.forward for kind='rayleigh' with imperfect-CSI zero forcing.

    Identical draw order to Channel.forward (noise, then h), then w is drawn
    from the same generator; r is divided by h_hat = h + sqrt(eps) * w.
    """

    def __init__(self, snr_db, eps):
        super().__init__("rayleigh", snr_db)
        self.eps = float(eps)

    def forward(self, t_x, snr_db=None):
        snr_db = self.snr_db if snr_db is None else snr_db
        shape = t_x.shape
        z = self._complex(t_x.float())
        sigma = math.sqrt(1.0 / (10 ** (snr_db / 10)))          # per real-dimension noise std
        noise = (torch.randn_like(z.real) + 1j * torch.randn_like(z.real)) * sigma
        h = self.fading(shape[0], t_x.device).view(-1, 1, 1)
        r = h * z + noise
        w = ((torch.randn(shape[0], device=t_x.device) + 1j * torch.randn(shape[0], device=t_x.device))
             / math.sqrt(2)).to(torch.complex64).view(-1, 1, 1)  # CN(0,1) per frame
        h_hat = h + math.sqrt(self.eps) * w
        r = r / h_hat                                           # ZF with the estimate
        return self._real(r, shape).to(t_x.dtype)


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


def record(scheme, designed, K, s, eps, ps, errs):
    rows.append([scheme, designed, K, int(s), eps, round(torch.cat(ps).mean().item(), 3),
                 round(torch.cat(errs).float().mean().item(), 4), len(torch.cat(errs))])
    print("%-14s N=%d K=%d snr=%g eps=%g psnr=%.3f ser=%.4f n=%d  [%.0fs]" % (
        scheme, designed, K, s, eps, rows[-1][5], rows[-1][6], rows[-1][7], time.time() - T0), flush=True)


def add_per(scheme, designed, K, s, eps, rep, u, b, ps, er):
    if a.out_perimage is None:
        return
    for i in range(ps.shape[0]):
        per.append([scheme, designed, K, int(s), eps, rep, u, b + i, "%.4f" % float(ps[i]), int(er[i])])


def pair(d):
    if not os.path.isabs(d) and not os.path.exists(os.path.join(d, "tx.pt")):
        d = os.path.join(CK, d)
    cfg = Config.load(os.path.join(d, "config.json"))
    tx = Transmitter(cfg).to(dev).eval(); tx.load_state_dict(torch.load(os.path.join(d, "tx.pt"), map_location=dev))
    rx = Receiver(cfg).to(dev).eval(); rx.load_state_dict(torch.load(os.path.join(d, "rx.pt"), map_location=dev))
    print("checkpoint %s: users=%s mask=%s" % (d, getattr(cfg, "users", "?"), getattr(cfg, "mask", "?")), flush=True)
    return cfg, tx, rx


def eval_chain(d, scheme, designed, actives):
    """Tensor path of ser_eval.eval_chain with CSIChannel in place of Channel."""
    cfg, tx, rx = pair(d)
    for K in actives:
        act = list(range(K))
        for s in a.snrs:
            for eps in a.eps:
                ch = CSIChannel(s, eps)
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
                            ps.append(pj); errs.append(ej); add_per(scheme, designed, K, s, eps, rep, u, b, pj, ej)
                record(scheme, designed, K, s, eps, ps, errs)


N0 = a.designed
CKPT = {"psma": "swinsc_ov_u%d_psma" % N0, "oma": "swinsc_ov_u%d_oma" % N0,
        "masking_var": "swinsc_ov_u%dvar_learned" % N0, "deepma_offload": "swinsc_ov_u%d_deepma" % N0}
act = [K for K in a.active if K <= N0]

for scheme in a.schemes:
    if scheme == "oma":
        # oma measured once at N = K; replicated as oma_static for every K (ser_eval.py rule)
        eval_chain(CKPT["oma"], "oma", N0, [N0])
        for r in [r for r in rows if r[0] == "oma"]:
            for K in act:
                rows.append(["oma_static", N0, K] + r[3:])
    else:
        eval_chain(CKPT[scheme], scheme, N0, act)

with open(a.out, "w", newline="") as fo:
    w = csv.writer(fo)
    w.writerow(["scheme", "designed", "active", "snr", "eps", "psnr", "ser", "n_images"]); w.writerows(rows)
print("saved", a.out, len(rows), "rows; %.0fs" % (time.time() - T0))
if a.out_perimage:
    with open(a.out_perimage, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["scheme", "designed", "active", "snr", "eps", "rep", "user", "image_index", "psnr", "err"]); w.writerows(per)
    print("saved", a.out_perimage, len(per), "rows")
