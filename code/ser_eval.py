"""Semantic error rate (SER) and per-user PSNR for every chain, load and SNR.

SER definition (manuscript Sec. VI-A): a delivered image is a semantic error
when a fixed, publicly available ImageNet classifier (ResNet-50,
IMAGENET1K_V2 weights, logits restricted to the ten Imagenette classes)
assigns it a class different from the image's ground-truth label. The
classifier is never trained or fine-tuned here; the clean-image error of the
same classifier on the same crops is reported as the floor
(scheme = "clean"). The semantic throughput of the manuscript is then
T(K) = sum over served users of (1 - SER), images correctly delivered per
frame, which a scheme that BLOCKS users beyond its provisioned population
cannot raise with K.

Rows: scheme, designed, active, snr, psnr, ser, n_images
  masking_var    N=8 model, K = 1..8            one transceiver for the whole range
  masking_var    N=4 model, K = 1..4
  masking_fixed  N=K in {2,4,6,8}               retrained per load
  oma            N=K in {2,4,8}                 re-encoded per load (B = L/N)
  oma_static     N=4 model, K = 1..4            static allocation; K > 4 is BLOCKED (no row)
  deepma         N=K in {2,4,6,8}, and the N=4 model at K = 1..4 (scheme deepma_static)
  todma          K = 1..8                       token signatures (genie)
  clean          classifier on the source crops

    python code/ser_eval.py --out data/ser_eval.csv
"""
import argparse, os, sys, glob, random, csv
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_main import MAIN, device as MAIN_DEVICE, wpath   # the ONE configuration (standard 7.9)
from swinsc import Config, Transmitter, Receiver, Channel
from swinsc.swin import SwinEncoder, SwinDecoder
from deepsc_ri.metrics import psnr

CLASSES = ["tench", "English_springer", "cassette_player", "chain_saw", "church",
           "French_horn", "garbage_truck", "gas_pump", "golf_ball", "parachute"]
IMAGENET_IDS = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]   # ResNet-50 output indices of the ten classes

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=MAIN["VAL_IMAGES"])
p.add_argument("--snrs", type=float, nargs="+", default=list(MAIN["SNR_GRID"]))
p.add_argument("--crop", type=int, default=MAIN["CROP"])
p.add_argument("--nmax", type=int, default=8)
p.add_argument("--out", default=wpath("data", "ser_eval.csv"))
p.add_argument("--skip_missing", action="store_true", help="skip chains whose checkpoint is absent")
a = p.parse_args()
dev = MAIN_DEVICE()
random.seed(MAIN["SEED"]); torch.manual_seed(MAIN["SEED"])
CK = wpath("checkpoints")
val = sorted(glob.glob(wpath("data", "imagenette160", "val", "*", "*.png")))
random.shuffle(val)


def label_of(path):
    return CLASSES.index(os.path.basename(os.path.dirname(path)))


def load(path):
    im = Image.open(path).convert("RGB"); w, h = im.size; s = a.crop
    l, t = (w - s) // 2, (h - s) // 2
    return torch.from_numpy(__import__("numpy").array(im.crop((l, t, l + s, t + s)))).permute(2, 0, 1).float() / 255


# ---- classifier -------------------------------------------------------------
from torchvision.models import resnet50, ResNet50_Weights
clf = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(dev).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


@torch.no_grad()
def predict(x):
    """x in [0,1], (B,3,H,W) -> predicted Imagenette class index (0..9)."""
    x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
    logits = clf((x - MEAN) / STD)[:, IMAGENET_IDS]
    return logits.argmax(1)


# ---- data: user u reads images u*n .. u*n+n-1, exactly as tsp_eval*.py ------
paths = [[val[(u * a.n + i) % len(val)] for i in range(a.n)] for u in range(a.nmax)]
imgs = [torch.stack([load(q) for q in paths[u]]) for u in range(a.nmax)]
labels = [torch.tensor([label_of(q) for q in paths[u]]) for u in range(a.nmax)]

rows = []


def record(scheme, designed, K, s, ps, errs):
    rows.append([scheme, designed, K, int(s), round(torch.cat(ps).mean().item(), 3),
                 round(torch.cat(errs).float().mean().item(), 4), len(torch.cat(errs))])
    print(scheme, designed, K, s, rows[-1][4], rows[-1][5], flush=True)


def pair(name):
    d = os.path.join(CK, name)
    if not os.path.exists(os.path.join(d, "tx.pt")):
        if a.skip_missing:
            print("skip (no checkpoint):", name, flush=True); return None
        raise FileNotFoundError(d)
    cfg = Config.load(os.path.join(d, "config.json"))
    tx = Transmitter(cfg).to(dev).eval(); tx.load_state_dict(torch.load(os.path.join(d, "tx.pt"), map_location=dev))
    rx = Receiver(cfg).to(dev).eval(); rx.load_state_dict(torch.load(os.path.join(d, "rx.pt"), map_location=dev))
    return cfg, tx, rx


def eval_chain(name, scheme, designed, actives):
    pr = pair(name)
    if pr is None:
        return
    cfg, tx, rx = pr
    for K in actives:
        act = list(range(K))
        for s in a.snrs:
            ch = Channel("rayleigh", s)
            ps, errs = [], []
            with torch.no_grad():
                for b in range(0, a.n, 25):
                    xs = [imgs[u][b:b + 25].to(dev) for u in act]
                    f, hw = tx(xs, active=act)
                    z = ch(f)
                    for j, u in enumerate(act):
                        out = rx(z, u, hw, K=len(act))
                        ps.append(psnr(out, xs[j]).cpu())
                        errs.append((predict(out).cpu() != labels[u][b:b + 25]))
            record(scheme, designed, K, s, ps, errs)


# ---- clean floor -----------------------------------------------------------------
errs = []
with torch.no_grad():
    for u in range(a.nmax):
        for b in range(0, a.n, 25):
            errs.append((predict(imgs[u][b:b + 25].to(dev)).cpu() != labels[u][b:b + 25]))
rows.append(["clean", 0, 0, 0, 99.0, round(torch.cat(errs).float().mean().item(), 4), len(torch.cat(errs))])
print("clean classifier error on the source crops:", rows[-1][5], flush=True)

# ---- learned chains ---------------------------------------------------------------
N0 = MAIN["N"][-1]
eval_chain("swinsc_ov_u8var_learned", "masking_var", 8, list(range(1, 9)))
eval_chain("swinsc_ov_u%dvar_learned" % N0, "masking_var", N0, list(range(1, N0 + 1)))
for K in (2, 4, 6, 8):
    eval_chain("swinsc_ov_u%d_learned" % K, "masking_fixed", K, [K])
for K in (1, 2, 4, 8):
    eval_chain("swinsc_ov_u%d_oma" % K, "oma", K, [K])
# Static OMA keeps gamma^o at its full-load value and leaves idle blocks silent,
# so an active user's block is unchanged by K and its quality equals the
# full-load measurement (Proposition 1). It is therefore REPLICATED from the
# N=K row, exactly as tsp_eval.py does. Running the transmitter at K<N would
# instead renormalize the frame over the surviving blocks and hand them extra
# power, which is not the static allocation.
for r in [r for r in rows if r[0] == "oma" and r[1] == N0 and r[2] == N0]:
    for K in range(1, N0):
        rows.append(["oma_static", N0, K] + r[3:])
    rows.append(["oma_static", N0, N0] + r[3:])
for K in (2, 4, 6, 8):
    eval_chain("swinsc_ov_u%d_deepma" % K, "deepma", K, [K])
# DeepMA below its training load: its frame is renormalized over the active
# set like the masked chain's, so this is a physical evaluation of the N=4
# pairs at K<4 (the DeepMA counterpart of "fixed load, off-load").
eval_chain("swinsc_ov_u%d_deepma" % N0, "deepma_offload", N0, list(range(1, N0 + 1)))

# ---- token signatures (genie), same code path as tsp_eval.py ------------------------
st = torch.load(os.path.join(CK, "todma_v256", "model.pt"), map_location=dev)
cfgT = Config.load(os.path.join(CK, "todma_v256", "config.json"))
V, D = st["v"], st["l_s"]
enc = SwinEncoder(cfgT.in_ch, cfgT.patch, cfgT.dims, cfgT.depths, cfgT.heads, cfgT.window, D).to(dev).eval()
dec = SwinDecoder(cfgT.in_ch, cfgT.patch, tuple(reversed(cfgT.dims)), tuple(reversed(cfgT.depths)),
                  tuple(reversed(cfgT.heads)), cfgT.window, D).to(dev).eval()
enc.load_state_dict(st["enc"]); dec.load_state_dict(st["dec"])
Cb = st["vq"]["codebook"].to(dev)
g = torch.Generator(device="cpu").manual_seed(MAIN["SIGNATURE_SEED"])
S = torch.randn(V, D, generator=g)
S = (S / S.norm(dim=1, keepdim=True) * D ** 0.5).to(dev)


def omp(z, K):
    resid = z.clone(); picked = []
    A = torch.zeros(z.shape[0], 0, D, device=z.device)
    for k in range(K):
        corr = (resid @ S.t()).abs()
        if picked:
            corr.scatter_(1, torch.stack(picked, 1), -1e9)
        idx = corr.argmax(-1); picked.append(idx)
        A = torch.cat([A, S[idx].unsqueeze(1)], 1)
        At = A.transpose(1, 2)
        coef = torch.linalg.solve(A @ At + 1e-4 * torch.eye(k + 1, device=z.device), A @ z.unsqueeze(-1))
        resid = z - (At @ coef).squeeze(-1)
    return torch.stack(picked, 1)


for K in range(1, a.nmax + 1):
    act = list(range(K))
    for s in a.snrs:
        ch = Channel("rayleigh", s)
        ps, errs = [], []
        with torch.no_grad():
            for b in range(0, a.n, 25):
                xs = [imgs[u][b:b + 25].to(dev) for u in act]
                qidx, f, hw = [], None, None
                for x in xs:
                    ssem, hw = enc(x)
                    d2 = ssem.pow(2).sum(-1, keepdim=True) - 2 * ssem @ Cb.t() + Cb.pow(2).sum(-1)
                    qi = d2.argmin(-1); qidx.append(qi)
                    cw = S[qi]
                    f = cw if f is None else f + cw
                f = f / f.pow(2).mean().sqrt().clamp_min(1e-8)
                z = ch(f)
                B, Ntok, _ = z.shape
                det = omp(z.reshape(-1, D), K).reshape(B, Ntok, K)
                for j, u in enumerate(act):
                    true = qidx[j]
                    hit = (det == true.unsqueeze(-1)).any(-1)
                    rec = torch.where(hit, true, det[..., 0])
                    out = dec(Cb[rec], hw)
                    ps.append(psnr(out, xs[j]).cpu())
                    errs.append((predict(out).cpu() != labels[u][b:b + 25]))
        record("todma", N0, K, s, ps, errs)

with open(a.out, "w", newline="") as fo:
    w = csv.writer(fo)
    w.writerow(["scheme", "designed", "active", "snr", "psnr", "ser", "n_images"]); w.writerows(rows)
print("saved", a.out, len(rows), "rows")
