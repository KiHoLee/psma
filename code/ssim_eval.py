"""SSIM companion metric for the TSP manuscript, 10 dB operating point only.

Writes data/ssim_eval.csv: scheme,designed,active,snr,ssim
Same models, images, seeds, and channel conventions as tsp_eval.py.

    python code/ssim_eval.py --out data/ssim_eval.csv
"""
import argparse, os, sys, glob, random, csv
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_main import MAIN, device as MAIN_DEVICE, wpath   # the ONE configuration (standard 7.9)
from swinsc import Config, Transmitter, Receiver, Channel
from swinsc.swin import SwinEncoder, SwinDecoder

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=200)
p.add_argument("--snr", type=float, default=10.0)
p.add_argument("--crop", type=int, default=128)
p.add_argument("--out", default=wpath("data", "ssim_eval.csv"))
a = p.parse_args()
dev = MAIN_DEVICE()
random.seed(MAIN["SEED"]); torch.manual_seed(MAIN["SEED"])
CK = wpath("checkpoints")
val = sorted(glob.glob(wpath("data", "imagenette160", "val", "*", "*.png")))
random.shuffle(val)


def load(path):
    im = Image.open(path).convert("RGB"); w, h = im.size; s = a.crop
    l, t = (w - s) // 2, (h - s) // 2
    return torch.from_numpy(__import__("numpy").array(im.crop((l, t, l + s, t + s)))).permute(2, 0, 1).float() / 255


# standard single-scale SSIM: 11x11 Gaussian window, sigma 1.5, C1/C2 for range 1
_g = torch.arange(11).float() - 5
_g = torch.exp(-_g ** 2 / (2 * 1.5 ** 2)); _g = _g / _g.sum()
KERN = (_g[:, None] @ _g[None, :]).expand(3, 1, 11, 11).contiguous().to(dev)
C1, C2 = 0.01 ** 2, 0.03 ** 2


def ssim(x, y):
    x = x.clamp(0, 1); y = y.clamp(0, 1)
    mx = F.conv2d(x, KERN, padding=5, groups=3)
    my = F.conv2d(y, KERN, padding=5, groups=3)
    mx2, my2, mxy = mx * mx, my * my, mx * my
    sx = F.conv2d(x * x, KERN, padding=5, groups=3) - mx2
    sy = F.conv2d(y * y, KERN, padding=5, groups=3) - my2
    sxy = F.conv2d(x * y, KERN, padding=5, groups=3) - mxy
    m = ((2 * mxy + C1) * (2 * sxy + C2)) / ((mx2 + my2 + C1) * (sx + sy + C2))
    return m.mean(dim=(1, 2, 3))


def pair(name):
    d = os.path.join(CK, name)
    cfg = Config.load(os.path.join(d, "config.json"))
    tx = Transmitter(cfg).to(dev).eval(); tx.load_state_dict(torch.load(os.path.join(d, "tx.pt"), map_location=dev))
    rx = Receiver(cfg).to(dev).eval(); rx.load_state_dict(torch.load(os.path.join(d, "rx.pt"), map_location=dev))
    return cfg, tx, rx


rows = []
NMAX = 4
imgs = [torch.stack([load(val[(u * a.n + i) % len(val)]) for i in range(a.n)]) for u in range(NMAX)]


def eval_masked(name, scheme, designed, actives):
    cfg, tx, rx = pair(name)
    for K in actives:
        act = list(range(K))
        ch = Channel("rayleigh", a.snr)
        ps = []
        with torch.no_grad():
            for b in range(0, a.n, 25):
                xs = [imgs[u][b:b + 25].to(dev) for u in act]
                f, hw = tx(xs, active=act)
                z = ch(f)
                for j, u in enumerate(act):
                    ps.append(ssim(rx(z, u, hw, K=len(act)), xs[j]).cpu())
        v = round(torch.cat(ps).mean().item(), 4)
        rows.append([scheme, designed, K, int(a.snr), v])
        print(scheme, designed, K, v, flush=True)


eval_masked("swinsc_ov_u4_learned", "masking_fixed", 4, [4])
eval_masked("swinsc_ov_u4_oma", "oma", 4, [4])
eval_masked("swinsc_ov_u4_deepma", "deepma", 4, [4])
eval_masked("swinsc_ov_u4var_learned", "masking_var", 4, [1, 2, 3, 4])
for r in [r for r in rows if r[0] == "oma"]:
    for K in (1, 2, 3):
        rows.append(["oma_static", 4, K, r[3], r[4]])
    rows.append(["oma_static", 4, 4, r[3], r[4]])

# ---------------- ToDMA ----------------
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
        G = A @ At
        coef = torch.linalg.solve(G + 1e-4 * torch.eye(k + 1, device=z.device), A @ z.unsqueeze(-1))
        resid = z - (At @ coef).squeeze(-1)
    return torch.stack(picked, 1)


for K in (1, 2, 3, 4):
    act = list(range(K))
    ch = Channel("rayleigh", a.snr)
    ps = []
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
            Bsz, Ntok, _ = z.shape
            det = omp(z.reshape(-1, D), K).reshape(Bsz, Ntok, K)
            for j, u in enumerate(act):
                true = qidx[j]
                hit = (det == true.unsqueeze(-1)).any(-1)
                rec = torch.where(hit, true, det[..., 0])
                ps.append(ssim(dec(Cb[rec], hw), xs[j]).cpu())
    v = round(torch.cat(ps).mean().item(), 4)
    rows.append(["todma", 4, K, int(a.snr), v])
    print("todma", 4, K, v, flush=True)

with open(a.out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scheme", "designed", "active", "snr", "ssim"])
    w.writerows(rows)
print("saved", a.out)
