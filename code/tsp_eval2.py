"""Additional evaluation rows for Figs. 4-5 completion (paper 12).

Evaluates the N=2 variable-load model (L=8) and the retrained
Rayleigh dimension-rich set (L=32: fixed, variable, OMA, ToDMA)
with the same images, seeds, and channel conventions as tsp_eval.py.

Writes data/tsp_eval2_raw.csv: scheme,designed,active,snr,psnr

    python code/tsp_eval2.py --out data/tsp_eval2_raw.csv
"""
import argparse, os, sys, glob, random, csv
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_main import MAIN, device as MAIN_DEVICE, wpath   # the ONE configuration (standard 7.9)
from swinsc import Config, Transmitter, Receiver, Channel
from swinsc.swin import SwinEncoder, SwinDecoder
from deepsc_ri.metrics import psnr

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=200)
p.add_argument("--snrs", type=float, nargs="+", default=[-5, 0, 5, 10, 15, 20])
p.add_argument("--crop", type=int, default=128)
p.add_argument("--out", default=wpath("data", "tsp_eval2.csv"))
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
        for s in a.snrs:
            ch = Channel("rayleigh", s)
            ps = []
            with torch.no_grad():
                for b in range(0, a.n, 25):
                    xs = [imgs[u][b:b + 25].to(dev) for u in act]
                    f, hw = tx(xs, active=act)
                    z = ch(f)
                    for j, u in enumerate(act):
                        ps.append(psnr(rx(z, u, hw, K=len(act)), xs[j]).cpu())
            v = round(torch.cat(ps).mean().item(), 3)
            rows.append([scheme, designed, K, int(s), v])
            print(scheme, designed, K, s, v, flush=True)


eval_masked("swinsc_ov_u2var_learned", "masking_var", 2, [1, 2])
eval_masked("swinsc_rich_u2_learned", "masking_rich", 2, [2])
eval_masked("swinsc_rich_u2_oma", "oma_rich", 2, [2])
eval_masked("swinsc_rich_u2var_learned", "masking_var_rich", 2, [1, 2])


def eval_todma(ckpt, scheme, designed, actives):
    st = torch.load(os.path.join(CK, ckpt, "model.pt"), map_location=dev)
    cfgT = Config.load(os.path.join(CK, ckpt, "config.json"))
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

    for K in actives:
        act = list(range(K))
        for s in a.snrs:
            ch = Channel("rayleigh", s)
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
                        ps.append(psnr(dec(Cb[rec], hw), xs[j]).cpu())
            v = round(torch.cat(ps).mean().item(), 3)
            rows.append([scheme, designed, K, int(s), v])
            print(scheme, designed, K, s, v, flush=True)


eval_todma("todma_v256_ls32", "todma_rich", 2, [1, 2])

with open(a.out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scheme", "designed", "active", "snr", "psnr"])
    w.writerows(rows)
print("saved", a.out)
