"""Deep-overload sweep at full load, N = K in {6, 8} (paper 12, Fig. 8).

Adds to the canonical grid the points the existing runs do not cover:
  masking_fixed  designed=K, active=K   for K in (6, 8)   (models trained at that load)
  oma            designed=8, active=8                     (re-encoded OMA, B = L/N = 1)
  todma          designed=4, active=K   for K in (6, 8)   (no provisioned population;
                                                           designed=4 is the placeholder
                                                           the K<=4 rows already use)
Re-encoded OMA has no N = 6 point because B = L/N must be an integer on the
L = 8 frame, so the OMA curve of Fig. 8 runs over N in {2, 4, 8} only.

Same images, seeds, SNR grid and channel conventions as tsp_eval.py: users
u = 0..7 read validation images u*n .. u*n+n-1, so users 0..3 see exactly
the images of the earlier runs. Writes the raw rows to --out and MERGES them
into data/tsp_eval.csv (replacing any row with the same scheme, designed,
active, snr), so plot_results.py keeps reading one file.

    python code/tsp_eval3.py --out data/tsp_eval3_raw.csv
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
p.add_argument("--n", type=int, default=MAIN["VAL_IMAGES"])
p.add_argument("--snrs", type=float, nargs="+", default=list(MAIN["SNR_GRID"]))
p.add_argument("--crop", type=int, default=MAIN["CROP"])
p.add_argument("--loads", type=int, nargs="+", default=[6, 8])
p.add_argument("--offload", action="store_true",
               help="also evaluate the N=4 fixed-load model BELOW its training load "
                    "(rows masking_fixed4off, designed=4, active=1..4; the Fig. 7 off-load curve)")
p.add_argument("--no_sweep", action="store_true", help="skip the N=K sweep and the token-signature rows")
p.add_argument("--out", default=wpath("data", "tsp_eval3_raw.csv"))
p.add_argument("--merge", default=wpath("data", "tsp_eval.csv"))
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
if a.no_sweep:
    a.loads = []
NMAX = max(a.loads + [MAIN["N"][-1]])
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


for K in a.loads:
    if os.path.isdir(os.path.join(CK, "swinsc_ov_u%d_learned" % K)):
        eval_masked("swinsc_ov_u%d_learned" % K, "masking_fixed", K, [K])
    else:
        print("no masked model at N=%d; skipping" % K, flush=True)
    oma_dir = os.path.join(CK, "swinsc_ov_u%d_oma" % K)
    if os.path.isdir(oma_dir):
        eval_masked("swinsc_ov_u%d_oma" % K, "oma", K, [K])
    else:
        print("no OMA model at N=%d (B = L/N is not an integer); skipping" % K, flush=True)

# ---------------- token signatures (genie), same code path as tsp_eval.py ----------------
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


for K in a.loads:
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
                B, Ntok, _ = z.shape
                det = omp(z.reshape(-1, D), K).reshape(B, Ntok, K)
                for j, u in enumerate(act):
                    true = qidx[j]
                    hit = (det == true.unsqueeze(-1)).any(-1)
                    rec = torch.where(hit, true, det[..., 0])
                    ps.append(psnr(dec(Cb[rec], hw), xs[j]).cpu())
        v = round(torch.cat(ps).mean().item(), 3)
        rows.append(["todma", 4, K, int(s), v])
        print("todma", 4, K, s, v, flush=True)

# ---------------- fixed-load model below its training load (Fig. 7 off-load curve) ----------------
if a.offload:
    N0 = MAIN["N"][0]
    eval_masked("swinsc_ov_u%d_learned" % N0, "masking_fixed4off", N0, list(range(1, N0 + 1)))

with open(a.out, "w", newline="") as fo:
    w = csv.writer(fo); w.writerow(["scheme", "designed", "active", "snr", "psnr"]); w.writerows(rows)
print("saved", a.out, len(rows), "rows")

# merge into the canonical grid, replacing rows with the same key
if a.merge and os.path.exists(a.merge):
    old = list(csv.DictReader(open(a.merge)))
    key = lambda r: (r["scheme"], str(r["designed"]), str(r["active"]), str(r["snr"]))
    new = {key(dict(zip(["scheme", "designed", "active", "snr", "psnr"], map(str, r)))): r for r in rows}
    kept = [r for r in old if key(r) not in new]
    with open(a.merge, "w", newline="") as fo:
        w = csv.writer(fo); w.writerow(["scheme", "designed", "active", "snr", "psnr"])
        w.writerows([[r["scheme"], r["designed"], r["active"], r["snr"], r["psnr"]] for r in kept])
        w.writerows(rows)
    print("merged into", a.merge, ": kept", len(kept), "+ new", len(rows))
