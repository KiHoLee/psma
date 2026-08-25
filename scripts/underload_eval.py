"""Underload evaluation on the N=4-designed models (no retraining).

Masking: only K active users are superposed; frame power normalisation reallocates the FULL
bandwidth and power to them (less interference, more power per user).
Static OMA: each user owns L_e/N dimensions by design; idle blocks stay silent, the active
user's per-symbol power is unchanged, and there is no interference either way -> per-user
quality is independent of K and equals the full-load (K=N) figure, which we measure directly.

    python scripts/underload_eval.py --out ~/ViT/logs/underload.json
"""
import argparse, os, sys, json, glob, random
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from swinsc import Config, Transmitter, Receiver, Channel
from deepsc_ri.metrics import psnr

p = argparse.ArgumentParser()
p.add_argument("--learned", default=os.path.expanduser("~/ViT/checkpoints/swinsc_ov_u4_learned"))
p.add_argument("--oma", default=os.path.expanduser("~/ViT/checkpoints/swinsc_ov_u4_oma"))
p.add_argument("--n", type=int, default=200)
p.add_argument("--snrs", type=float, nargs="+", default=[0, 10, 20])
p.add_argument("--crop", type=int, default=128)
p.add_argument("--out", default=os.path.expanduser("~/ViT/logs/underload.json"))
a = p.parse_args()
dev = "cuda"
random.seed(0); torch.manual_seed(0)
val = sorted(glob.glob(os.path.expanduser("~/ViT/data/imagenette160/val/*/*.png")))
random.shuffle(val)


def load(path):
    im = Image.open(path).convert("RGB"); w, h = im.size; s = a.crop
    l, t = (w - s) // 2, (h - s) // 2
    return torch.from_numpy(__import__("numpy").array(im.crop((l, t, l + s, t + s)))).permute(2, 0, 1).float() / 255


def load_pair(ck):
    cfg = Config.load(os.path.join(ck, "config.json"))
    tx = Transmitter(cfg).to(dev).eval(); tx.load_state_dict(torch.load(os.path.join(ck, "tx.pt"), map_location=dev))
    rx = Receiver(cfg).to(dev).eval(); rx.load_state_dict(torch.load(os.path.join(ck, "rx.pt"), map_location=dev))
    return cfg, tx, rx


out = {}
# ---------------- masking: K = 1..4 active users on the shared frame ----------------
cfg, tx, rx = load_pair(a.learned)
N = cfg.users
imgs = [torch.stack([load(val[(u * a.n + i) % len(val)]) for i in range(a.n)]) for u in range(N)]
for K in range(1, N + 1):
    active = list(range(K))
    res = {}
    for s in a.snrs:
        ch = Channel("rayleigh", s)
        ps = []
        with torch.no_grad():
            for b in range(0, a.n, 25):
                xs = [imgs[u][b:b + 25].to(dev) for u in active]
                f, hw = tx(xs, active=active)
                z = ch(f)
                for j, u in enumerate(active):
                    ps.append(psnr(rx(z, u, hw), xs[j]).cpu())
        res[str(int(s))] = round(torch.cat(ps).mean().item(), 2)
    out[f"masking_k{K}"] = {"active": K, "designed": N, "kind": "masking", "curves": res}
    print("masking", K, res, flush=True)

# ---------------- static OMA: per-user quality independent of load = full-load figure ----------------
cfg_o, tx_o, rx_o = load_pair(a.oma)
res = {}
for s in a.snrs:
    ch = Channel("rayleigh", s)
    ps = []
    with torch.no_grad():
        for b in range(0, a.n, 25):
            xs = [imgs[u][b:b + 25].to(dev) for u in range(N)]
            f, hw = tx_o(xs)
            z = ch(f)
            for u in range(N):
                ps.append(psnr(rx_o(z, u, hw), xs[u]).cpu())
    res[str(int(s))] = round(torch.cat(ps).mean().item(), 2)
for K in range(1, N + 1):
    out[f"oma_k{K}"] = {"active": K, "designed": N, "kind": "oma", "curves": res,
                        "note": "static allocation: L_e/N dims per user regardless of load"}
print("oma(static, flat)", res, flush=True)

json.dump(out, open(a.out, "w"), indent=2)
print("saved", a.out)
