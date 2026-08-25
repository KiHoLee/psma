"""ToDMA baseline, stage 2 (training-free multiple access): evaluate overload (all U active)
and underload (K of N active) on the shared frame.

TX: every active user quantises its tokens and transmits its codeword vector; the frame is the
power-normalised superposition on the SAME L_e dims (full band reuse, like masking).
Channel: Rayleigh flat block fading, perfect-CSI ZF.
TX signatures follow the SSE letter (paper 10): every vocabulary entry carries a FIXED random
Gaussian signature of length L_e (training-free dictionary, seed-shared); the semantic VQ codebook
is used only for reconstruction after the index is recovered.
RX: per token, OMP over the signature dictionary with K iterations detects the active entries;
genie-aided association (as in the SSE letter) assigns a detected codeword to a user whenever the
user's true codeword is in the detected support -> that user decodes the CLEAN codeword (digital
index recovered). Otherwise the user decodes the nearest detected codeword (detection error).

    python scripts/todma_eval.py --ckpt ~/ViT/checkpoints/todma_v256 --out ~/ViT/logs/todma.json
"""
import argparse, os, sys, json, glob, random
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from swinsc import Config, Channel
from swinsc.swin import SwinEncoder, SwinDecoder
from deepsc_ri.metrics import psnr

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default=os.path.expanduser("~/ViT/checkpoints/todma_v256"))
p.add_argument("--n", type=int, default=200)
p.add_argument("--snrs", type=float, nargs="+", default=[0, 10, 20])
p.add_argument("--users", type=int, nargs="+", default=[1, 2, 3, 4])
p.add_argument("--crop", type=int, default=128)
p.add_argument("--out", default=os.path.expanduser("~/ViT/logs/todma.json"))
a = p.parse_args()
dev = "cuda"
random.seed(0); torch.manual_seed(0)

st = torch.load(os.path.join(a.ckpt, "model.pt"), map_location=dev)
cfg = Config.load(os.path.join(a.ckpt, "config.json"))
V, D = st["v"], st["l_s"]
enc = SwinEncoder(cfg.in_ch, cfg.patch, cfg.dims, cfg.depths, cfg.heads, cfg.window, D).to(dev).eval()
dec = SwinDecoder(cfg.in_ch, cfg.patch, tuple(reversed(cfg.dims)), tuple(reversed(cfg.depths)),
                  tuple(reversed(cfg.heads)), cfg.window, D).to(dev).eval()
enc.load_state_dict(st["enc"]); dec.load_state_dict(st["dec"])
C = st["vq"]["codebook"].to(dev)                                   # (V, D) semantic codebook (decoder side)
g = torch.Generator(device="cpu").manual_seed(2026)
S = torch.randn(V, D, generator=g)                                 # fixed random Gaussian signatures (paper 10)
S = (S / S.norm(dim=1, keepdim=True) * D ** 0.5).to(dev)           # unit average energy per dimension

val = sorted(glob.glob(os.path.expanduser("~/ViT/data/imagenette160/val/*/*.png")))
random.shuffle(val)


def load(path):
    im = Image.open(path).convert("RGB"); w, h = im.size; s = a.crop
    l, t = (w - s) // 2, (h - s) // 2
    return torch.from_numpy(__import__("numpy").array(im.crop((l, t, l + s, t + s)))).permute(2, 0, 1).float() / 255


def omp(z, K):
    """z: (M, D) received tokens; K-sparse OMP over codebook C. Returns (M, K) detected indices."""
    resid = z.clone()
    picked = []
    A = torch.zeros(z.shape[0], 0, D, device=z.device)
    for k in range(K):
        corr = (resid @ S.t()).abs()                               # (M, V)
        if picked:
            corr.scatter_(1, torch.stack(picked, 1), -1e9)         # without replacement
        idx = corr.argmax(-1)                                      # (M,)
        picked.append(idx)
        A = torch.cat([A, S[idx].unsqueeze(1)], 1)                 # (M, k+1, D)
        # least-squares refit of coefficients, then residual
        At = A.transpose(1, 2)                                     # (M, D, k+1)
        G = A @ At                                                 # (M, k+1, k+1)
        coef = torch.linalg.solve(G + 1e-4 * torch.eye(k + 1, device=z.device), A @ z.unsqueeze(-1))
        resid = z - (At @ coef).squeeze(-1)
    return torch.stack(picked, 1)                                  # (M, K)


N = max(a.users)
imgs = [torch.stack([load(val[(u * a.n + i) % len(val)]) for i in range(a.n)]) for u in range(N)]
out = {}
for K in a.users:
    active = list(range(K))
    res = {}
    for s in a.snrs:
        ch = Channel("rayleigh", s)
        ps = []
        with torch.no_grad():
            for b in range(0, a.n, 25):
                xs = [imgs[u][b:b + 25].to(dev) for u in active]
                qidx, hw = [], None
                f = None
                for x in xs:
                    ssem, hw = enc(x)
                    d2 = ssem.pow(2).sum(-1, keepdim=True) - 2 * ssem @ C.t() + C.pow(2).sum(-1)
                    qi = d2.argmin(-1)                             # (B, Ntok)
                    qidx.append(qi)
                    cw = S[qi]                                     # transmit SIGNATURE, not the codeword
                    f = cw if f is None else f + cw
                f = f / f.pow(2).mean().sqrt().clamp_min(1e-8)     # unit-power frame
                z = ch(f)                                          # rayleigh + ZF
                B, Ntok, _ = z.shape
                det = omp(z.reshape(-1, D), K).reshape(B, Ntok, K)  # detected supports
                for j, u in enumerate(active):
                    true = qidx[j]                                 # (B, Ntok)
                    hit = (det == true.unsqueeze(-1)).any(-1)      # genie association
                    # miss -> decode the first detected atom (wrong codeword)
                    rec_idx = torch.where(hit, true, det[..., 0])
                    outp = dec(C[rec_idx], hw)
                    ps.append(psnr(outp, xs[j]).cpu())
        res[str(int(s))] = round(torch.cat(ps).mean().item(), 2)
    out[f"todma_k{K}"] = {"active": K, "kind": "todma", "v": V, "curves": res}
    print("todma", K, res, flush=True)

json.dump(out, open(a.out, "w"), indent=2)
print("saved", a.out)
