"""Qualitative reconstruction panels for the TSP manuscript.

Writes fig_visual_overload.pdf and fig_visual_underload.pdf plus a CSV of the
per-panel PSNRs. Same channel realization (fixed seed) across schemes so the
panels differ only by the access scheme.

    ~/tr_env/bin/python scripts/visual_eval.py --out ~/ViT/logs/visual
"""
import argparse, os, sys, glob, random, csv
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from swinsc import Config, Transmitter, Receiver, Channel
from swinsc.swin import SwinEncoder, SwinDecoder
from deepsc_ri.metrics import psnr

p = argparse.ArgumentParser()
p.add_argument("--crop", type=int, default=128)
p.add_argument("--snr", type=float, default=10.0)
p.add_argument("--img_index", type=int, default=7, help="validation image offset")
p.add_argument("--out", default=os.path.expanduser("~/ViT/logs/visual"))
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
CK = os.path.expanduser("~/ViT/checkpoints")
random.seed(0); torch.manual_seed(0)
val = sorted(glob.glob(os.path.expanduser("~/ViT/data/imagenette160/val/*/*.png")))
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


def channel_fixed(f, seed):
    """Rayleigh + ZF with a fixed realization so schemes share the channel."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    h = (torch.randn(1, generator=g) + 1j * torch.randn(1, generator=g)) / (2 ** 0.5)
    sigma = (10 ** (-a.snr / 10)) ** 0.5
    noise = torch.randn(f.shape, generator=g) * sigma
    return f + noise.to(f.device) / h.abs().item()


# the four users' images; user 0 is the displayed one
imgs = [load(val[u * 200 + a.img_index]).unsqueeze(0).to(dev) for u in range(4)]
rows = []


def run_masked(name, active):
    cfg, tx, rx = pair(name)
    xs = [imgs[u] for u in active]
    with torch.no_grad():
        f, hw = tx(xs, active=active)
        z = channel_fixed(f, 2026)
        out = rx(z, active[0], hw)
    v = psnr(out, xs[0]).item()
    return out[0].cpu(), v


def run_todma(active):
    st = torch.load(os.path.join(CK, "todma_v256", "model.pt"), map_location=dev)
    cfg = Config.load(os.path.join(CK, "todma_v256", "config.json"))
    V, D = st["v"], st["l_s"]
    enc = SwinEncoder(cfg.in_ch, cfg.patch, cfg.dims, cfg.depths, cfg.heads, cfg.window, D).to(dev).eval()
    dec = SwinDecoder(cfg.in_ch, cfg.patch, tuple(reversed(cfg.dims)), tuple(reversed(cfg.depths)),
                      tuple(reversed(cfg.heads)), cfg.window, D).to(dev).eval()
    enc.load_state_dict(st["enc"]); dec.load_state_dict(st["dec"])
    Cb = st["vq"]["codebook"].to(dev)
    g = torch.Generator(device="cpu").manual_seed(2026)
    S = torch.randn(V, D, generator=g)
    S = (S / S.norm(dim=1, keepdim=True) * D ** 0.5).to(dev)
    K = len(active)
    with torch.no_grad():
        qidx, f, hw = [], None, None
        for u in active:
            ssem, hw = enc(imgs[u])
            d2 = ssem.pow(2).sum(-1, keepdim=True) - 2 * ssem @ Cb.t() + Cb.pow(2).sum(-1)
            qi = d2.argmin(-1); qidx.append(qi)
            cw = S[qi]
            f = cw if f is None else f + cw
        f = f / f.pow(2).mean().sqrt().clamp_min(1e-8)
        z = channel_fixed(f, 2026)
        # OMP
        zz = z.reshape(-1, D); resid = zz.clone(); picked = []
        A = torch.zeros(zz.shape[0], 0, D, device=dev)
        for k in range(K):
            corr = (resid @ S.t()).abs()
            if picked:
                corr.scatter_(1, torch.stack(picked, 1), -1e9)
            idx = corr.argmax(-1); picked.append(idx)
            A = torch.cat([A, S[idx].unsqueeze(1)], 1)
            At = A.transpose(1, 2)
            coef = torch.linalg.solve(A @ At + 1e-4 * torch.eye(k + 1, device=dev), A @ zz.unsqueeze(-1))
            resid = zz - (At @ coef).squeeze(-1)
        det = torch.stack(picked, 1).reshape(z.shape[0], z.shape[1], K)
        true = qidx[0]
        hit = (det == true.unsqueeze(-1)).any(-1)
        rec = torch.where(hit, true, det[..., 0])
        out = dec(Cb[rec], hw)
    return out[0].cpu(), psnr(out, imgs[active[0]]).item()


def grid(rows2, fname):
    nrows, ncols = len(rows2), len(rows2[0])
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.05 * ncols, 2.28 * nrows))
    if nrows == 1:
        axes = [axes]
    flat = [ax for row in axes for ax in (row if hasattr(row, "__len__") else [row])]
    panels = [p for row in rows2 for p in row]
    for ax, (img, title) in zip(flat, panels):
        ax.imshow(img.clamp(0, 1).permute(1, 2, 0).numpy())
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.4)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.90 if nrows>1 else 0.86, bottom=0.02, wspace=0.04, hspace=0.22)
    fig.savefig(os.path.join(a.out, fname))
    plt.close(fig)
    print("wrote", fname)


def panels_for(idx):
    global imgs
    imgs = [load(val[u * 200 + idx]).unsqueeze(0).to(dev) for u in range(4)]
    orig = imgs[0][0].cpu()
    r = {}
    r["ov_fixed"] = run_masked("swinsc_ov_u4_learned", [0, 1, 2, 3])
    r["ov_var"] = run_masked("swinsc_ov_u4var_learned", [0, 1, 2, 3])
    r["ov_oma"] = run_masked("swinsc_ov_u4_oma", [0, 1, 2, 3])
    r["ov_todma"] = run_todma([0, 1, 2, 3])
    r["ul_k1"] = run_masked("swinsc_ov_u4var_learned", [0])
    r["ul_k2"] = run_masked("swinsc_ov_u4var_learned", [0, 1])
    r["ul_todma1"] = run_todma([0])
    return orig, r


def cap(label, val, first_row):
    return (label + "\n" if first_row else "") + f"{val:.1f} dB"


rows_ov, rows_ul, allres = [], [], {}
for idx in (7, 21):
    fr = idx == 7
    orig, r = panels_for(idx)
    rows_ov.append([(orig, "Original" if fr else ""),
        (r["ov_fixed"][0], cap("Proposed (fixed)", r["ov_fixed"][1], fr)),
        (r["ov_var"][0], cap("Proposed (variable)", r["ov_var"][1], fr)),
        (r["ov_oma"][0], cap("Re-encoded OMA", r["ov_oma"][1], fr)),
        (r["ov_todma"][0], cap("ToDMA (genie)", r["ov_todma"][1], fr))])
    rows_ul.append([(orig, "Original" if fr else ""),
        (r["ul_k1"][0], cap("Proposed, $K{=}1$", r["ul_k1"][1], fr)),
        (r["ul_k2"][0], cap("Proposed, $K{=}2$", r["ul_k2"][1], fr)),
        (r["ov_var"][0], cap("Proposed, $K{=}4$", r["ov_var"][1], fr)),
        (r["ov_oma"][0], cap("Static OMA, any $K$", r["ov_oma"][1], fr)),
        (r["ul_todma1"][0], cap("ToDMA, $K{=}1$", r["ul_todma1"][1], fr))])
    for k, v in r.items():
        allres[f"{k}_img{idx}"] = v
grid(rows_ov, "fig_visual_overload.pdf")
grid(rows_ul, "fig_visual_underload.pdf")
res = allres
import sys as _sys
with open(os.path.join(a.out, "visual_psnr.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["panel", "psnr_db", "snr_db", "channel_seed"])
    for k, (_, v) in res.items():
        w.writerow([k, round(v, 3), a.snr, 2026])
print("saved CSV")
_sys.exit(0)
res = {}
res["ov_fixed"] = run_masked("swinsc_ov_u4_learned", [0, 1, 2, 3])
res["ov_var"] = run_masked("swinsc_ov_u4var_learned", [0, 1, 2, 3])
res["ov_oma"] = run_masked("swinsc_ov_u4_oma", [0, 1, 2, 3])
res["ov_todma"] = run_todma([0, 1, 2, 3])
grid([(orig, "Original"),
      (res["ov_fixed"][0], f"Proposed (fixed)\n{res['ov_fixed'][1]:.1f} dB"),
      (res["ov_var"][0], f"Proposed (variable)\n{res['ov_var'][1]:.1f} dB"),
      (res["ov_oma"][0], f"Re-encoded OMA\n{res['ov_oma'][1]:.1f} dB"),
      (res["ov_todma"][0], f"ToDMA (genie)\n{res['ov_todma'][1]:.1f} dB")],
     "fig_visual_overload.pdf")

res["ul_k1"] = run_masked("swinsc_ov_u4var_learned", [0])
res["ul_k2"] = run_masked("swinsc_ov_u4var_learned", [0, 1])
res["ul_k4"] = res["ov_var"]
res["ul_oma"] = run_masked("swinsc_ov_u4_oma", [0, 1, 2, 3])   # static OMA: load-invariant
res["ul_todma1"] = run_todma([0])
grid([(orig, "Original"),
      (res["ul_k1"][0], f"Proposed, $K{{=}}1$\n{res['ul_k1'][1]:.1f} dB"),
      (res["ul_k2"][0], f"Proposed, $K{{=}}2$\n{res['ul_k2'][1]:.1f} dB"),
      (res["ul_k4"][0], f"Proposed, $K{{=}}4$\n{res['ul_k4'][1]:.1f} dB"),
      (res["ul_oma"][0], f"Static OMA, any $K$\n{res['ul_oma'][1]:.1f} dB"),
      (res["ul_todma1"][0], f"ToDMA, $K{{=}}1$\n{res['ul_todma1'][1]:.1f} dB")],
     "fig_visual_underload.pdf")

with open(os.path.join(a.out, "visual_psnr.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["panel", "psnr_db", "snr_db", "img_index", "channel_seed"])
    for k, (_, v) in res.items():
        w.writerow([k, round(v, 3), a.snr, a.img_index, 2026])
print("saved CSV")
