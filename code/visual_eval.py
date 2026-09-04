"""Reconstructed images of every compared scheme (PSMA manuscript, 2026-09-04).

For a few validation images, the script transmits the SAME source images
through every scheme at 10 dB under ONE fixed Rayleigh block-fading
realization (seed 2026), so the panels differ only by the access scheme and
the load. It writes, under --out (default data/visual):

    img<idx>/<scheme>_K<k>.png     one PNG per scheme and load (user 0's image)
    img<idx>/original.png
    montage_N4_img<idx>.png        the N = 4 frame: PSMA, learned masks, WH masks,
                                   DeepMA (N = 4), static OMA, token signatures
    montage_N8_img<idx>.png        the N = 8 frame: PSMA, learned masks, WH masks,
                                   DeepMA (N = 8)
    visual_psnr.csv                PSNR of every panel

The panels are released with the reproducibility package and are referred to
from Section VI-A of the manuscript; the tables carry the averages.

    python code/visual_eval.py --out data/visual
"""
import argparse, os, sys, glob, random, csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_main import MAIN, device as MAIN_DEVICE, wpath   # the ONE configuration (standard 7.9)
from swinsc import Config, Transmitter, Receiver
from swinsc.mask_mux import power_normalize
from swinsc.swin import SwinEncoder, SwinDecoder
from deepsc_ri.metrics import psnr

p = argparse.ArgumentParser()
p.add_argument("--crop", type=int, default=MAIN["CROP"])
p.add_argument("--snr", type=float, default=MAIN["SNR_OP"])
p.add_argument("--img_index", type=int, nargs="+", default=[7, 21, 33], help="validation image offsets")
p.add_argument("--seed", type=int, default=2026, help="channel realization shared by every panel")
p.add_argument("--out", default=wpath("data", "visual"))
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = MAIN_DEVICE()
CK = wpath("checkpoints")
L = MAIN["L"]
random.seed(MAIN["SEED"]); torch.manual_seed(MAIN["SEED"])
val = sorted(glob.glob(wpath("data", "imagenette160", "val", "*", "*.png")))
random.shuffle(val)


def load(path):
    im = Image.open(path).convert("RGB"); w, h = im.size; s = a.crop
    l, t = (w - s) // 2, (h - s) // 2
    return torch.from_numpy(np.array(im.crop((l, t, l + s, t + s)))).permute(2, 0, 1).float() / 255


_pairs = {}


def pair(name):
    if name not in _pairs:
        d = os.path.join(CK, name)
        cfg = Config.load(os.path.join(d, "config.json"))
        tx = Transmitter(cfg).to(dev).eval(); tx.load_state_dict(torch.load(os.path.join(d, "tx.pt"), map_location=dev))
        rx = Receiver(cfg).to(dev).eval(); rx.load_state_dict(torch.load(os.path.join(d, "rx.pt"), map_location=dev))
        _pairs[name] = (cfg, tx, rx)
    return _pairs[name]


def channel_fixed(f):
    """Rayleigh block fading + ZF with ONE realization (a.seed) for every panel."""
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    h = (torch.randn(1, generator=g) + 1j * torch.randn(1, generator=g)) / (2 ** 0.5)
    sigma = (10 ** (-a.snr / 10)) ** 0.5
    noise = torch.randn(f.shape, generator=g) * sigma
    return f + noise.to(f.device) / h.abs().item()


def hadamard(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / n ** 0.5


Hn = hadamard(L).to(dev)
imgs = None      # list of (1,3,H,W) tensors, one per user; user 0 is displayed


def run_chain(name, active):
    """Every learned chain that goes through Transmitter/Receiver (PSMA, learned
    masks, DeepMA, static OMA at full load)."""
    cfg, tx, rx = pair(name)
    xs = [imgs[u] for u in active]
    with torch.no_grad():
        f, hw = tx(xs, active=active)
        z = channel_fixed(f)
        out = rx(z, active[0], hw, K=len(active))
    return out[0].cpu(), psnr(out, xs[0]).item()


def run_wh(active):
    """WH masks with per-rate heads (the dynamic WH allocation of ser_eval.py)."""
    K = len(active)
    Bc = 1 << ((L // K).bit_length() - 1)
    cfg, tx, rx = pair("swinsc_ov_u%d_oma" % (L // Bc))
    scale = (K * Bc / L) ** 0.5
    with torch.no_grad():
        f, hw = None, None
        for j, u in enumerate(active):
            e, hw = tx.encode_user(imgs[u], u)
            if f is None:
                f = e.new_zeros(e.shape[0], e.shape[1], L)
            f[:, :, u * Bc:(u + 1) * Bc] = e
        f = power_normalize(f) @ Hn
        z = channel_fixed(f) @ Hn.t() * scale
        out = rx(z, active[0], hw)
    return out[0].cpu(), psnr(out, imgs[active[0]]).item()


def run_todma(active):
    st = torch.load(os.path.join(CK, "todma_v256", "model.pt"), map_location=dev)
    cfg = Config.load(os.path.join(CK, "todma_v256", "config.json"))
    V, D = st["v"], st["l_s"]
    enc = SwinEncoder(cfg.in_ch, cfg.patch, cfg.dims, cfg.depths, cfg.heads, cfg.window, D).to(dev).eval()
    dec = SwinDecoder(cfg.in_ch, cfg.patch, tuple(reversed(cfg.dims)), tuple(reversed(cfg.depths)),
                      tuple(reversed(cfg.heads)), cfg.window, D).to(dev).eval()
    enc.load_state_dict(st["enc"]); dec.load_state_dict(st["dec"])
    Cb = st["vq"]["codebook"].to(dev)
    g = torch.Generator(device="cpu").manual_seed(MAIN["SIGNATURE_SEED"])
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
        z = channel_fixed(f)
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


# (label, function, active set) per frame; user 0 is always the displayed user
PANELS_N4 = [
    ("PSMA, $K{=}1$", lambda: run_chain("swinsc_ov_u4_psma", [0])),
    ("PSMA, $K{=}2$", lambda: run_chain("swinsc_ov_u4_psma", [0, 1])),
    ("PSMA, $K{=}4$", lambda: run_chain("swinsc_ov_u4_psma", [0, 1, 2, 3])),
    ("WH masks, $K{=}1$", lambda: run_wh([0])),
    ("WH masks, $K{=}4$", lambda: run_wh([0, 1, 2, 3])),
    ("Learned masks, $K{=}1$", lambda: run_chain("swinsc_ov_u4var_learned", [0])),
    ("Learned masks, $K{=}4$", lambda: run_chain("swinsc_ov_u4var_learned", [0, 1, 2, 3])),
    ("Token signatures, $K{=}1$", lambda: run_todma([0])),
    ("Token signatures, $K{=}2$", lambda: run_todma([0, 1])),
    ("Token signatures, $K{=}4$", lambda: run_todma([0, 1, 2, 3])),
    ("DeepMA $N{=}4$, $K{=}1$", lambda: run_chain("swinsc_ov_u4_deepma", [0])),
    ("DeepMA $N{=}4$, $K{=}4$", lambda: run_chain("swinsc_ov_u4_deepma", [0, 1, 2, 3])),
    ("Static OMA, any $K$", lambda: run_chain("swinsc_ov_u4_oma", [0, 1, 2, 3])),
]
PANELS_N8 = [
    ("PSMA $N{=}8$, $K{=}1$", lambda: run_chain("swinsc_ov_u8_psma", [0])),
    ("PSMA $N{=}8$, $K{=}4$", lambda: run_chain("swinsc_ov_u8_psma", [0, 1, 2, 3])),
    ("PSMA $N{=}8$, $K{=}8$", lambda: run_chain("swinsc_ov_u8_psma", list(range(8)))),
    ("WH masks, $K{=}8$", lambda: run_wh(list(range(8)))),
    ("Learned masks $N{=}8$, $K{=}1$", lambda: run_chain("swinsc_ov_u8var_learned", [0])),
    ("Learned masks $N{=}8$, $K{=}8$", lambda: run_chain("swinsc_ov_u8var_learned", list(range(8)))),
    ("DeepMA $N{=}8$, $K{=}1$", lambda: run_chain("swinsc_ov_u8_deepma", [0])),
    ("DeepMA $N{=}8$, $K{=}8$", lambda: run_chain("swinsc_ov_u8_deepma", list(range(8)))),
]


def slug(label):
    return (label.replace("$", "").replace("{", "").replace("}", "").replace("=", "")
            .replace(",", "").replace(" ", "_").lower())


def save_png(img, path):
    Image.fromarray((img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(path)


def montage(panels, fname, ncols):
    n = len(panels); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.1 * ncols, 2.35 * nrows))
    flat = list(np.array(axes).reshape(-1))
    for ax in flat[n:]:
        ax.axis("off")
    for ax, (img, title) in zip(flat, panels):
        ax.imshow(img.clamp(0, 1).permute(1, 2, 0).numpy())
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.subplots_adjust(left=0.005, right=0.995, top=0.92, bottom=0.01, wspace=0.04, hspace=0.30)
    fig.savefig(os.path.join(a.out, fname), dpi=150)
    plt.close(fig)
    print("wrote", fname, flush=True)


#: Manuscript figure (fig_visual.pdf): two rows (K = 1 and K = 4 on the N = 4
#: frame) by six columns, authored 12.30 x 4.85 in and included at
#: 0.95\textwidth (6.84 in), print scale 0.556: the 14 pt headers print at
#: 7.8 pt, in line with the other figures' 7.6-7.8 pt (standard 9.3).
PANEL_W, ROW_H, HEADER_PT = 2.05, 2.28, 14.0
PAPER_COLS = ["Original", "PSMA", "Learned masks", "Token signatures", "DeepMA, $N{=}4$", "Static OMA"]
PAPER_KEY = {  # column label -> slug of the panel that supplies it, per row K
    1: {"PSMA": "psma_k1", "Learned masks": "learned_masks_k1", "Static OMA": "static_oma_any_k",
        "DeepMA, $N{=}4$": "deepma_n4_k1", "Token signatures": "token_signatures_k1"},
    4: {"PSMA": "psma_k4", "Learned masks": "learned_masks_k4", "Static OMA": "static_oma_any_k",
        "DeepMA, $N{=}4$": "deepma_n4_k4", "Token signatures": "token_signatures_k4"},
}


def paper_figure(orig, results, fname):
    """results: slug -> (image, psnr). Two rows (K = 1, K = 4), six columns."""
    fig, axes = plt.subplots(2, len(PAPER_COLS), figsize=(PANEL_W * len(PAPER_COLS), ROW_H * 2))
    for r, K in enumerate((1, 4)):
        for c, col in enumerate(PAPER_COLS):
            ax = axes[r][c]
            if col == "Original":
                img, title = orig, ("Original" if r == 0 else "")
            else:
                img, v = results[PAPER_KEY[K][col]]
                title = (col + "\n" if r == 0 else "") + "$K{=}%d$, %.1f dB" % (K, v)
            ax.imshow(img.clamp(0, 1).permute(1, 2, 0).numpy())
            ax.set_title(title, fontsize=HEADER_PT)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.86, bottom=0.01, wspace=0.04, hspace=0.24)
    fig.savefig(os.path.join(a.out, fname))   # preview copy; the manuscript figure is
    plt.close(fig)                                # assembled from the PNGs by plot_results.py
    print("wrote", fname, flush=True)


rows = []
for n_img, idx in enumerate(a.img_index):
    imgs = [load(val[u * 200 + idx]).unsqueeze(0).to(dev) for u in range(8)]
    d = os.path.join(a.out, "img%d" % idx); os.makedirs(d, exist_ok=True)
    orig = imgs[0][0].cpu(); save_png(orig, os.path.join(d, "original.png"))
    results = {}
    for frame, panels, ncols in (("N4", PANELS_N4, 4), ("N8", PANELS_N8, 4)):
        shown = [(orig, "Original")]
        for label, fn in panels:
            img, v = fn()
            save_png(img, os.path.join(d, slug(label) + ".png"))
            results[slug(label)] = (img, v)
            rows.append([idx, frame, label.replace("$", "").replace("{", "").replace("}", ""), round(v, 2)])
            shown.append((img, "%s\n%.1f dB" % (label, v)))
            print(idx, frame, label, round(v, 2), flush=True)
        montage(shown, "montage_%s_img%d.png" % (frame, idx), ncols)
    if n_img == 0:
        paper_figure(orig, results, "fig_visual.pdf")

with open(os.path.join(a.out, "visual_psnr.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["image", "frame", "panel", "psnr_db"]); w.writerows(rows)
print("saved", os.path.join(a.out, "visual_psnr.csv"))
