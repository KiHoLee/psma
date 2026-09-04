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
  wh_dynamic     K = 1..8 (designed = 8 codes)  dynamic Walsh-Hadamard code allocation,
                                                B = 8/K codes per user (largest power of two),
                                                re-encoded OMA heads, idle-code power reused
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
from swinsc.mask_mux import power_normalize
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
p.add_argument("--wh_only", action="store_true",
               help="evaluate only the clean floor and the dynamic WH chain (rows to be merged into ser_eval.csv)")
p.add_argument("--prog_only", action="store_true",
               help="evaluate only the clean floor and the progressive-spread prototype (research note)")
p.add_argument("--overload_only", action="store_true",
               help="evaluate only the clean floor, the PSMA tight-frame extension beyond the frame "
                    "dimension (K = L+1..nmax, scheme psma_tf) and the token-signature chain at those loads; "
                    "run with --nmax 16")
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
    if a.wh_only or a.overload_only or (a.prog_only and not scheme.startswith(("prog", "psma"))):
        return
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
N0 = MAIN["N"][0]              # the main provisioned frame (4)
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
# Progressive-spread prototype (research note, 2026-09-03): one N=8 model,
# prefix b_j(K) of an importance-ordered 8-symbol code on disjoint orthonormal
# Walsh-Hadamard codes, evaluated at K = 1..8. Skipped when not trained.
if os.path.exists(os.path.join(CK, "swinsc_ov_u8_prog", "tx.pt")):
    eval_chain("swinsc_ov_u8_prog", "prog_v1", 8, list(range(1, 9)))
# PSMA proper (2026-09-04): nested-dropout term (--multi_prefix) and prefix-length
# conditioning of the decoders (--load_cond); one N=4 model and one N=8 model.
for N in (4, 8):
    if os.path.exists(os.path.join(CK, "swinsc_ov_u%d_psma" % N, "tx.pt")):
        eval_chain("swinsc_ov_u%d_psma" % N, "psma", N, list(range(1, N + 1)))

# ---- dynamic Walsh-Hadamard code allocation (author request, 2026-09-03) -----------
# Code-domain orthogonal access with a pool of L_e length-L_e Walsh-Hadamard
# codes assigned at run time: K active users receive B = L_e/K codes each when
# K divides L_e and the largest power of two below L_e/K otherwise (on L_e = 8:
# K = 3 -> 2 codes, K = 5..7 -> 1 code), the codes left over stay idle, and the
# frame is normalized over the ACTIVE codes so the idle-code power returns to
# the active users (the receiver, which knows the grant, undoes that gain so
# its decoder sees the scale it was trained at; the noise shrinks with it).
# User u's B symbols per token ride its B codes and are despread by the same
# orthonormal rows. The chain therefore needs the re-encoded OMA head trained
# for its code count (checkpoints swinsc_ov_u{L_e/B}_oma, B in {8,4,2,1}).
# Under a scalar block-fading gain with ZF and white noise the Hadamard matrix
# is an orthonormal rotation of the block layout, so this is a physical
# implementation of adaptive OMA (Table III) that also serves the populations
# where N does not divide L, at reduced code efficiency K*B/L_e.
def hadamard(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / n ** 0.5


L_E = MAIN["L"]
Hn = hadamard(L_E).to(dev)
for K in (range(1, min(a.nmax, L_E) + 1) if not (a.prog_only or a.overload_only) else []):
    Bc = 1 << ((L_E // K).bit_length() - 1)            # largest power of two <= L_e/K
    pr = pair("swinsc_ov_u%d_oma" % (L_E // Bc))
    if pr is None:
        continue
    cfg, tx, rx = pr
    assert tx.blk == Bc, (tx.blk, Bc)
    act = list(range(K))
    scale = (K * Bc / L_E) ** 0.5
    for s in a.snrs:
        ch = Channel("rayleigh", s)
        ps, errs = [], []
        with torch.no_grad():
            for b in range(0, a.n, 25):
                xs = [imgs[u][b:b + 25].to(dev) for u in act]
                f, hw = None, None
                for j, u in enumerate(act):
                    e, hw = tx.encode_user(xs[j], u)           # (B, N, Bc) symbols of user u
                    if f is None:
                        f = e.new_zeros(e.shape[0], e.shape[1], L_E)
                    f[:, :, u * Bc:(u + 1) * Bc] = e            # symbol i of user u rides code u*Bc + i
                f = power_normalize(f) @ Hn                     # spread (row j of H is code j)
                z = ch(f) @ Hn.t() * scale                      # despread, undo the idle-code gain
                for j, u in enumerate(act):
                    out = rx(z, u, hw)
                    ps.append(psnr(out, xs[j]).cpu())
                    errs.append((predict(out).cpu() != labels[u][b:b + 25]))
        record("wh_dynamic", L_E, K, s, ps, errs)

# ---- PSMA beyond the frame dimension (2026-09-04): tight-frame signatures + LMMSE --
# For K > L no orthonormal assignment exists. Every active user sends the FIRST
# symbol of its progressive code (b = 1) on a unit-norm real harmonic tight
# frame S in R^{L x K}, S S^T = (K/L) I, the frame is normalized to unit power
# per dimension (so each symbol carries power L/K), and receiver k combines with
# the LMMSE vector W_k = (S S^T + sigma^2/|h|^2 I)^{-1} s_k, which for a tight
# frame is a positive scalar times s_k, so the matched filter s_k^T z and the
# LMMSE combiner deliver the same SINR; the matched filter is used because it
# leaves the symbol at the unit scale the decoder was trained on. The N = 8 PSMA
# model is used WITHOUT retraining: its b = 1 prefix and prefix-length
# conditioning were trained at K = 5..8, and users beyond eight reuse the
# eight trained decoders cyclically (all decoders share one training recipe).
# A companion Monte Carlo at |h|^2 = 1 (AWGN) records the post-combining SINR
# against the closed form P / (P (K/L - 1) + sigma^2) with P the measured
# per-symbol power, the check the writing standard asks for (6.6).
def harmonic_frame(L, K):
    m = torch.arange(1, L // 2 + 1, dtype=torch.float64).view(-1, 1)
    k = torch.arange(K, dtype=torch.float64).view(1, -1)
    ang = 2 * torch.pi * m * k / K
    S = torch.cat([torch.cos(ang), torch.sin(ang)], 0) * (2.0 / L) ** 0.5   # (L, K), unit-norm columns
    G = S @ S.t()
    assert torch.allclose(G, (K / L) * torch.eye(L, dtype=torch.float64), atol=1e-9), "not a tight frame"
    return S.float()


if a.overload_only and os.path.exists(os.path.join(CK, "swinsc_ov_u8_psma", "tx.pt")):
    cfg8, tx8, rx8 = pair("swinsc_ov_u8_psma")
    sinr_rows = []
    for K in range(L_E + 1, a.nmax + 1):
        S = harmonic_frame(L_E, K).to(dev)                     # (L, K)
        act = list(range(K))
        for s in a.snrs + ["sinr"]:
            ch = Channel("awgn", 10.0) if s == "sinr" else Channel("rayleigh", s)
            ps, errs = [], []
            sig_p, int_p, noi_p, sym_p = 0.0, 0.0, 0.0, 0.0
            with torch.no_grad():
                for b in range(0, a.n, 25):
                    xs = [imgs[u][b:b + 25].to(dev) for u in act]
                    syms, hw = [], None
                    for j, u in enumerate(act):
                        e, hw = tx8.encode_user(xs[j], u % cfg8.users)
                        syms.append(e[..., :1])                # (B, T, 1) first symbol
                    X = torch.cat(syms, -1)                    # (B, T, K)
                    fr = X @ S.t()                             # (B, T, L) superposed frame
                    pw = fr.reshape(fr.shape[0], -1).pow(2).mean(1).sqrt().clamp_min(1e-8).view(-1, 1, 1)
                    f = fr / pw
                    z = ch(f)
                    if s == "sinr":
                        n = z - f
                        for j in range(K):
                            sk = S[:, j]
                            sig = (X[..., j:j + 1] / pw) * 1.0      # own symbol after unit-norm MF
                            intf = (f @ sk).unsqueeze(-1) - sig
                            noi = (n @ sk).unsqueeze(-1)
                            sig_p += sig.pow(2).mean().item(); int_p += intf.pow(2).mean().item()
                            noi_p += noi.pow(2).mean().item(); sym_p += sig.pow(2).mean().item()
                        continue
                    for j, u in enumerate(act):
                        y = torch.nn.functional.pad((z @ S[:, j]).unsqueeze(-1), (0, L_E - 1))
                        d = u % cfg8.users
                        dec = rx8.decoders[d]
                        out = dec(rx8.reduce[d](y), hw, **({"K": 1} if dec.cond is not None else {}))
                        ps.append(psnr(out, xs[j]).cpu())
                        errs.append((predict(out).cpu() != labels[u][b:b + 25]))
            if s == "sinr":
                nb = (a.n // 25) * K
                P = sym_p / nb; sigma2 = 10 ** (-10.0 / 10)
                meas = 10 * torch.log10(torch.tensor(sig_p / (int_p + noi_p))).item()
                form = 10 * torch.log10(torch.tensor(P / (P * (K / L_E - 1) + sigma2))).item()
                sinr_rows.append([K, round(P, 5), round(meas, 3), round(form, 3)])
                print("sinr check K=%d P=%.4f meas=%.3f dB formula=%.3f dB" % (K, P, meas, form), flush=True)
            else:
                record("psma_tf", cfg8.users, K, s, ps, errs)
    with open(os.path.join(os.path.dirname(a.out), "overload_sinr.csv"), "w", newline="") as fo:
        w = csv.writer(fo); w.writerow(["active", "symbol_power", "sinr_meas_db", "sinr_formula_db"]); w.writerows(sinr_rows)

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


for K in ([] if (a.wh_only or a.prog_only) else
          (range(L_E + 1, a.nmax + 1) if a.overload_only else range(1, a.nmax + 1))):
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
