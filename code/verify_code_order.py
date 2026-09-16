"""Importance-ordering diagnostic of the PSMA code (audit, 2026-09-08).

Tests the claim that variable-load training with the per-prefix term orders
the L = 8 symbols of the PSMA code by importance. For the N=4 and N=8 PSMA
encoders, user 0's 200 validation images of ser_eval.py (the same images,
the same 25-image batches) are encoded alone (K = 1, load-agnostic encoder,
no channel) and, per code position i = 1..L:

  var, mean, mean_abs   statistics of the raw ordered symbol s(i) over all
                        tokens (1024 per image) and images (200): 204 800 values
  power_share           mean s(i)^2 / sum_j mean s(j)^2 (fraction of the frame power)
  var_norm, mean_abs_norm  the same after the per-image frame normalization of
                        the transmitter (Q is orthonormal, so the frame RMS equals
                        the RMS of the 8 symbols of that image)
  psnr_full             PSNR of the lone user with its FULL 8-symbol code at
                        10 dB, Rayleigh block fading, ZF, five draws, seeds
                        seed_fading(K=1, 10, rep, b) exactly as ser_eval.py /
                        verify_eval.py (this row equals psma K=1 of user 0)
  psnr_leave_one_out    PSNR when position i alone is zeroed before spreading,
                        the frame re-normalized to unit power (the transmitter's
                        normal pipeline, so the seven surviving symbols receive
                        the power the zeroed one released)
  psnr_loo_fixed_power  PSNR when position i is zeroed but the frame keeps the
                        scale of the full code (no re-normalization; surviving
                        symbols keep their per-symbol power, total power drops)

A monotone decrease of var / power_share with i, and a monotone RISE of the
leave-one-out PSNR with i (losing an early symbol hurts more), support the
claim. Rows: model, designed, position, ..., see header.

    PYTHONPATH=~/ViT python scripts/verify_code_order.py --out data/verify_code_order.csv
"""
import argparse, os, sys, glob, random, csv, time
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_main import MAIN, device as MAIN_DEVICE, wpath
from swinsc import Config, Transmitter, Receiver, Channel
from swinsc.mask_mux import power_normalize
from deepsc_ri.metrics import psnr

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=MAIN["VAL_IMAGES"])
p.add_argument("--reps", type=int, default=MAIN["EVAL_REPS"])
p.add_argument("--snr", type=float, default=MAIN["SNR_OP"])
p.add_argument("--crop", type=int, default=MAIN["CROP"])
p.add_argument("--models", nargs="+", default=["swinsc_ov_u4_psma", "swinsc_ov_u8_psma"])
p.add_argument("--out", default=wpath("data", "verify_code_order.csv"))
a = p.parse_args()
dev = MAIN_DEVICE()
random.seed(MAIN["SEED"]); torch.manual_seed(MAIN["SEED"])
T0 = time.time()


def batches():
    return [(rep, b) for rep in range(a.reps) for b in range(0, a.n, 25)]


def seed_fading(K, s, rep, b):
    torch.manual_seed(MAIN["SEED"] * 1_000_003 + K * 10_007 + (int(s) + 5) * 101 + rep * 13 + b // 25)


CK = wpath("checkpoints")
val = sorted(glob.glob(wpath("data", "imagenette160", "val", "*", "*.png")))
random.shuffle(val)


def load(path):
    im = Image.open(path).convert("RGB"); w, h = im.size; s = a.crop
    l, t = (w - s) // 2, (h - s) // 2
    return torch.from_numpy(__import__("numpy").array(im.crop((l, t, l + s, t + s)))).permute(2, 0, 1).float() / 255


imgs = torch.stack([load(val[i]) for i in range(a.n)])          # user 0's images (ser_eval.py order)
rows = []
for name in a.models:
    d = os.path.join(CK, name)
    cfg = Config.load(os.path.join(d, "config.json"))
    assert cfg.mask_type == "prog", name
    tx = Transmitter(cfg).to(dev).eval(); tx.load_state_dict(torch.load(os.path.join(d, "tx.pt"), map_location=dev))
    rx = Receiver(cfg).to(dev).eval(); rx.load_state_dict(torch.load(os.path.join(d, "rx.pt"), map_location=dev))
    L = cfg.l_e; N = cfg.users; Q = tx.spreader.Q
    assert tx.spreader.prefixes(1)[0] == [L]
    # ---- symbol statistics (no channel) ----------------------------------------
    s1 = torch.zeros(L, dtype=torch.float64); s2 = torch.zeros(L, dtype=torch.float64); sa = torch.zeros(L, dtype=torch.float64)
    n1 = torch.zeros(L, dtype=torch.float64); n2 = torch.zeros(L, dtype=torch.float64); na = torch.zeros(L, dtype=torch.float64)
    cnt = 0
    with torch.no_grad():
        for b in range(0, a.n, 25):
            x = imgs[b:b + 25].to(dev)
            e, hw = tx.encode_user(x, 0, K=1)                            # (B, T, L) ordered symbols
            ed = e.double()
            s1 += ed.sum((0, 1)).cpu(); s2 += ed.pow(2).sum((0, 1)).cpu(); sa += ed.abs().sum((0, 1)).cpu()
            en = power_normalize(ed)                                     # per-image unit RMS over the L*T values
            n1 += en.sum((0, 1)).cpu(); n2 += en.pow(2).sum((0, 1)).cpu(); na += en.abs().sum((0, 1)).cpu()
            cnt += ed.shape[0] * ed.shape[1]
    mean = s1 / cnt; var = s2 / cnt - mean ** 2; mabs = sa / cnt; ms2 = s2 / cnt
    meann = n1 / cnt; varn = n2 / cnt - meann ** 2; mabsn = na / cnt
    share = ms2 / ms2.sum()
    print("%s: %d symbol values per position; mean s^2 per position %s" % (name, cnt, ["%.4f" % v for v in ms2.tolist()]), flush=True)
    # ---- leave-one-out PSNR at K = 1 ------------------------------------------
    ch = Channel("rayleigh", a.snr)
    ps_full, ps_loo, ps_fix = [], [[] for _ in range(L)], [[] for _ in range(L)]
    with torch.no_grad():
        for rep, b in batches():
            x = imgs[b:b + 25].to(dev)
            e, hw = tx.encode_user(x, 0, K=1)
            f_full = e @ Q                                               # spread(e, 0, 1) for K = 1
            rms = f_full.reshape(f_full.shape[0], -1).pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-8).view(-1, 1, 1)
            seed_fading(1, a.snr, rep, b)
            z = ch(power_normalize(f_full))                              # identical to tx(xs, active=[0]) + ch
            ps_full.append(psnr(rx(z, 0, hw, K=1), x).cpu())
            for i in range(L):
                em = e.clone(); em[..., i] = 0
                fm = em @ Q
                seed_fading(1, a.snr, rep, b)                            # same fading and noise as the full code
                z = ch(power_normalize(fm))
                ps_loo[i].append(psnr(rx(z, 0, hw, K=1), x).cpu())
                seed_fading(1, a.snr, rep, b)
                z = ch(fm / rms)                                         # keep the full-code scale
                ps_fix[i].append(psnr(rx(z, 0, hw, K=1), x).cpu())
    pf = torch.cat(ps_full).mean().item()
    print("%s: psnr_full(K=1, %g dB) = %.3f  [%.0fs]" % (name, a.snr, pf, time.time() - T0), flush=True)
    for i in range(L):
        rows.append([name.replace("swinsc_ov_u", "psma_u"), N, i + 1,
                     "%.6g" % var[i].item(), "%.6g" % mean[i].item(), "%.6g" % mabs[i].item(), "%.6g" % share[i].item(),
                     "%.6g" % varn[i].item(), "%.6g" % mabsn[i].item(),
                     "%.3f" % pf, "%.3f" % torch.cat(ps_loo[i]).mean().item(), "%.3f" % torch.cat(ps_fix[i]).mean().item()])
        print("  pos %d var=%.4f mean_abs=%.4f share=%.3f  loo=%.3f fixed=%.3f" % (i + 1, var[i], mabs[i], share[i], float(rows[-1][-2]), float(rows[-1][-1])), flush=True)

with open(a.out, "w", newline="") as fo:
    w = csv.writer(fo)
    w.writerow(["model", "designed", "position", "var", "mean", "mean_abs", "power_share", "var_norm", "mean_abs_norm",
                "psnr_full", "psnr_leave_one_out", "psnr_loo_fixed_power"]); w.writerows(rows)
print("saved", a.out, len(rows), "rows; %.0fs" % (time.time() - T0))
