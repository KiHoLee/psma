"""Paired statistics for the 10 dB, N=4 frame (audit, 2026-09-08).

Reads data/verify_psma_10db_perimage.csv (written by verify_eval.py) and
data/ser_eval.csv and writes data/verify_paired_stats.csv.

Part A (rows kind=level): per scheme and load K, the mean PSNR over the
  n = 200 images x 5 draws x K users samples and its standard error
  s / sqrt(n) treating the n samples as independent (they are not strictly:
  the K users of a slot share one fading draw, and the 5 draws of an image
  share the image; the 'se_by_image' column therefore also gives the standard
  error of the 200 per-image means, s_img / sqrt(200), which respects the
  common-fading design).
Part B (rows kind=paired): PSMA minus reference on identical (image, draw,
  user) samples: mean difference, its standard error, the paired t statistic
  (n-1 degrees of freedom) and, for the design-respecting variant, the same
  statistics over the 200 per-image mean differences (df 199).
  References: masking_var (paired at every K: same seeds), oma_static_k (the
  static allocation of Prop. 1 re-evaluated under the load-K seeds, paired at
  every K), oma_static (the manuscript's replicated K=4 value; paired only at
  K=4, unpaired Welch statistics at K<4).
Part C (rows kind=throughput_se): binomial standard error of the throughput
  Theta(K) = sum_u (1 - SER_u) from the SER values of ser_eval.csv:
  se_1000 = sqrt( K * SER*(1-SER) / 1000 ) with n = 1000 trials per user (5 draws)
  se_200  = sqrt( K * SER*(1-SER) / 200 )  with n = 200 source images per user

    python code/verify_paired_stats.py
"""
import csv, math, os, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PER = os.path.join(ROOT, "data", "verify_psma_10db_perimage.csv")
SER = os.path.join(ROOT, "data", "ser_eval.csv")
OUT = os.path.join(ROOT, "data", "verify_paired_stats.csv")


def mean_sd(v):
    n = len(v); m = sum(v) / n
    var = sum((x - m) ** 2 for x in v) / (n - 1) if n > 1 else float("nan")
    return m, math.sqrt(var), n


def t_p(t, df):
    """two-sided p-value of Student t with df degrees of freedom (regularized beta via continued fraction)."""
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5
    # regularized incomplete beta I_x(a,b) by Lentz continued fraction
    def betacf(a, b, x):
        MAXIT, EPS, FPMIN = 200, 3e-14, 1e-300
        qab, qap, qam = a + b, a + 1, a - 1
        c, d = 1.0, 1 - qab * x / qap
        d = 1 / (d if abs(d) > FPMIN else FPMIN); h = d
        for m in range(1, MAXIT + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1 + aa * d; d = 1 / (d if abs(d) > FPMIN else FPMIN); c = 1 + aa / c; c = c if abs(c) > FPMIN else FPMIN; h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1 + aa * d; d = 1 / (d if abs(d) > FPMIN else FPMIN); c = 1 + aa / c; c = c if abs(c) > FPMIN else FPMIN
            de = d * c; h *= de
            if abs(de - 1) < EPS:
                break
        return h
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta)
    if x < (a + 1) / (a + b + 2):
        ib = front * betacf(a, b, x) / a
    else:
        ib = 1 - front * betacf(b, a, 1 - x) / b
    return ib


rows = list(csv.DictReader(open(PER)))
# key: (scheme, K) -> {(rep, user, image): psnr}
D = defaultdict(dict)
for r in rows:
    if int(float(r["snr"])) != 10 or int(r["designed"]) != 4:
        continue
    D[(r["scheme"], int(r["active"]))][(int(r["rep"]), int(r["user"]), int(r["image_index"]))] = float(r["psnr"])

out = []
hdr = ["kind", "scheme", "reference", "K", "n", "mean", "se", "se_by_image", "diff", "se_diff", "t", "df", "p", "diff_by_image_se", "t_by_image", "p_by_image", "note"]
schemes = ["psma", "masking_var", "oma_static", "oma_static_k", "deepma_offload"]
for K in (1, 2, 3, 4):
    for sch in schemes:
        d = D.get((sch, K))
        if not d:
            continue
        v = list(d.values()); m, s, n = mean_sd(v)
        img = defaultdict(list)
        for (rep, u, i), x in d.items():
            img[(u, i)].append(x)
        # per-image means (200 K images at load K); SE over those
        mi, si, ni = mean_sd([sum(x) / len(x) for x in img.values()])
        out.append(["level", sch, "", K, n, "%.4f" % m, "%.4f" % (s / math.sqrt(n)), "%.4f" % (si / math.sqrt(ni)),
                    "", "", "", "", "", "", "", "", "%d image-user slots" % ni])
    for ref in ("masking_var", "oma_static_k", "oma_static", "deepma_offload"):
        A, B = D.get(("psma", K)), D.get((ref, K))
        if not A or not B:
            continue
        keys = sorted(set(A) & set(B))
        paired = len(keys) == len(A) == len(B)
        if ref == "oma_static" and K < 4:
            paired = False                            # different seeds (K=4 draws vs K draws)
        if paired:
            diffs = [A[k] - B[k] for k in keys]
            md, sd, n = mean_sd(diffs); se = sd / math.sqrt(n); t = md / se
            img = defaultdict(list)
            for (rep, u, i), x in zip(keys, diffs):
                img[(u, i)].append(x)
            mdi, sdi, ni = mean_sd([sum(x) / len(x) for x in img.values()]); sei = sdi / math.sqrt(ni); ti = mdi / sei
            out.append(["paired", "psma", ref, K, n, "", "", "", "%.4f" % md, "%.4f" % se, "%.2f" % t, n - 1, "%.3g" % t_p(t, n - 1),
                        "%.4f" % sei, "%.2f" % ti, "%.3g" % t_p(ti, ni - 1), "same image, same fading draw, same user slot"])
        else:
            ma, sa, na = mean_sd(list(A.values())); mb, sb, nb = mean_sd(list(B.values()))
            se = math.sqrt(sa ** 2 / na + sb ** 2 / nb); t = (ma - mb) / se
            df = se ** 4 / ((sa ** 2 / na) ** 2 / (na - 1) + (sb ** 2 / nb) ** 2 / (nb - 1))
            out.append(["unpaired", "psma", ref, K, "%d/%d" % (na, nb), "", "", "", "%.4f" % (ma - mb), "%.4f" % se, "%.2f" % t, "%.0f" % df, "%.3g" % t_p(t, df),
                        "", "", "", "Welch: oma_static row is the K=4 measurement (K=4 seeds), not the load-K draws"])

# Part C: throughput standard error from ser_eval.csv at 10 dB
ser = [r for r in csv.DictReader(open(SER)) if int(float(r["snr"])) == 10]
for r in ser:
    sch, N, K, s = r["scheme"], int(r["designed"]), int(r["active"]), float(r["ser"])
    if sch not in ("psma", "masking_var", "oma_static", "deepma_offload", "todma"):
        continue
    if not (K == N or (sch == "psma" and N == 8) or (sch in ("masking_var", "deepma_offload") and N == 8)):
        continue
    n_img = int(r["n_images"]) // K
    v = s * (1 - s)
    out.append(["throughput_se", sch, "", K, n_img, "%.4f" % (K * (1 - s)), "%.4f" % math.sqrt(K * v / 1000), "%.4f" % math.sqrt(K * v / 200),
                "", "", "", "", "", "", "", "", "designed N=%d, SER=%.4f; se column = sqrt(K*SER*(1-SER)/1000), se_by_image column = sqrt(K*SER*(1-SER)/200)" % (N, s)])

with open(OUT, "w", newline="") as f:
    w = csv.writer(f); w.writerow(hdr); w.writerows(out)
for r in out:
    print(",".join(str(x) for x in r))
print("saved", OUT, len(out), "rows")
