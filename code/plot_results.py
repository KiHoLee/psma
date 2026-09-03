"""Canonical replot script for paper 12 (TSP), PSMA version (2026-09-04).

Reads ONLY data/ser_eval.csv (code/ser_eval.py: PSNR and semantic error rate
of every chain at every load and SNR, one source for every number in the
manuscript) and writes the five result figures to fig/ as vector PDF with
one shared geometry. Every population, operating point and axis label
comes from code/config_main.py. It also prints the rows of the load table
(Table V) so the manuscript quotes the same file the figures draw.

Geometry: every result figure is authored on a 6.8 x 3.0 in canvas with its
legend BELOW the axes (six or seven entries do not fit inside the axes of a
column-width print without stretching the y axis) and is included at
0.95 columnwidth, so every authored font prints at 0.95 * 252 / (6.8 * 72)
= 0.49 of itself: ticks 14 -> 6.8 pt, labels 14.5 -> 7.1 pt, legend 13 ->
6.4 pt. Guards (standard 12.7): every label and the legend box are checked
against the canvas and every curve point against the legend box.

    python code/plot_results.py

The mask-era script (Figs. 3-8 of the 2026-09-03 submission) is kept in
archive/masks_submission_20260903/plot_results.py.
"""
import csv
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_main import MAIN   # the ONE configuration (standard 7.9)

N_MAIN = MAIN["N"][0]           # 4, the provisioned frame of Figs. 2-4
L = MAIN["L"]                   # 8, the frame dimension and the N of Figs. 5-6
SNR_OP = MAIN["SNR_OP"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "ser_eval.csv")
FIG = os.path.join(ROOT, "fig")
os.makedirs(FIG, exist_ok=True)

TW, TH = 6.8, 3.45
plt.rcParams.update({
    "font.size": 14.0, "axes.labelsize": 14.5, "legend.fontsize": 13.0,
    "lines.markersize": 6.5, "lines.linewidth": 1.7,
    "pdf.fonttype": 42, "figure.figsize": (TW, TH),
    "legend.labelspacing": 0.25, "legend.handlelength": 1.6,
    "legend.handletextpad": 0.4, "legend.borderpad": 0.3,
    "legend.borderaxespad": 0.3, "legend.framealpha": 0.9,
})
AXRECT = [0.125, 0.50, 0.855, 0.48]       # axes; the band below holds the x label and the legend

# one shared label dictionary (7.2); every value is echoed verbatim in main.tex
LBL = {
    "psma": "PSMA (one model)",
    "masking_var": "Masked superposition (one model)",
    "masking_fixed": "Masked superposition, $N{=}K$",
    "oma_static": "Static OMA",
    "wh_dynamic": "Dynamic WH-OMA",
    "oma": "Re-encoded OMA, $N{=}K$",
    "deepma": "DeepMA, $N{=}K$",
    "deepma_offload": "DeepMA, off-load",
    "todma": "Token signatures (genie)",
}
STYLE = {
    "psma": dict(color="#d95f02", marker="o", ls="-"),
    "masking_var": dict(color="#e6a02a", marker="s", ls="--"),
    "masking_fixed": dict(color="#e6a02a", marker="s", ls="none", mfc="none", mew=1.5),
    "oma_static": dict(color="#1b5d99", marker="v", ls="--"),
    "wh_dynamic": dict(color="#1b5d99", marker="P", ls=":"),
    "oma": dict(color="#1b5d99", marker="^", ls="none"),
    "deepma": dict(color="#1b9e77", marker="D", ls="none"),
    "deepma_offload": dict(color="#1b9e77", marker="D", ls="-"),
    "todma": dict(color="#7570b3", marker="d", ls="-."),
}

rows = list(csv.DictReader(open(DATA)))
for r in rows:
    r["designed"] = int(r["designed"]); r["active"] = int(r["active"])
    r["snr"] = int(r["snr"]); r["psnr"] = float(r["psnr"]); r["ser"] = float(r["ser"])


def sel(scheme, designed=None, active=None, snr=None):
    out = [r for r in rows if r["scheme"] == scheme
           and (designed is None or r["designed"] == designed)
           and (active is None or r["active"] == active)
           and (snr is None or r["snr"] == snr)]
    return sorted(out, key=lambda r: (r["snr"], r["active"]))


def new_figure():
    fig = plt.figure(figsize=(TW, TH))
    return fig, fig.add_axes(AXRECT)


def legend_below(fig, ax, ncol=2):
    return ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.33), ncol=ncol,
                     bbox_transform=fig.transFigure, columnspacing=0.6)


def guard_and_save(fig, ax, name, legend):
    fig.canvas.draw()
    fw, fh = fig.get_size_inches() * fig.dpi

    def visible_ticklabels(axis, lo, hi):
        return [t.label1 for t in axis.get_major_ticks()
                if lo - 1e-9 <= t.get_loc() <= hi + 1e-9 and t.label1.get_text()]
    items = ([ax.xaxis.label, ax.yaxis.label]
             + visible_ticklabels(ax.xaxis, *ax.get_xlim())
             + visible_ticklabels(ax.yaxis, *ax.get_ylim()))
    for it in items:
        bb = it.get_window_extent()
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > fw + 0.5 or bb.y1 > fh + 0.5:
            raise RuntimeError(f"{name}: label '{it.get_text()}' clipped {bb}")
    lb = legend.get_window_extent()
    if lb.x0 < 0 or lb.y0 < 0 or lb.x1 > fw or lb.y1 > fh:
        raise RuntimeError(f"{name}: legend leaves canvas: {lb} vs {fw}x{fh}")
    xl = ax.xaxis.label.get_window_extent()
    if lb.y1 > xl.y0 - 2:
        raise RuntimeError(f"{name}: legend top {lb.y1:.0f} overlaps the x label bottom {xl.y0:.0f}")
    pad = 4
    for ln in ax.get_lines():
        for x, y in ax.transData.transform(list(zip(ln.get_xdata(), ln.get_ydata()))):
            if lb.x0 - pad < x < lb.x1 + pad and lb.y0 - pad < y < lb.y1 + pad:
                raise RuntimeError(f"{name}: curve '{ln.get_label()}' under legend at ({x:.0f},{y:.0f})")
    fig.savefig(os.path.join(FIG, name)); plt.close(fig)
    print("wrote", name)


def draw(ax, series, x_of, y_of, snr=None, active=None):
    """series: list of (scheme, designed, active-or-None). Missing rows are
    skipped with a message, so the script runs before and after a merge."""
    drawn = 0
    for k, (scheme, designed, act) in enumerate(series):
        d = sel(scheme, designed, act if act is not None else active, snr)
        if not d:
            print("skipping (no data yet):", scheme, designed); continue
        ax.plot([x_of(r) for r in d], [y_of(r) for r in d], label=LBL[scheme],
                markevery=(k % 2, 2) if x_of is X_SNR else None, **STYLE[scheme])
        drawn += 1
    return drawn


X_SNR = lambda r: r["snr"]
X_K = lambda r: r["active"]
Y_PSNR = lambda r: r["psnr"]
Y_T = lambda r: r["active"] * (1.0 - r["ser"])

# ---- Fig. 2: load sweep on the N = 4 frame at the operating point ----------
fig, ax = new_figure()
n = draw(ax, [("psma", N_MAIN, None), ("masking_var", N_MAIN, None), ("oma_static", N_MAIN, None),
              ("wh_dynamic", L, None), ("deepma_offload", N_MAIN, None), ("todma", N_MAIN, None)],
         X_K, Y_PSNR, snr=SNR_OP)
# the load-4 sweep stops at K = N_MAIN: WH-OMA and token rows run to K = 8
for ln in ax.get_lines():
    xs, ys = list(ln.get_xdata()), list(ln.get_ydata())
    keep = [i for i, x in enumerate(xs) if x <= N_MAIN]
    ln.set_data([xs[i] for i in keep], [ys[i] for i in keep])
ax.relim(); ax.autoscale_view()
ax.set_xlabel("Active users $K$ (provisioned $N=%d$)" % N_MAIN); ax.set_ylabel("PSNR (dB)")
ax.set_xticks(range(1, N_MAIN + 1)); ax.grid(True, alpha=0.3)
guard_and_save(fig, ax, "fig_load4.pdf", legend_below(fig, ax))

# ---- Figs. 3-4: PSNR vs SNR with one active user and at full load ----------
for name, K, series in (
        ("fig_snr_k1.pdf", 1, [("psma", N_MAIN, 1), ("masking_var", N_MAIN, 1), ("oma_static", N_MAIN, 1),
                               ("wh_dynamic", L, 1), ("deepma_offload", N_MAIN, 1), ("todma", N_MAIN, 1)]),
        ("fig_snr_k4.pdf", N_MAIN, [("psma", N_MAIN, N_MAIN), ("masking_var", N_MAIN, N_MAIN),
                                    ("masking_fixed", N_MAIN, N_MAIN), ("oma_static", N_MAIN, N_MAIN),
                                    ("wh_dynamic", L, N_MAIN), ("deepma", N_MAIN, N_MAIN),
                                    ("todma", N_MAIN, N_MAIN)])):
    fig, ax = new_figure()
    # at full load the retrained chains are single points per SNR: draw them as lines
    for scheme in ("masking_fixed", "oma", "deepma"):
        STYLE[scheme] = dict(STYLE[scheme]); STYLE[scheme]["ls"] = "-"
    draw(ax, series, X_SNR, Y_PSNR)
    for scheme in ("masking_fixed", "oma", "deepma"):
        STYLE[scheme]["ls"] = "none"
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("PSNR (dB)"); ax.grid(True, alpha=0.3)
    guard_and_save(fig, ax, name, legend_below(fig, ax))

# ---- Fig. 5: one model across eight loads --------------------------------------
fig, ax = new_figure()
series = [("psma", L, None), ("masking_var", L, None), ("wh_dynamic", L, None), ("todma", N_MAIN, None)]
draw(ax, series, X_K, Y_PSNR, snr=SNR_OP)
for scheme in ("oma", "masking_fixed", "deepma"):           # retrained at N = K: one marker per population
    d = [r for r in rows if r["scheme"] == scheme and r["snr"] == SNR_OP and r["designed"] == r["active"]]
    d = sorted(d, key=lambda r: r["active"])
    if d:
        ax.plot([r["active"] for r in d], [r["psnr"] for r in d], label=LBL[scheme], **STYLE[scheme])
ax.set_xlabel("Active users $K$ (one model, $N=%d$)" % L); ax.set_ylabel("PSNR (dB)")
ax.set_xticks(range(1, L + 1)); ax.grid(True, alpha=0.3)
guard_and_save(fig, ax, "fig_load8.pdf", legend_below(fig, ax))

# ---- Fig. 6: semantic throughput T(K) = sum over served users of (1 - SER) -----
# A scheme provisioned for N serves min(K, N): static OMA and DeepMA (N = 4)
# stop at their K = N value (the author's convention, 2026-09-03).
fig, ax = new_figure()
drawn = 0
for scheme, designed, cap in (("psma", L, None), ("masking_var", L, None), ("oma_static", N_MAIN, N_MAIN),
                              ("deepma_offload", N_MAIN, N_MAIN), ("wh_dynamic", L, None), ("todma", N_MAIN, None)):
    pts = []
    for K in range(1, L + 1):
        Ke = min(K, cap) if cap else K
        d = sel(scheme, designed, Ke, SNR_OP)
        if d:
            pts.append((K, Ke * (1.0 - d[0]["ser"])))
    if len(pts) < 2:
        print("skipping (no data yet): throughput,", scheme); continue
    ax.plot([p[0] for p in pts], [p[1] for p in pts], label=LBL[scheme], **STYLE[scheme]); drawn += 1
d = [r for r in rows if r["scheme"] == "oma" and r["snr"] == SNR_OP and r["designed"] == r["active"]]
d = sorted(d, key=lambda r: r["active"])
ax.plot([r["active"] for r in d], [r["active"] * (1 - r["ser"]) for r in d], label=LBL["oma"], **STYLE["oma"])
ax.set_xlabel("Active users $K$"); ax.set_ylabel("Throughput $T(K)$")
ax.set_xticks(range(1, L + 1)); ax.grid(True, alpha=0.3)
y0, y1 = ax.get_ylim(); ax.set_ylim(0.0, y1)
guard_and_save(fig, ax, "fig_throughput.pdf", legend_below(fig, ax))

# ---- Table V rows (PSNR at 10 and 20 dB, K = 1, 2, 3, 4, 6, 8) -----------------------
COLS = (1, 2, 3, 4, 6, 8)


def row(label, scheme, designed=None, nk=False):
    cells = []
    for K in COLS:
        if nk:
            d = [r for r in rows if r["scheme"] == scheme and r["designed"] == r["active"] == K and r["snr"] == snr]
        else:
            d = sel(scheme, designed, K, snr)
        cells.append("%.1f" % d[0]["psnr"] if d else "---")
    return "%s & %s \\\\" % (label, " & ".join(cells))


for snr in (SNR_OP, 20):
    print("\n%% Table V rows at %d dB" % snr)
    print(row("PSMA (one model), $N{=}4$", "psma", N_MAIN))
    print(row("PSMA (one model), $N{=}8$", "psma", L))
    print(row("Masked superposition (one model), $N{=}4$", "masking_var", N_MAIN))
    print(row("Masked superposition (one model), $N{=}8$", "masking_var", L))
    print(row("Masked superposition, $N{=}K$", "masking_fixed", nk=True))
    print(row("Re-encoded OMA, $N{=}K$", "oma", nk=True))
    print(row("DeepMA, $N{=}K$", "deepma", nk=True))
    print(row("Dynamic WH-OMA", "wh_dynamic", L))
    print(row("Static OMA", "oma_static", N_MAIN))
    print(row("DeepMA, off-load", "deepma_offload", N_MAIN))
    print(row("Token signatures (genie)", "todma", N_MAIN))
# ---- Fig. 7 (Sec. V-B): trained masks of the mask-based chain, per-user squared
# mask profile and overlap matrix (data/masks.csv, dumped by dump_artifacts.py).
# Own canvas (4.0 x 2.05 in, included at 0.80 columnwidth, print scale 0.70):
# authored 12 / 10.7 pt print at 8.4 / 7.5 pt.
MASKS = os.path.join(ROOT, "data", "masks.csv")
if os.path.exists(MASKS):
    mrows = list(csv.DictReader(open(MASKS)))
    U = len(mrows)
    LM = len([k for k in mrows[0] if k.startswith("m2_")])
    M2 = [[float(mrows[u][f"m2_{i}"]) for i in range(LM)] for u in range(U)]
    beta = [[sum(M2[u][i] * M2[v][i] for i in range(LM)) / LM for v in range(U)]
            for u in range(U)]
    fig, (axm, axb) = plt.subplots(
        1, 2, figsize=(4.0, 2.05), gridspec_kw={"width_ratios": [LM, U + 1.5]})
    for ax_ in (axm, axb):
        ax_.tick_params(labelsize=10.7)
    im0 = axm.imshow(M2, cmap="Oranges", aspect="auto", vmin=0)
    axm.set_xlabel("Dimension $i$", fontsize=12.0); axm.set_ylabel("User $u$", fontsize=12.0)
    axm.set_xticks(range(LM)); axm.set_yticks(range(U))
    axm.set_yticklabels([str(u + 1) for u in range(U)])
    axm.set_xticklabels([str(i + 1) for i in range(LM)])
    im1 = axb.imshow(beta, cmap="Oranges", aspect="auto", vmin=0)
    axb.set_xlabel("User $v$", fontsize=12.0); axb.set_ylabel("User $u$", fontsize=12.0)
    axb.set_xticks(range(U)); axb.set_yticks(range(U))
    axb.set_xticklabels([str(u + 1) for u in range(U)])
    axb.set_yticklabels([str(u + 1) for u in range(U)])
    axb.set_title(r"$\beta_{uv}$", fontsize=12.0)
    for u in range(U):
        for v in range(U):
            axb.text(v, u, f"{beta[u][v]:.1f}", ha="center", va="center", fontsize=10.7,
                     color="white" if beta[u][v] >= 1.6 else "black")
    cb = fig.colorbar(im0, ax=axm, fraction=0.046, pad=0.04)
    cb.ax.set_title("$m_u^2(i)$", fontsize=12.0, pad=3)
    cb.ax.tick_params(labelsize=10.7)
    fig.subplots_adjust(left=0.135, right=0.97, top=0.855, bottom=0.285, wspace=0.55)
    fig.canvas.draw()
    fw, fh = fig.get_size_inches() * fig.dpi
    marks = []
    for ax_ in (axm, axb, cb.ax):
        for it in (ax_.xaxis.label, ax_.yaxis.label, ax_.title):
            if not it.get_text():
                continue
            bb = it.get_window_extent()
            if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > fw + 0.5 or bb.y1 > fh + 0.5:
                raise RuntimeError(f"fig_masks: label '{it.get_text()}' clipped {bb}")
            marks.append((it.get_text(), bb))
    for i in range(len(marks)):
        for j in range(i + 1, len(marks)):
            if marks[i][1].overlaps(marks[j][1]):
                raise RuntimeError("fig_masks: labels '%s' and '%s' overlap" % (marks[i][0], marks[j][0]))
    fig.savefig(os.path.join(FIG, "fig_masks.pdf")); plt.close(fig)
    print("wrote fig_masks.pdf")

# ---- Fig. 8 (Sec. VI): quality of the progressive code versus prefix length,
# one user alone on the frame (data/prefix_eval.csv from code/prefix_eval.py)
PREFIX = os.path.join(ROOT, "data", "prefix_eval.csv")
if os.path.exists(PREFIX):
    prow = list(csv.DictReader(open(PREFIX)))
    fig, ax = new_figure()
    for name, d, st in (("PSMA (one model), $N{=}4$", 4, dict(color="#d95f02", marker="o", ls="-")),
                        ("PSMA (one model), $N{=}8$", 8, dict(color="#d95f02", marker="o", ls="--", mfc="none", mew=1.5))):
        pts = sorted([(int(r["prefix"]), float(r["psnr"])) for r in prow
                      if int(r["designed"]) == d and int(r["snr"]) == SNR_OP])
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], label=name, **st)
    # reference: the stand-alone head trained for b symbols at unit symbol power
    # is the dynamic WH-OMA point with K = L / b users (B = b codes each), so the
    # heads B = 1, 2, 4, 8 sit at b = 1, 2, 4, 8
    pts = []
    for b in (1, 2, 4, 8):
        r = sel("wh_dynamic", L, L // b, SNR_OP)
        if r:
            pts.append((b, r[0]["psnr"]))
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], label="Dynamic WH-OMA", **STYLE["wh_dynamic"])
    ax.set_xlabel("Prefix length $b$ (symbols per token, one user, %d dB)" % SNR_OP); ax.set_ylabel("PSNR (dB)")
    ax.set_xticks(range(1, L + 1)); ax.grid(True, alpha=0.3)
    guard_and_save(fig, ax, "fig_prefix.pdf", legend_below(fig, ax))

print("done")
