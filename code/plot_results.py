"""Canonical replot script for paper 12 (TSP), PSMA version (2026-09-04).

Reads only data/ (ser_eval.csv, masks.csv, prefix_eval.csv, verify_code_order.csv, visual/) (code/ser_eval.py: PSNR and semantic error rate
of every chain at every load and SNR, one source for every number in the
manuscript) and writes the result figures (Figs. 3-8) and the reconstruction panel
(Fig. 9, from data/visual/) to fig/ as vector PDF with
one shared geometry. Every population, operating point and axis label
comes from code/config_main.py. It also prints the rows of the load table
(Table VI) so the manuscript quotes the same file the figures draw.

Geometry: every result figure is authored on a 6.8 x 5.3 in canvas with an
8:6 axes box and its legend INSIDE the axes (author's request), placed by a
corner sweep with y-axis headroom so that no curve point lies under it, and
is included at 0.58 columnwidth (2026-09-09, after the page budget), so
every authored font prints at 0.58 * 252 / (6.8 * 72) = 0.299 of itself:
ticks 20.5 -> 6.1 pt, axis labels 21 -> 6.3 pt, legend 20.5 -> 6.1 pt (the
legend is kept small on the author's request so its box takes little of
the plot area, but no printed font falls below 6 pt, standard 9.8). Guards (standard 12.7): every label and the legend box are checked
against the canvas and every curve point against the legend box.

    python code/plot_results.py

The mask-era script (Figs. 3-8 of the 2026-09-03 submission) is kept in
archive/masks_submission_20260903/plot_results.py.
"""
import csv
import os
import sys
from decimal import Decimal, ROUND_HALF_UP
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_main import MAIN   # the ONE configuration (standard 7.9)

N_MAIN = MAIN["N"][0]           # 4, the provisioned frame of Figs. 3-5
L = MAIN["L"]                   # 8, the frame dimension and the N of Figs. 7-8
SNR_OP = MAIN["SNR_OP"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "ser_eval.csv")
FIG = os.path.join(ROOT, "fig")
os.makedirs(FIG, exist_ok=True)

# Standard 9.3 / 9.8: every result figure shares one canvas (6.8 x 5.3 in) and
# one axes box of 5.51 x 4.08 in, included at 0.58 columnwidth (146 pt),
# so the print scale is 146 / (6.8 * 72) = 0.299 and the authored 20.5 pt
# fonts print at 6.1 pt (ticks and legend), 21 pt labels at 6.3 pt. The
# legend sits inside the axes (author's request, kept small so it takes
# little of the plot area) with y-axis headroom.
TW, TH = 6.8, 4.6      # wider aspect (2026-09-10): more plot width at the same printed height
plt.rcParams.update({
    "font.size": 14.5, "axes.labelsize": 15.0, "legend.fontsize": 12.5,
    "lines.markersize": 7.0, "lines.linewidth": 1.8,
    "pdf.fonttype": 42, "figure.figsize": (TW, TH), "mathtext.fontset": "cm",
    # compact legend (author's request, 2026-09-10): about 0.85 of the tick
    # font, short handles and tight padding so the box takes less of the axes
    "legend.labelspacing": 0.15, "legend.handlelength": 1.1,
    "legend.handletextpad": 0.3, "legend.borderpad": 0.25,
    "legend.borderaxespad": 0.25, "legend.framealpha": 0.9,
})
AXRECT = [0.165, 0.18, 0.815, 0.775]       # 8:6 axes (5.68 x 4.24 in); the legend sits inside the axes

# one shared label dictionary (7.2); every value is echoed verbatim in main.tex.
# The N = 4 and N = 8 PSMA models are two trained networks and carry two
# entries (9.1: one scheme, one style); so do the N = 8 mask-based model and
# its N = 4 counterpart.
LBL = {
    "psma": "PSMA",
    "psma4": "PSMA, $N{=}4$",
    "psma8": "PSMA, $N{=}8$",
    "masking_var": "Learned masks",
    "masking_var8": "Learned masks, $N{=}8$",
    "oma_static": "Static OMA",
    "oma_static4": "Static OMA, $N{=}4$",
    "oma_static8": "Static OMA, $N{=}8$",
    "deepma": "DeepMA",
    "deepma8": "DeepMA, $N{=}8$",
    "deepma_offload": "DeepMA, $N{=}4$",
    "todma": "Token signatures",
    # Fig. 5 ablation of the per-prefix term (prefix_eval.csv, column "model")
    "psma_pow2": "PSMA, $N{=}4$, powers of two",
    "psma8_pow2": "PSMA, $N{=}8$, powers of two",
    "prog_v1": "PSMA, $N{=}8$, no prefix term",
}
# One (color, marker, line style) triple per scheme, never overridden per
# figure: proposed = solid, single-model conventional = dashed, the
# no-prefix-term prototype = dotted, token signatures = dash-dot; N = 8
# variants share the triple of their N = 4 model with open markers.
STYLE = {
    "psma": dict(color="#d95f02", marker="o", ls="-"),
    "psma4": dict(color="#d95f02", marker="o", ls="-"),
    "psma8": dict(color="#d95f02", marker="o", ls="-", mfc="none", mew=1.6),
    "masking_var": dict(color="#e6a02a", marker="s", ls="--"),
    "masking_var8": dict(color="#e6a02a", marker="s", ls="--", mfc="none", mew=1.6),
    "oma_static": dict(color="#1b5d99", marker="v", ls="--"),
    "oma_static8": dict(color="#1b5d99", marker="v", ls="--", mfc="none", mew=1.6),
    "deepma": dict(color="#1b9e77", marker="D", ls="-", mfc="none", mew=1.6),
    "deepma8": dict(color="#1b9e77", marker="D", ls="--", mfc="none", mew=1.6),
    "deepma_offload": dict(color="#1b9e77", marker="D", ls="--"),
    "todma": dict(color="#7570b3", marker="d", ls="-."),
    "psma_pow2": dict(color="#7f3b08", marker="^", ls="--"),
    "psma8_pow2": dict(color="#7f3b08", marker="^", ls="--", mfc="none", mew=1.6),
    "prog_v1": dict(color="#6a6a6a", marker="x", ls=":", mew=1.8),
}
# one declared order for legends (tables in main.tex follow the same order)
ORDER = ["psma", "psma4", "psma8", "psma_pow2", "psma8_pow2", "prog_v1", "masking_var", "masking_var8",
         "todma", "deepma", "deepma8", "deepma_offload", "oma_static", "oma_static8"]
ORDER_LBL = [LBL[k] for k in ORDER]

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


def guard_pad(fig):
    """Legend/inset boxes are inflated by the marker radius plus the line width
    (standard 12.9), in display pixels: a marker whose center clears the box
    can still touch its frame."""
    return (plt.rcParams["lines.markersize"] / 2 + plt.rcParams["lines.linewidth"]) * fig.dpi / 72


def curve_samples(ax, per_segment=24):
    """Display-space samples of every curve, including points interpolated
    along each segment, so a legend or inset resting on a LINE between two
    markers is caught as well as one resting on a marker (12.9)."""
    out = []
    for ln in ax.get_lines():
        xs, ys = list(ln.get_xdata()), list(ln.get_ydata())
        pts = []
        for i in range(len(xs)):
            pts.append((xs[i], ys[i]))
            if i + 1 < len(xs):
                for k in range(1, per_segment):
                    t = k / per_segment
                    pts.append((xs[i] + t * (xs[i + 1] - xs[i]), ys[i] + t * (ys[i + 1] - ys[i])))
        out.append((ln, ax.transData.transform(pts) if pts else []))
    return out


def legend_clear(fig, ax, leg, pad=None):
    """True when the legend box lies inside the axes and no curve (points or
    segments) lies under it."""
    fig.canvas.draw()
    pad = guard_pad(fig) if pad is None else pad
    lb = leg.get_window_extent(); ab = ax.get_window_extent()
    if lb.x0 < ab.x0 or lb.y0 < ab.y0 or lb.x1 > ab.x1 or lb.y1 > ab.y1:
        return False
    for _, pts in curve_samples(ax):
        for x, y in pts:
            if lb.x0 - pad < x < lb.x1 + pad and lb.y0 - pad < y < lb.y1 + pad:
                return False
    return True


def legend_inside(fig, ax, ncol=1, accept=None, prefer_low=False):
    """Legend INSIDE the axes box (author's request, 2026-09-04), entries in
    the declared ORDER (9.2). Placement sweeps the corners first at a small
    y-axis headroom and then raises the headroom, so the legend never covers
    a curve point (9.4); one column keeps the box narrower than the axes.
    accept(leg), when given, may veto a clean legend (e.g. because no inset
    fits beside it), and the sweep then continues."""
    handles, labels = ax.get_legend_handles_labels()
    pairs = sorted(zip(handles, labels),
                   key=lambda hl: ORDER_LBL.index(hl[1]) if hl[1] in ORDER_LBL else len(ORDER_LBL))
    hs, ls = [h for h, _ in pairs], [l for _, l in pairs]
    y0, y1 = ax.get_ylim()
    if prefer_low:
        # author's request (Fig. 3, 2026-09-08): keep the legend low, in the
        # band between the collapsed token curve and the other curves, anchored
        # at the right edge; a two-column box is tried when one column is too
        # tall for the band, and a little footroom is added if needed
        for g in (0.0, 0.05, 0.10, 0.15):
            for nc in (1, 2):
                for yb in (0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.21, 0.24, 0.28, 0.32):
                    ax.set_ylim(y0 - g * (y1 - y0), y1)
                    leg = ax.legend(hs, ls, loc="lower right", bbox_to_anchor=(0.995, yb), ncol=nc)
                    if legend_clear(fig, ax, leg) and (accept is None or accept(leg)):
                        return leg
        ax.set_ylim(y0, y1)
    locs = ("upper left", "upper right", "lower right", "lower left", "center right", "center left")
    # one column first; a two-column box (shorter, wider) is the fallback when
    # the curves leave no band tall enough for one column (Fig. 6, six entries)
    for nc in ((ncol, 2) if ncol == 1 else (ncol,)):
        for f in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.0, 1.2, 1.4, 1.6, 1.9, 2.2):
            for loc in locs:
                ax.set_ylim(y0, y1 + f * (y1 - y0))
                leg = ax.legend(hs, ls, loc=loc, ncol=nc)
                if legend_clear(fig, ax, leg) and (accept is None or accept(leg)):
                    return leg
    ax.set_ylim(y0, y1)
    raise RuntimeError("no clean legend placement found")


legend_below = legend_inside     # every figure calls legend_below(fig, ax)


def guard_and_save(fig, ax, name, legend, inset=None):
    """Label-clipping, legend-in-canvas and curve-under-legend guards (12.9);
    an inset axes, when given, is tested against the curves like the legend."""
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
    pad = guard_pad(fig)
    boxes = [("legend", lb)] + ([("inset", inset.get_tightbbox())] if inset is not None else [])
    for what, bb in boxes:
        for ln, pts in curve_samples(ax):
            for x, y in pts:
                if bb.x0 - pad < x < bb.x1 + pad and bb.y0 - pad < y < bb.y1 + pad:
                    raise RuntimeError(f"{name}: curve '{ln.get_label()}' under {what} at ({x:.0f},{y:.0f})")
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

# ---- Fig. 3: load sweep on the N = 4 frame at the operating point ----------
fig, ax = new_figure()
n = draw(ax, [("psma", N_MAIN, None), ("masking_var", N_MAIN, None), ("oma_static", N_MAIN, None),
              ("deepma_offload", N_MAIN, None), ("todma", N_MAIN, None)],
         X_K, Y_PSNR, snr=SNR_OP)
# the load-4 sweep stops at K = N_MAIN: WH-OMA and token rows run to K = 8
for ln in ax.get_lines():
    xs, ys = list(ln.get_xdata()), list(ln.get_ydata())
    keep = [i for i, x in enumerate(xs) if x <= N_MAIN]
    ln.set_data([xs[i] for i in keep], [ys[i] for i in keep])
ax.relim(); ax.autoscale_view()
ax.set_xlabel("Active users $K$"); ax.set_ylabel("PSNR (dB)")
ax.set_xticks(range(1, N_MAIN + 1)); ax.grid(True, alpha=0.3)
from matplotlib.ticker import MultipleLocator
ax.yaxis.set_major_locator(MultipleLocator(2))
guard_and_save(fig, ax, "fig_load4.pdf", legend_inside(fig, ax, prefer_low=True))

# ---- Fig. 4: PSNR vs SNR with one active user (the full-load counterpart,
# fig_snr_k4.pdf, left the paper on 2026-09-09; its numbers stay in the text
# and in data/ser_eval.csv) ----------------------------------------------------
for name, K, series in (
        ("fig_snr_k1.pdf", 1, [("psma", N_MAIN, 1), ("masking_var", N_MAIN, 1), ("oma_static", N_MAIN, 1),
                               ("deepma_offload", N_MAIN, 1), ("todma", N_MAIN, 1)]),):
    # Figs. 3-5 and Table VI quote ONE evaluation run of the N = 4 DeepMA pairs
    # (scheme deepma_offload, K = 1..4); the separate N = K run of the same
    # checkpoint differs from it only by the fading draws (0.5 dB at 10 dB).
    fig, ax = new_figure()
    draw(ax, series, X_SNR, Y_PSNR)
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("PSNR (dB)"); ax.grid(True, alpha=0.3)
    # (an inset with the difference to Static OMA was tried on 2026-09-08 and
    # removed at the author's request; the differences are quoted in the text)
    guard_and_save(fig, ax, name, legend_below(fig, ax))

# ---- Fig. 6 (Sec. VI-C): per-dimension power of the trained N = 4 code when it
# is assigned to blocks (Q = I) at K = 1, 2, 4 versus WH spreading. Read from
# data/verify_code_order.csv (per-position variance of the code, normalized to
# mean 1, measured by code/verify_code_order.py on the 200 evaluation images).
CODE_ORDER = os.path.join(ROOT, "data", "verify_code_order.csv")
if os.path.exists(CODE_ORDER):
    crow = sorted([r for r in csv.DictReader(open(CODE_ORDER)) if r["model"] == "psma_u4_psma"],
                  key=lambda r: int(r["position"]))
    vnorm = [float(r["power_share"]) * L for r in crow]   # power per position, sums to L
    fig, ax = new_figure()
    # triples used by no other figure (9.1): three blues, markers P, X, *
    pw_style = {1: dict(color="#08306b", marker="P", ls="--"), 2: dict(color="#2171b5", marker="X", ls="--"),
                4: dict(color="#6baed6", marker="*", ls="--")}
    for K in (1, 2, 4):
        b = L // K
        prof = vnorm[:b] * K                                   # user j's prefix fills block j
        m = sum(prof) / L
        ax.plot(range(1, L + 1), [x / m for x in prof], label="Orthogonal partition, $K{=}%d$" % K, **pw_style[K])
    ax.plot([1, L], [1.0, 1.0], label="PSMA, all $K$", color=STYLE["psma"]["color"], ls="-", lw=2.2, marker=None)
    ax.set_xlabel("Embedding dimension $i$"); ax.set_ylabel("Relative power")
    ax.set_xticks(range(1, L + 1)); ax.grid(True, alpha=0.3)
    guard_and_save(fig, ax, "fig_power.pdf", legend_inside(fig, ax))

# ---- Fig. 7: one model across eight loads --------------------------------------
fig, ax = new_figure()
# The figures stop at K = L (author's decision, 2026-09-04): the K = 9..16
# tight-frame rows (scheme psma_tf) stay in data/ser_eval.csv as author
# material and are not drawn.
KMAX = L
d = sel("psma", L, None, SNR_OP)
if d:
    ax.plot([r["active"] for r in d], [r["psnr"] for r in d], label=LBL["psma8"], **STYLE["psma8"])
d = sel("masking_var", L, None, SNR_OP)
if d:
    ax.plot([r["active"] for r in d], [r["psnr"] for r in d], label=LBL["masking_var8"], **STYLE["masking_var8"])
draw(ax, [("todma", N_MAIN, None)], X_K, Y_PSNR, snr=SNR_OP)
# DeepMA held fixed at its N = 8 training load and evaluated at K = 1..8, the
# fixed-model counterpart of the PSMA and learned-mask curves (author's
# decision, 2026-09-04).
d = sel("deepma_offload", L, None, SNR_OP)
if d:
    ax.plot([r["active"] for r in d], [r["psnr"] for r in d], label=LBL["deepma8"], **STYLE["deepma8"])
# the Static OMA head trained for N = 8 (scheme oma, N = K = 8) is flat at every
# K by Proposition 1, like the N = 4 head in Fig. 3 (disclosure, 9.6; 2026-09-09)
d = sel("oma", L, L, SNR_OP)
if d:
    ax.plot(range(1, KMAX + 1), [d[0]["psnr"]] * KMAX, label=LBL["oma_static8"], markevery=(1, 2), **STYLE["oma_static8"])
ax.set_xlabel("Active users $K$"); ax.set_ylabel("PSNR (dB)")
ax.set_xticks(range(1, KMAX + 1)); ax.grid(True, alpha=0.3)
guard_and_save(fig, ax, "fig_load8.pdf", legend_below(fig, ax))

# ---- Fig. 8: semantic throughput T(K) = sum over served users of (1 - SER) -----
# A scheme provisioned for N serves min(K, N): static OMA and DeepMA (N = 4)
# stop at their K = N value (the author's convention, 2026-09-03).
fig, ax = new_figure()
drawn = 0
# the N = 4 designs stop at four (cap N_MAIN); every other chain runs to K = L.
# The N = 8 DeepMA pairs (Fig. 7's curve, Table VI's row) are drawn too, so the
# prose of Sec. VI-E names only curves the reader can see (2026-09-08).
for key, scheme, designed, cap in (("psma8", "psma", L, None), ("masking_var8", "masking_var", L, None),
                                   ("oma_static", "oma_static", N_MAIN, N_MAIN),
                                   ("deepma_offload", "deepma_offload", N_MAIN, N_MAIN),
                                   ("deepma8", "deepma_offload", L, None),
                                   ("oma_static8", "oma", L, None),
                                   ("todma", "todma", N_MAIN, None)):
    pts = []
    for K in range(1, KMAX + 1):
        Ke = min(K, cap) if cap else K
        # the N = 8 Static OMA head is measured at N = K = 8 only and is flat by Prop. 1
        d = sel(scheme, designed, L if key == "oma_static8" else Ke, SNR_OP)
        if d:
            pts.append((K, Ke * (1.0 - d[0]["ser"])))
    if len(pts) < 2:
        print("skipping (no data yet): throughput,", key); continue
    # the two capped curves coincide from K = 4 on: stagger their markers (9.4)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], label=LBL["oma_static4"] if key == "oma_static" else LBL[key],
            markevery=(drawn % 2, 2) if scheme in ("oma_static", "deepma_offload") else None, **STYLE[key]); drawn += 1
ax.set_xlabel("Active users $K$"); ax.set_ylabel(r"$\Theta(K)$ (images/frame)")
ax.set_xticks(range(1, KMAX + 1)); ax.grid(True, alpha=0.3)
y0, y1 = ax.get_ylim(); ax.set_ylim(0.0, y1)
guard_and_save(fig, ax, "fig_throughput.pdf", legend_below(fig, ax))

# ---- Table VI rows (PSNR at 10 and 20 dB, K = 1, 2, 3, 4, 6, 8) -----------------------
COLS = (1, 2, 3, 4, 6, 8)


def row(label, scheme, designed=None, nk=False):
    cells = []
    for K in COLS:
        if nk:
            d = [r for r in rows if r["scheme"] == scheme and r["designed"] == r["active"] == K and r["snr"] == snr]
        else:
            d = sel(scheme, designed, K, snr)
        cells.append(str(Decimal(str(d[0]["psnr"])).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)) if d else "---")
    return "%s & %s \\\\" % (label, " & ".join(cells))


for snr in (SNR_OP, 20):
    print("\n%% PSNR rows at %d dB, K = 1,2,3,4,6,8 (the load table left the paper on 2026-09-04; kept for the README)" % snr)
    print(row(LBL["psma4"], "psma", N_MAIN))
    print(row(LBL["psma8"], "psma", L))
    print(row("Learned masks, $N{=}4$", "masking_var", N_MAIN))
    print(row("Learned masks, $N{=}8$", "masking_var", L))
    print(row(LBL["todma"], "todma", N_MAIN))
    print(row(LBL["deepma8"], "deepma_offload", L))
    print(row(LBL["deepma_offload"], "deepma_offload", N_MAIN))
    print(row(LBL["oma_static"], "oma_static", N_MAIN))


def ser_row(label, scheme, designed, snr=SNR_OP):
    cells = []
    for K in range(1, L + 1):
        d = sel(scheme, designed, K, snr)
        cells.append(str(Decimal(str(d[0]["ser"])).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)) if d else "---")
    return "%s & %s \\\\" % (label, " & ".join(cells))


print("\n%% Table VI SER rows at %d dB, K = 1..8" % SNR_OP)
for label, scheme, designed in ((LBL["psma4"], "psma", N_MAIN), (LBL["psma8"], "psma", L),
                                ("Learned masks, $N{=}8$", "masking_var", L),
                                (LBL["todma"], "todma", N_MAIN), (LBL["deepma8"], "deepma_offload", L),
                                (LBL["deepma_offload"], "deepma_offload", N_MAIN), (LBL["oma_static"], "oma_static", N_MAIN)):
    print(ser_row(label, scheme, designed))
# ---- Fig. 2 (Sec. V-B): trained masks of the mask-based chain, per-user squared
# mask profile and overlap matrix (data/masks.csv, dumped by dump_artifacts.py).
# Own canvas (5.0 x 2.05 in, included at 0.48 columnwidth, print scale 0.336):
# authored 18 pt labels, ticks and cell values print at 6.0 pt on a panel wide
# enough that neighboring cells never touch (12.9).
MASKS = os.path.join(ROOT, "data", "masks.csv")
if os.path.exists(MASKS):
    mrows = list(csv.DictReader(open(MASKS)))
    U = len(mrows)
    LM = len([k for k in mrows[0] if k.startswith("m2_")])
    M2 = [[float(mrows[u][f"m2_{i}"]) for i in range(LM)] for u in range(U)]
    beta = [[sum(M2[u][i] * M2[v][i] for i in range(LM)) / LM for v in range(U)]
            for u in range(U)]
    fig, (axm, axb) = plt.subplots(
        1, 2, figsize=(5.0, 2.05), gridspec_kw={"width_ratios": [LM, U + 6.0]})
    for ax_ in (axm, axb):
        ax_.tick_params(labelsize=18.0)
    im0 = axm.imshow(M2, cmap="Oranges", aspect="auto", vmin=0)
    axm.set_xlabel("Dimension $i$", fontsize=18.0); axm.set_ylabel("User $u$", fontsize=18.0)
    axm.set_xticks(range(0, LM, 2)); axm.set_yticks(range(U))
    axm.set_yticklabels([str(u + 1) for u in range(U)])
    axm.set_xticklabels([str(i + 1) for i in range(0, LM, 2)])
    im1 = axb.imshow(beta, cmap="Oranges", aspect="auto", vmin=0)
    axb.set_xlabel("User $v$", fontsize=18.0); axb.set_ylabel("User $u$", fontsize=18.0)
    axb.set_xticks(range(U)); axb.set_yticks(range(U))
    axb.set_xticklabels([str(u + 1) for u in range(U)])
    axb.set_yticklabels([str(u + 1) for u in range(U)])
    axb.set_title(r"$\beta_{uv}$", fontsize=18.0)
    for u in range(U):
        for v in range(U):
            marks_cells = axb.text(v, u, f"{beta[u][v]:.1f}", ha="center", va="center", fontsize=18.0,
                                   color="white" if beta[u][v] >= 1.6 else "black")
    cb = fig.colorbar(im0, ax=axm, fraction=0.046, pad=0.04)
    cb.ax.set_title("$m_u(i)^2$", fontsize=18.0, pad=3)
    cb.ax.tick_params(labelsize=18.0)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.82, bottom=0.315, wspace=0.55)
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
    # the overlap cells are labels too: no two cell values may touch (12.9)
    for t in axb.texts:
        marks.append(("cell " + t.get_text(), t.get_window_extent()))
    for i in range(len(marks)):
        for j in range(i + 1, len(marks)):
            if marks[i][1].overlaps(marks[j][1]):
                raise RuntimeError("fig_masks: labels '%s' and '%s' overlap" % (marks[i][0], marks[j][0]))
    fig.savefig(os.path.join(FIG, "fig_masks.pdf")); plt.close(fig)
    print("wrote fig_masks.pdf")

# ---- Fig. 5 (Sec. VI-C): quality of the progressive code versus prefix length,
# one user alone on the frame (data/prefix_eval.csv from code/prefix_eval.py)
PREFIX = os.path.join(ROOT, "data", "prefix_eval.csv")
if os.path.exists(PREFIX):
    prow = list(csv.DictReader(open(PREFIX)))
    fig, ax = new_figure()
    # (legend key, model column, provisioned N): the two PSMA models trained
    # with the per-prefix term over every length, the earlier models whose
    # term visited the powers of two only, and the prototype without the term
    for key, model, d in (("psma4", "psma", 4), ("psma8", "psma", 8),
                          ("psma_pow2", "psma_pow2", 4), ("psma8_pow2", "psma_pow2", 8),
                          ("prog_v1", "prog_v1", 8)):
        pts = sorted([(int(r["prefix"]), float(r["psnr"])) for r in prow
                      if r.get("model", "psma") == model and int(r["designed"]) == d and int(r["snr"]) == SNR_OP])
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], label=LBL[key], **STYLE[key])
        else:
            print("skipping (no data yet): prefix", model, d)
    # conventional reference in every experiment (9.6): Static OMA with one
    # user at the operating point, read from ser_eval.csv (its two-dimension
    # head at unit power per symbol is the level the b = 2 prefix competes with)
    ref = sel("oma_static", N_MAIN, 1, SNR_OP)
    if ref:
        # reference line named in the caption, not the legend (12.9: a redundant
        # reference-line entry moves to the caption rather than shrinking the
        # common legend size); six entries would not fit inside these axes
        ax.plot([1, L], [ref[0]["psnr"]] * 2, label="_nolegend_", color=STYLE["oma_static"]["color"],
                ls=":", lw=1.8, marker=None)
    ax.set_xlabel("Prefix length $b$ (symbols per token)"); ax.set_ylabel("PSNR (dB)")
    ax.set_xticks(range(1, L + 1)); ax.grid(True, alpha=0.3)
    guard_and_save(fig, ax, "fig_prefix.pdf", legend_below(fig, ax))

# ---- Fig. 9 (Sec. VI-F): reconstructions of one image by every scheme on the
# N = 4 frame at K = 1 and K = 4, assembled from the panels that
# code/visual_eval.py stored under data/visual/ (PNG per scheme and load, PSNR
# per panel in visual_psnr.csv), so the manuscript figure is regenerated from
# data/ alone. Authored 12.30 x 4.56 in, included at 0.44\textwidth (3.16 in,
# print scale 0.256): the 23.5 pt headers print at 6.0 pt (standard 9.8).
VIS = os.path.join(ROOT, "data", "visual")
if os.path.exists(os.path.join(VIS, "visual_psnr.csv")):
    from PIL import Image
    IDX = 7
    vp = {(int(r["image"]), r["panel"]): float(r["psnr_db"])
          for r in csv.DictReader(open(os.path.join(VIS, "visual_psnr.csv")))}
    # the label text is verbatim (7.2); the long ones break onto two lines so
    # that neighboring headers never touch at the 19 pt header size
    COLS = [("Original", None, None), ("PSMA", "psma", "PSMA"),                 # declared ORDER
            ("Learned\nmasks", "learned_masks", "Learned masks"),
            ("Token\nsignatures", "token_signatures", "Token signatures"),
            ("DeepMA,\n$N{=}4$", "deepma_n4", "DeepMA N=4"),
            ("Static\nOMA", "static_oma_any_k", "Static OMA, any K")]
    PANEL_W, ROW_H, HEADER_PT = 2.05, 2.28, 14.0
    figv, axes = plt.subplots(2, len(COLS), figsize=(PANEL_W * len(COLS), ROW_H * 2))
    for r, K in enumerate((1, 4)):
        for c, (title, slug, key) in enumerate(COLS):
            ax = axes[r][c]
            if slug is None:
                img = Image.open(os.path.join(VIS, "img%d" % IDX, "original.png"))
                head = "Original" if r == 0 else ""
                # the load of each row is written once, on the Original panel
                ax.set_ylabel("$K{=}%d$" % K, fontsize=HEADER_PT, labelpad=4)
            else:
                fname = slug + ".png" if slug == "static_oma_any_k" else "%s_k%d.png" % (slug, K)
                img = Image.open(os.path.join(VIS, "img%d" % IDX, fname))
                v = vp[(IDX, key if slug == "static_oma_any_k" else "%s, K=%d" % (key, K))]
                # half-up rounding (12.15), never the half-even of "%.1f"
                v1 = Decimal(str(v)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
                head = (title + "\n" if r == 0 else "") + "%s dB" % v1
            ax.imshow(img); ax.set_title(head, fontsize=HEADER_PT)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
    figv.subplots_adjust(left=0.055, right=0.99, top=0.745, bottom=0.01, wspace=0.07, hspace=0.32)
    # header guard (12.9): no two panel headers may overlap, none may leave the canvas
    figv.canvas.draw()
    fw, fh = figv.get_size_inches() * figv.dpi
    hb = [(a.title.get_text(), a.title.get_window_extent()) for row_ in axes for a in row_ if a.title.get_text()]
    hb += [(a.yaxis.label.get_text(), a.yaxis.label.get_window_extent()) for row_ in axes for a in row_ if a.yaxis.label.get_text()]
    for t, bb in hb:
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > fw + 0.5 or bb.y1 > fh + 0.5:
            raise RuntimeError(f"fig_visual: header '{t}' clipped {bb}")
    for i in range(len(hb)):
        for j in range(i + 1, len(hb)):
            if hb[i][1].overlaps(hb[j][1]):
                raise RuntimeError("fig_visual: headers '%s' and '%s' overlap" % (hb[i][0], hb[j][0]))
    figv.savefig(os.path.join(FIG, "fig_visual.pdf")); plt.close(figv)
    print("wrote fig_visual.pdf")

print("done")
