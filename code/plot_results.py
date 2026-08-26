"""Canonical replot script for paper 12 (TSP). Reads ONLY files under data/
(tsp_eval.csv, sinr_model.csv, masks.csv and the stored panels in visual/) and
writes every result figure to fig/ as vector PDF with one shared geometry.
Every population, operating point and axis label comes from code/config_main.py.

Guards (writing-standard 12.7): after draw(), every axis label / tick label /
legend box is checked against the canvas, and every curve point is checked
against the legend box; violations raise.

    python code/plot_results.py
"""
import csv
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_main import MAIN   # the ONE configuration (standard 7.9)

N_SMALL, N_MAIN = MAIN["N"][0], MAIN["N"][-1]
SNR_OP = MAIN["SNR_OP"]
LOADS = list(range(1, N_MAIN + 1))
XLABEL_LOAD = "Active users $K$ (provisioned $N=%d$)" % N_MAIN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "tsp_eval.csv")
FIG = os.path.join(ROOT, "fig")
os.makedirs(FIG, exist_ok=True)

# Standard 9.6: the smallest label must be legible at final column width, and
# the check is on the PRINTED size, not the authored one. The five result plots
# are authored on a 4.0 in canvas and included at 0.685\columnwidth. IEEEtran
# 10pt journal gives \columnwidth = 252 pt, so the canvas is scaled by
# 0.685 * 252 / 288 = 0.599 and an authored size prints at 0.6 of itself.
# The sizes below are chosen so that NOTHING prints below 6.5 pt:
#   ticks 12.0 -> 7.2 pt, axis labels 12.5 -> 7.5 pt, legend 11.0 -> 6.6 pt.
# If the include width changes, recompute this comment and these numbers.
PRINT_SCALE = 0.685 * 252.0 / 288.0
plt.rcParams.update({
    "font.size": 12.0, "axes.labelsize": 12.5, "legend.fontsize": 11.0,
    "lines.markersize": 5.8, "lines.linewidth": 1.6,
    "pdf.fonttype": 42, "figure.figsize": (4.0, 3.0),
    # A compact legend box, so a five-entry legend fits without stretching the
    # y axis to make room. The alternative was a smaller legend font, which 9.6
    # forbids at this include width. Only the whitespace shrinks, not the text.
    "legend.labelspacing": 0.25, "legend.handlelength": 1.5,
    "legend.handletextpad": 0.4, "legend.borderpad": 0.3,
    "legend.borderaxespad": 0.3, "legend.framealpha": 0.9,
})
# Identical axes rectangle for every result figure, at the 8:6 ratio item 9.1
# asks for: 0.800 * 4.0 = 3.200 in wide by 0.800 * 3.0 = 2.400 in tall.
AXRECT = [0.180, 0.180, 0.800, 0.800]

# one shared label dictionary (7.2)
LBL = {
    "masking_fixed": "Proposed (fixed load)",
    "masking_var": "Proposed (variable load)",
    "oma": "Re-encoded OMA",
    "oma_static": "Static OMA",
    "todma": "Token signatures (genie)",
    "masking_fixed4off": "Proposed (fixed load, off-load)",
    "masking_rich": "Proposed (fixed load)",
    "masking_var_rich": "Proposed (variable load)",
    "oma_rich": "Re-encoded OMA",
    "todma_rich": "Token signatures (genie)",
    # the same variable-load model at one active user, drawn beside its
    # full-load curve so the dimension-rich underload claim is readable
    "masking_var_rich_k1": "Proposed (variable load), $K{=}1$",
}
STYLE = {
    "masking_fixed": dict(color="#d95f02", marker="o", ls="-"),
    "masking_var": dict(color="#d95f02", marker="s", ls="--"),
    "oma": dict(color="#1b5d99", marker="^", ls="-"),
    "oma_static": dict(color="#1b5d99", marker="v", ls="--"),
    "todma": dict(color="#7570b3", marker="d", ls="-."),
    # the off-load curve is the SAME trained fixed-load model, so it carries the
    # same style as "masking_fixed"; the two never appear in one figure
    "masking_fixed4off": dict(color="#d95f02", marker="o", ls="-"),
    "masking_rich": dict(color="#d95f02", marker="o", ls="-"),
    "masking_var_rich": dict(color="#d95f02", marker="s", ls="--"),
    "oma_rich": dict(color="#1b5d99", marker="^", ls="-"),
    "todma_rich": dict(color="#7570b3", marker="d", ls="-."),
    # proposed-scheme hue, lighter and open-faced so it reads as the same
    # model at a lighter load rather than as a different scheme
    "masking_var_rich_k1": dict(color="#d95f02", marker="s", ls="--",
                                mfc="none", mew=1.4),
}

rows = list(csv.DictReader(open(DATA)))
for r in rows:
    r["designed"] = int(r["designed"]); r["active"] = int(r["active"])
    r["snr"] = int(r["snr"]); r["psnr"] = float(r["psnr"])


def sel(scheme, designed=None, active=None):
    out = [r for r in rows if r["scheme"] == scheme
           and (designed is None or r["designed"] == designed)
           and (active is None or r["active"] == active)]
    return sorted(out, key=lambda r: (r["snr"], r["active"]))


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
    if legend is not None:
        lb = legend.get_window_extent()
        if lb.x0 < 0 or lb.y0 < 0 or lb.x1 > fw or lb.y1 > fh:
            raise RuntimeError(f"{name}: legend leaves canvas")
        pad = 4  # marker radius + line width, display px
        for ln in ax.get_lines():
            xy = ax.transData.transform(list(zip(ln.get_xdata(), ln.get_ydata())))
            for x, y in xy:
                if lb.x0 - pad < x < lb.x1 + pad and lb.y0 - pad < y < lb.y1 + pad:
                    raise RuntimeError(f"{name}: curve '{ln.get_label()}' under legend at ({x:.0f},{y:.0f})")
    fig.savefig(os.path.join(FIG, name))
    print("wrote", name)


def legend_clear(fig, ax, leg, pad=4):
    """True when no curve point lies inside the legend box and the box is on canvas."""
    fig.canvas.draw()
    fw, fh = fig.get_size_inches() * fig.dpi
    lb = leg.get_window_extent()
    if lb.x0 < 0 or lb.y0 < 0 or lb.x1 > fw or lb.y1 > fh:
        return False
    for ln in ax.get_lines():
        for x, y in ax.transData.transform(list(zip(ln.get_xdata(), ln.get_ydata()))):
            if lb.x0 - pad < x < lb.x1 + pad and lb.y0 - pad < y < lb.y1 + pad:
                return False
    return True


def place_legend(fig, ax, **kw):
    """Sweep candidate placements on the EXACT final layout; raise ylim headroom if needed."""
    y0, y1 = ax.get_ylim()
    # Placement is the OUTER loop and headroom the inner one, so every result
    # figure lands in the same corner (9.1 uniform geometry) and a figure whose
    # curves reach into that corner is given room rather than a new location or
    # a smaller legend (12.7).
    steps = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
             0.40, 0.50, 0.60, 0.75, 0.90, 1.10, 1.30]
    locs = (("upper left", 1), ("upper right", 1), ("lower right", 1),
            ("lower left", 1), ("upper center", 1), ("upper center", 2))
    # Pass 1 sweeps LOCATION first at a small headroom budget, so every figure
    # whose upper-left corner is genuinely free lands there and the set reads as
    # one family.
    for loc, ncol in locs:
        for f in [s for s in steps if s <= 0.30]:
            ax.set_ylim(y0, y1 + f * (y1 - y0))
            leg = ax.legend(loc=loc, ncol=ncol, **kw)
            if legend_clear(fig, ax, leg):
                return leg
    # Pass 2 sweeps HEADROOM first. A modest extension in some other corner beats
    # a large one in the preferred corner, because stretching the axis squashes
    # the curves into the lower part of the box and costs more readability than
    # a moved legend does (12.7). Getting this order wrong pushed the
    # dimension-rich figure's y axis to 50 dB for data that spans 12 to 31.
    for f in steps:
        for loc, ncol in locs:
            ax.set_ylim(y0, y1 + f * (y1 - y0))
            leg = ax.legend(loc=loc, ncol=ncol, **kw)
            if legend_clear(fig, ax, leg):
                return leg
    ax.set_ylim(y0, y1)
    raise RuntimeError("no clean legend placement found")


def snr_figure(name, series):
    """series entries are (scheme, designed, active) or, when one scheme is
    drawn at two loads, (scheme, designed, active, style_key)."""
    fig = plt.figure(); ax = fig.add_axes(AXRECT)
    for k, entry in enumerate(series):
        scheme, designed, active = entry[:3]
        key = entry[3] if len(entry) > 3 else scheme
        d = sel(scheme, designed, active)
        # staggered markevery offsets so coinciding curves never stack markers (9.2)
        ax.plot([r["snr"] for r in d], [r["psnr"] for r in d],
                label=LBL[key], markevery=(k % 2, 2), **STYLE[key])
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("PSNR (dB)")
    ax.grid(True, alpha=0.3)
    leg = place_legend(fig, ax)
    guard_and_save(fig, ax, name, leg)


def available(series):
    """Keep only schemes whose rows exist, so the script runs before and
    after a new experiment's CSV rows are merged."""
    kept = [t for t in series if sel(*t[:3])]
    for t in series:
        if t not in kept:
            print("skipping (no data yet):", t)
    return kept


# Fig: full load N=K=2 (all four chains; ToDMA has no provisioned
# population, so its K=2 rows are stored under designed=4)
snr_figure("fig_snr_u2.pdf",
           available([("masking_fixed", N_SMALL, N_SMALL), ("masking_var", N_SMALL, N_SMALL),
                      ("oma", N_SMALL, N_SMALL), ("todma", N_MAIN, N_SMALL)]))
# Fig: overload N=K at the largest provisioned population
snr_figure("fig_snr_u4.pdf",
           [("masking_fixed", N_MAIN, N_MAIN), ("masking_var", N_MAIN, N_MAIN),
            ("oma", N_MAIN, N_MAIN), ("todma", N_MAIN, N_MAIN)])

# Fig: underload PSNR vs K at 10 dB (one curve per scheme, canonical styles;
# the 20 dB values are listed in the manuscript's load table instead)
fig = plt.figure(); ax = fig.add_axes(AXRECT)
for scheme in ("masking_var", "masking_fixed4off", "oma_static", "todma"):
    d = [r for r in rows if r["scheme"] == scheme
         and r["snr"] == SNR_OP and r["designed"] == N_MAIN]
    d = sorted(d, key=lambda r: r["active"])
    ax.plot([r["active"] for r in d], [r["psnr"] for r in d],
            label=LBL[scheme], **STYLE[scheme])
ax.set_xlabel(XLABEL_LOAD); ax.set_ylabel("PSNR (dB)")
ax.set_xticks(LOADS); ax.grid(True, alpha=0.3)
leg = place_legend(fig, ax)
guard_and_save(fig, ax, "fig_underload.pdf", leg)

# Fig: dimension-rich full load (L=32, N=K=2, all four chains)
snr_figure("fig_snr_rich.pdf",
           available([("masking_rich", N_SMALL, N_SMALL),
                      ("masking_var_rich", N_SMALL, N_SMALL),
                      # one active user: the best case for the masked chain,
                      # against an OMA curve Prop. 1 holds at every load
                      ("masking_var_rich", N_SMALL, 1, "masking_var_rich_k1"),
                      ("oma_rich", N_SMALL, N_SMALL),
                      ("todma_rich", N_SMALL, N_SMALL)]))

# Fig: SINR model (Prop. 2) with the trained masks, vs K
import csv as _csv
mrows = list(_csv.DictReader(open(os.path.join(ROOT, "data", "sinr_model.csv"))))
fig = plt.figure(); ax = fig.add_axes(AXRECT)
# sequential ramp in the proposed-scheme hue (#d95f02), light to dark with SNR
MCOL = {-5: "#fdae61", 0: "#e6550d", 10: "#a63603", 20: "#7f2704"}
MMK = {-5: "o", 0: "s", 10: "^", 20: "d"}
MLS = {-5: "-", 0: "--", 10: "-.", 20: ":"}
for snr in (-5, 0, 10, 20):
    d = sorted([r for r in mrows if int(r["snr"]) == snr], key=lambda r: int(r["active"]))
    # the minus sign must match the one matplotlib prints on the axis ticks
    # (U+2212); an ASCII hyphen in the legend prints visibly shorter beside it
    ax.plot([int(r["active"]) for r in d], [float(r["model_sinr_db"]) for r in d],
            color=MCOL[snr], marker=MMK[snr], ls=MLS[snr],
            label="Model, %s dB" % str(snr).replace("-", "−"))
# conventional-scheme reference (9.4): the interference-free orthogonal member
# of the same mask family, drawn at the operating point in the OMA colour
_d = sorted([r for r in mrows if int(r["snr"]) == SNR_OP], key=lambda r: int(r["active"]))
ax.plot([int(r["active"]) for r in _d], [float(r["orth_sinr_db"]) for r in _d],
        color="#1b5d99", marker="^", ls="--", label=f"Orthogonal, {SNR_OP} dB")
ax.set_xlabel(XLABEL_LOAD); ax.set_ylabel("SINR (dB)")
ax.set_xticks(LOADS); ax.grid(True, alpha=0.3)
leg = place_legend(fig, ax)
guard_and_save(fig, ax, "fig_sinr_model.pdf", leg)

# qualitative panels (Figs. 8-9): rendered by code/visual_eval.py on the GPU
# host into data/visual/; this step installs the stored panels into fig/ so
# one replot run refreshes every manuscript figure from data/ alone.
import shutil
for name in ("fig_visual_overload.pdf", "fig_visual_underload.pdf"):
    src = os.path.join(ROOT, "data", "visual", name)
    if os.path.exists(src):
        shutil.copyfile(src, os.path.join(FIG, name))
        print("installed", name)

# trained masks: per-user squared mask profile and overlap matrix (data/masks.csv)
MASKS = os.path.join(ROOT, "data", "masks.csv")
if os.path.exists(MASKS):
    mrows = list(csv.DictReader(open(MASKS)))
    U = len(mrows)
    L = len([k for k in mrows[0] if k.startswith("m2_")])
    M2 = [[float(mrows[u][f"m2_{i}"]) for i in range(L)] for u in range(U)]
    beta = [[sum(M2[u][i] * M2[v][i] for i in range(L)) / L for v in range(U)]
            for u in range(U)]
    fig, (axm, axb) = plt.subplots(
        1, 2, figsize=(4.0, 2.05), gridspec_kw={"width_ratios": [L, U + 1.5]})
    # same hue family as the proposed-scheme curves (#d95f02) in the result
    # figures, so the analysis figures read in the tone of the rest of the paper
    im0 = axm.imshow(M2, cmap="Oranges", aspect="auto", vmin=0)
    axm.set_xlabel("Dimension $i$"); axm.set_ylabel("User $u$")
    axm.set_xticks(range(L)); axm.set_yticks(range(U))
    axm.set_yticklabels([str(u + 1) for u in range(U)])
    axm.set_xticklabels([str(i + 1) for i in range(L)])
    # no panel title here: the colorbar below is this panel's value axis and
    # already names the quantity, so a title would repeat it
    im1 = axb.imshow(beta, cmap="Oranges", aspect="auto", vmin=0)
    axb.set_xlabel("User $v$"); axb.set_ylabel("User $u$")
    axb.set_xticks(range(U)); axb.set_yticks(range(U))
    axb.set_xticklabels([str(u + 1) for u in range(U)])
    axb.set_yticklabels([str(u + 1) for u in range(U)])
    axb.set_title(r"$\beta_{uv}$", fontsize=12.0)
    for u in range(U):
        for v in range(U):
            # 10.7 authored prints at 7.5 pt on this figure's 0.80\columnwidth
            # include width, the smallest text anywhere in the paper (9.6)
            axb.text(v, u, f"{beta[u][v]:.1f}", ha="center", va="center",
                     fontsize=10.7,
                     color="white" if beta[u][v] >= 1.6 else "black")
    # only the left panel carries a colorbar; the right panel prints its values,
    # so a second bar would cost cell width without adding information. The bar
    # is the value axis of the left panel and carries its own label (9.7).
    cb = fig.colorbar(im0, ax=axm, fraction=0.046, pad=0.04)
    # The label goes ABOVE the bar, not rotated beside it: a side label sits in
    # the gap between the panels and collides with the right panel's y label.
    cb.ax.set_title("$m_u^2(i)$", fontsize=12.0, pad=3)
    cb.ax.tick_params(labelsize=10.7)
    fig.subplots_adjust(left=0.135, right=0.97, top=0.855, bottom=0.285, wspace=0.55)
    # Guards (9.7, 12.7): the panels carry no legend, so the failure modes are a
    # label leaving the canvas and two labels landing on top of each other. The
    # second one bit once already, when the colorbar label was rotated into the
    # gap the right panel's y label occupies.
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
            (ta, ba), (tb, bb2) = marks[i], marks[j]
            if ba.overlaps(bb2):
                raise RuntimeError("fig_masks: labels '%s' and '%s' overlap"
                                   % (ta, tb))
    fig.savefig(os.path.join(FIG, "fig_masks.pdf"))
    plt.close(fig)
    print("wrote fig_masks.pdf")

# quoted-numbers digest for the manuscript
with open(os.path.join(ROOT, "data", "quoted_numbers.md"), "w") as f:
    def q(scheme, designed, active, snr):
        d = [r for r in rows if r["scheme"] == scheme and r["designed"] == designed
             and r["active"] == active and r["snr"] == snr]
        return d[0]["psnr"] if d else None
    f.write("# quoted numbers (regenerated by plot_results.py)\n")
    for tag, args in {
        "full2 mask": ("masking_fixed", 2, 2, 10), "full2 oma": ("oma", 2, 2, 10),
        "ov4 mask": ("masking_fixed", 4, 4, 10), "ov4 var": ("masking_var", 4, 4, 10),
        "ov4 oma": ("oma", 4, 4, 10), "ov4 oma20": ("oma", 4, 4, 20),
        "ov4 mask20": ("masking_fixed", 4, 4, 20),
        "ul k1 var": ("masking_var", 4, 1, 10), "ul k1 oma": ("oma_static", 4, 1, 10),
        "ul k4 var": ("masking_var", 4, 4, 10),
        "todma k2": ("todma", 4, 2, 10), "todma k4": ("todma", 4, 4, 10),
        "todma k1": ("todma", 4, 1, 10),
        "fixedoff k1": ("masking_fixed4off", 4, 1, 10), "fixedoff k2": ("masking_fixed4off", 4, 2, 10),
        "var k2": ("masking_var", 4, 2, 10), "var k3": ("masking_var", 4, 3, 10),
        "var k1 20": ("masking_var", 4, 1, 20), "static oma20": ("oma", 4, 4, 20),
        "full2 mask20": ("masking_fixed", 2, 2, 20), "full2 oma20": ("oma", 2, 2, 20),
        "todma k3": ("todma", 4, 3, 10),
        "u2 var": ("masking_var", 2, 2, 10), "u2 var20": ("masking_var", 2, 2, 20),
        "rich mask": ("masking_rich", 2, 2, 10), "rich mask20": ("masking_rich", 2, 2, 20),
        "rich var": ("masking_var_rich", 2, 2, 10), "rich var20": ("masking_var_rich", 2, 2, 20),
        "rich oma": ("oma_rich", 2, 2, 10), "rich oma20": ("oma_rich", 2, 2, 20),
        "rich todma": ("todma_rich", 2, 2, 10), "rich todma20": ("todma_rich", 2, 2, 20),
    }.items():
        f.write(f"- {tag}: {q(*args)}\n")
print("done")
