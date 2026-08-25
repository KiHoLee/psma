"""Canonical replot script for paper 12 (TSP). Reads ONLY data/tsp_eval.csv,
writes every result figure to fig/ as vector PDF with one shared geometry.

Guards (writing-standard 12.7): after draw(), every axis label / tick label /
legend box is checked against the canvas, and every curve point is checked
against the legend box; violations raise.

    python code/plot_results.py
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "tsp_eval.csv")
FIG = os.path.join(ROOT, "fig")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 9.5, "axes.labelsize": 10, "legend.fontsize": 8.2,
    "lines.markersize": 5, "lines.linewidth": 1.4,
    "pdf.fonttype": 42, "figure.figsize": (4.0, 3.0),
})
AXRECT = [0.155, 0.15, 0.825, 0.83]   # identical axes rectangle for all result figures

# one shared label dictionary (7.2)
LBL = {
    "masking_fixed": "Proposed (fixed load)",
    "masking_var": "Proposed (variable load)",
    "oma": "Re-encoded OMA",
    "oma_static": "Static OMA",
    "todma": "ToDMA (genie)",
    "masking_fixed4off": "Proposed (fixed load, off-load)",
}
STYLE = {
    "masking_fixed": dict(color="#d95f02", marker="o", ls="-"),
    "masking_var": dict(color="#d95f02", marker="s", ls="--"),
    "oma": dict(color="#1b5d99", marker="^", ls="-"),
    "oma_static": dict(color="#1b5d99", marker="v", ls="--"),
    "todma": dict(color="#7570b3", marker="d", ls="-."),
    "masking_fixed4off": dict(color="#d95f02", marker="x", ls=":"),
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
    for top in (y1, y1 + 0.22 * (y1 - y0), y1 + 0.45 * (y1 - y0)):
        ax.set_ylim(y0, top)
        for loc, ncol in (("upper left", 1), ("upper right", 1), ("lower right", 1),
                          ("lower left", 1), ("upper center", 2)):
            leg = ax.legend(loc=loc, ncol=ncol, **kw)
            if legend_clear(fig, ax, leg):
                return leg
    raise RuntimeError("no clean legend placement found")


def snr_figure(name, series):
    fig = plt.figure(); ax = fig.add_axes(AXRECT)
    for scheme, designed, active in series:
        d = sel(scheme, designed, active)
        ax.plot([r["snr"] for r in d], [r["psnr"] for r in d],
                label=LBL[scheme], **STYLE[scheme])
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("PSNR (dB)")
    ax.grid(True, alpha=0.3)
    leg = place_legend(fig, ax)
    guard_and_save(fig, ax, name, leg)


# Fig: full load N=K=2
snr_figure("fig_snr_u2.pdf",
           [("masking_fixed", 2, 2), ("oma", 2, 2), ("todma", 4, 2)])
# Fig: overload N=K=4
snr_figure("fig_snr_u4.pdf",
           [("masking_fixed", 4, 4), ("masking_var", 4, 4), ("oma", 4, 4), ("todma", 4, 4)])

# Fig: underload PSNR vs K at 10 and 20 dB
fig = plt.figure(); ax = fig.add_axes(AXRECT)
for snr, lw, alpha in ((10, 1.4, 1.0), (20, 1.4, 0.55)):
    for scheme in ("masking_var", "masking_fixed4off", "oma_static", "todma"):
        d = [r for r in rows if r["scheme"] == scheme and r["snr"] == snr and r["designed"] == 4]
        d = sorted(d, key=lambda r: r["active"])
        st = dict(STYLE[scheme]); st["alpha"] = alpha
        ax.plot([r["active"] for r in d], [r["psnr"] for r in d],
                label=(LBL[scheme] if snr == 10 else None), lw=lw, **st)
ax.set_xlabel("Active users $K$ (provisioned $N=4$)"); ax.set_ylabel("PSNR (dB)")
ax.set_xticks([1, 2, 3, 4]); ax.grid(True, alpha=0.3)
leg = place_legend(fig, ax)
guard_and_save(fig, ax, "fig_underload.pdf", leg)

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
    }.items():
        f.write(f"- {tag}: {q(*args)}\n")
print("done")
