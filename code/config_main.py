"""The ONE definition of the main experimental configuration.

Writing standard 7.9: every stage, plotting function and checker must read the
dimension, population, vocabulary and seeds from here rather than keeping its
own copy. A private copy silently stays behind when the configuration moves,
and the paper then reports two different systems as one.

Naming note, and a trap that already bit once. The manuscript's symbol ``L``
is the number of transmitted REAL dimensions per token. In the ``swinsc``
library that quantity is ``cfg.l_e``, derived as ``beta * l_s``. The SAME
``l_e`` can be reached by more than one ``(l_s, beta)`` pair, and the pair is
NOT a free choice: it changes the encoder projection, so a model trained at
``(4, 2)`` is a different network from one trained at ``(8, 1)`` even though
both transmit eight reals.

The values below are read back from the stored ``config.json`` of the trained
checkpoints on the GPU host, not chosen. The main L=8 chain was trained at
``l_s=8, beta=1``; the dimension-rich L=32 chain at ``l_s=16, beta=2``. An
earlier version of this file asserted ``BETA=2`` for the main chain, which
would have made ``swin_train.py`` retrain a network that no result in the
paper came from.

    from config_main import MAIN, l_s, cbr
"""

MAIN = {
    # --- shared frame -------------------------------------------------------
    "L": 8,                       # transmitted real dimensions per token
    "BETA": 1,                    # expansion factor of the main chain; l_s = L // BETA
    "L_RICH": 32,                 # dimension-rich variant of Section VI-B
    "RICH_L_S": 16,               # ... trained as l_s=16 with RICH_BETA=2
    "RICH_BETA": 2,
    "STAGES": 2,                  # Swin stages -> token is 4x4 pixels
    "DIMS": (96, 192),            # Swin stage widths, printed in Table IV
    "CHANNEL": "rayleigh",        # Rayleigh block fading with perfect-CSI ZF

    # --- population ---------------------------------------------------------
    "N": (4, 8),                  # provisioned populations of the PSMA manuscript: N[0] is the main frame (Figs. 2-4), N[1] = L the eight-load frame

    # --- source -------------------------------------------------------------
    "DATASET": "imagenette",
    "CROP": 128,                  # center-crop side in pixels
    "VAL_IMAGES": 200,            # validation images per user per point
    "EVAL_REPS": 5,               # independent fading draws per image (common random
                                  # numbers across schemes; author request 2026-09-04)

    # --- training -----------------------------------------------------------
    "EPOCHS": (20, 20),           # per-model budget; all compared chains ran 20
    "BATCH": 24,                  # per user
    "LR": 3e-4,
    "WEIGHT_DECAY": 1e-4,         # AdamW decoupled decay
    "GRAD_CLIP": 1.0,             # global-norm clipping, disclosed in Table IV
    "SNR_TRAIN": (0.0, 20.0),     # dB, uniform

    # --- evaluation ---------------------------------------------------------
    "SNR_GRID": (-5, 0, 5, 10, 15, 20),   # dB
    "SNR_OP": 10,                 # dB, the operating point of the load sweeps
    "VOCAB": 256,                 # ToDMA codebook / signature dictionary size

    # --- seeds --------------------------------------------------------------
    "SEED": 0,                    # models, data order, evaluation
    "SIGNATURE_SEED": 2026,       # ToDMA signature dictionary and panel channel
}


def workspace():
    """Root under which checkpoints, the dataset and logs live.

    Defaults to the repository root, the parent of this file's directory, so a
    fresh clone runs with no configuration at all. Set ``SWINSC_ROOT`` to point
    the scripts at a separate scratch tree instead; the runs behind the paper
    used ``SWINSC_ROOT=$HOME/ViT`` on the GPU host. Before this existed the
    defaults named that private tree directly, which meant the released package
    could not be run as its README documented.
    """
    import os
    return os.environ.get("SWINSC_ROOT") or os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))


def wpath(*parts):
    """A path inside :func:`workspace`."""
    import os
    return os.path.join(workspace(), *parts)


def device():
    """The ONE device flag (standard 2.3): CUDA when available, else CPU.

    Six scripts used to carry their own copy of this line. They are textually
    identical today, which is exactly how a drift starts.
    """
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"

# Pixels per token side: patch size 2 followed by (STAGES - 1) merges.
TOKEN_PX = 2 * 2 ** (MAIN["STAGES"] - 1)
# Source values carried by one token (3 color planes).
TOKEN_VALUES = TOKEN_PX * TOKEN_PX * 3


def split(L=None):
    """The (l_s, beta) pair the trained checkpoints actually carry for this L.

    Returns the pair rather than l_s alone, because beta differs between the
    two chains and a caller that assumes one global beta rebuilds the wrong
    encoder. Raises for any L the paper did not train, so a typo fails loudly
    instead of silently training a third architecture.
    """
    L = MAIN["L"] if L is None else L
    if L == MAIN["L"]:
        return L // MAIN["BETA"], MAIN["BETA"]
    if L == MAIN["L_RICH"]:
        return MAIN["RICH_L_S"], MAIN["RICH_BETA"]
    raise ValueError("no trained checkpoint uses L=%r (have %r and %r)"
                     % (L, MAIN["L"], MAIN["L_RICH"]))


def l_s(L=None):
    """Library-side embedding width for a manuscript-side dimension count."""
    return split(L)[0]


def beta(L=None):
    """Library-side expansion factor for a manuscript-side dimension count."""
    return split(L)[1]


def cbr(L=None):
    """Channel bandwidth ratio, (L/2) complex symbols per TOKEN_VALUES source values."""
    L = MAIN["L"] if L is None else L
    return L / 2.0 / TOKEN_VALUES


def block(L=None, N=None):
    """Per-user OMA block size B = L / N."""
    L = MAIN["L"] if L is None else L
    N = MAIN["N"][0] if N is None else N
    return L // N


def epochs_label():
    lo, hi = MAIN["EPOCHS"]
    return str(lo) if lo == hi else "%d-%d" % (lo, hi)


def snr_grid_step():
    g = MAIN["SNR_GRID"]
    steps = {g[i + 1] - g[i] for i in range(len(g) - 1)}
    if len(steps) != 1:
        raise ValueError("SNR_GRID is not uniformly spaced: %r" % (g,))
    return steps.pop()


#: Values that Table IV of the manuscript prints. `check_texhealth.py` asserts
#: that each string below still occurs in main.tex, so a configuration change
#: fails the pre-build gate instead of drifting into a stale table.
def table_iv_expectations():
    L = MAIN["L"]
    den = int(round(1.0 / cbr(L)))
    return {
        "token grid / dims per token": "$1024$ / $%d$" % L,
        "CBR": "$1/%d$" % den,
        "provisioned users": ", ".join(str(n) for n in MAIN["N"]),
        "SNR grid": "$%d$ to $%d$\\,dB, step $%d$"
                    % (MAIN["SNR_GRID"][0], MAIN["SNR_GRID"][-1], snr_grid_step()),
        "learning rate": "$%g{\\times}10^{-4}$" % round(MAIN["LR"] * 1e4, 6),
        "epochs / batch": "%s / %d" % (epochs_label(), MAIN["BATCH"]),
        "vocabulary": "%d /" % MAIN["VOCAB"],
        "seed": "%d (models), %d (signatures)" % (MAIN["SEED"], MAIN["SIGNATURE_SEED"]),
        "gradient clipping": "$%s$ \\\\" % ("%g" % MAIN["GRAD_CLIP"] if MAIN["GRAD_CLIP"] != 1.0 else "1.0"),
        "validation images": "%d per point, %d fading draws each" % (MAIN["VAL_IMAGES"], MAIN["EVAL_REPS"]),
        "crop": "$%d^2$ center crop" % MAIN["CROP"],
        "encoder": "Swin, %d stages, widths (%s)"
                   % (MAIN["STAGES"], ", ".join(str(d) for d in MAIN["DIMS"])),
    }


if __name__ == "__main__":
    for L in (MAIN["L"], MAIN["L_RICH"]):
        s, b = split(L)
        print("L=%d (l_s=%d, beta=%d)  CBR=1/%d  token=%dx%d px (%d values)"
              % (L, s, b, int(round(1 / cbr(L))), TOKEN_PX, TOKEN_PX, TOKEN_VALUES))
    print("N=%r  B at N=%d is %d" % (MAIN["N"], MAIN["N"][0], block()))
    for k, v in sorted(table_iv_expectations().items()):
        print("  Table IV | %-28s %s" % (k, v))
