# -*- coding: utf-8 -*-
"""Merge the raw evaluation files into the ONE grid plot_results.py reads.

    python code/merge_eval.py

data/tsp_eval.csv is written whole by tsp_eval.py (the N=2,4 main grid) and
then has to absorb the rows the companion evaluations produce:
  tsp_eval2_raw.csv   N=2 variable-load model and the L=32 dimension-rich set
  tsp_eval3_raw.csv   deep-overload sweep N=6,8 and token signatures at K=6,8
  tsp_eval3b_raw.csv  the N=4 fixed-load model below its training load
The merge is keyed on (scheme, designed, active, snr); a raw row replaces an
existing row with the same key, so the script is idempotent and a re-run of
any single evaluation updates only its own rows. Before this existed the
merge was done by hand, and the 2026-09-02 regeneration silently lost the
eval2 rows for a while.
"""
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
MAIN = os.path.join(DATA, "tsp_eval.csv")
RAW = ["tsp_eval2_raw.csv", "tsp_eval3_raw.csv", "tsp_eval3b_raw.csv"]
COLS = ["scheme", "designed", "active", "snr", "psnr"]


def key(r):
    return (r["scheme"], str(int(r["designed"])), str(int(r["active"])), str(int(float(r["snr"]))))


rows = {key(r): r for r in csv.DictReader(open(MAIN))}
n0 = len(rows)
for name in RAW:
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        print("  (absent)", name); continue
    k = 0
    for r in csv.DictReader(open(p)):
        rows[key(r)] = r; k += 1
    print("  merged %3d rows from %s" % (k, name))
with open(MAIN, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
    for r in sorted(rows.values(), key=lambda r: (r["scheme"], int(r["designed"]), int(r["active"]), int(float(r["snr"])))):
        w.writerow({c: r[c] for c in COLS})
print("tsp_eval.csv: %d -> %d rows" % (n0, len(rows)))
