"""Re-solve a sample of experiment C windows with tighter solver tolerances.

Reports lb - obj, where lb is the SDP bound of the window polynomial and obj the value of that same
polynomial at the stored SOS point, for CLARABEL tolerances 1e-7 (the default used in experiment C) and 1e-9.
Output: results/bound_tolerance.csv. Run with PYTHONPATH=src.
"""
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import exp_c_real_data as C  # noqa: E402
from sos_portfolio import build_portfolio_relaxation, empirical as em  # noqa: E402

TOLS = (1e-7, 1e-9)
rets, uni = C.load()
sector_of = dict(zip(uni.ticker, pd.factorize(uni.sector)[0]))
rows = {json.loads(l)["i"]: json.loads(l) for l in open(os.path.join(ROOT, "results/exp_c_alloc.jsonl"))}
idxs = C.rebalance_indices(rets, C.EVAL_START)
cols = list(rets.columns)


def run(U, i, w4):
    tick = C.U10 if U == "U10" else cols
    Rw = rets.values[i - C.WINDOW:i][:, [cols.index(t) for t in tick]]
    sec = np.array([sector_of[t] for t in tick])
    n, dense = len(tick), U == "U10"
    Rc = em.prepare_window(Rw)
    blocks = em.blocks_of(n, None if dense else sec)
    f = em.scale_polynomial(em.polynomial(Rc, blocks, em.lambdas(Rc, blocks, (1, 1, w4))), C.CAP)
    key = ("Poly_full_SOS_w4=" if dense else "Poly_sector_SOS_w4=") + f"{w4:g}"
    obj = float(f(np.array(rows[i][U][key]["w"]) / C.CAP))
    out = {"universe": U, "i": i, "w4": w4, "obj": obj}
    for tol in TOLS:
        pr = build_portfolio_relaxation(f, None if dense else blocks, 1 / C.CAP, 1 / C.CAP, 2,
                                        "dense" if dense else "sparse", budget="star" if len(blocks) <= 12 else "chain")
        r = pr.solve("CLARABEL", max_threads=1, tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol)
        out[f"excess_tol{tol:g}"] = (r["lower_bound"] - obj) / max(1.0, abs(obj))
        out[f"status_tol{tol:g}"] = r["status"]
    return out


if __name__ == "__main__":
    jobs = [("U37", i, 0.0) for i in (3545, 3588, 3630)]
    jobs += [("U10", idxs[k], w4) for k in range(0, 132, 11) for w4 in (0.0, 1.0, 3.0)]
    res = [run(*j) for j in jobs]
    with open(os.path.join(ROOT, "results/bound_tolerance.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)
