"""Study the instances of experiment A where the sparse bound is more than 1e-3 below the dense bound.

For each such instance: dense and sparse bounds at order 2 (star budget; the chain variant needs more than two
cliques and all these instances have two), the sparse bound with the per-clique ball constraint, and, for the
4-asset instances, both bounds at order 3. Output: results/sparse_gap.csv. Run with PYTHONPATH=src.
"""
import json
import os
import warnings

import pandas as pd

from sos_portfolio import ChordalExtension, SyntheticMarket, build_objective_from_market, build_portfolio_relaxation

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIGHT = dict(tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9)


def bound(f, cl, d, structure, **kw):
    r = build_portfolio_relaxation(f, cl, 0.95, 1.0, d, structure, **kw).solve("CLARABEL", max_threads=1, **TIGHT)
    return r["lower_bound"]


rows = []
for l in open(os.path.join(ROOT, "results/exp_a.jsonl")):
    r = json.loads(l)
    if r["lb_dense"] is None or r["lb_dense"] - r["lb_sparse"] <= 1e-3:
        continue
    p = {k: r["p_" + k] for k in ("skew_scale", "kurt_base", "impact_base")}
    m = SyntheticMarket.generate(r["n"], r["nc"], r["seed"], **p)
    f = build_objective_from_market(m)
    cl = ChordalExtension(m.adjacency_matrix()).build().maximal_cliques
    rec = {"key": r["key"], "n": r["n"], "cliques": len(cl)}
    rec["dense_d2"] = bound(f, None, 2, "dense")
    rec["sparse_d2"] = bound(f, cl, 2, "sparse")
    rec["sparse_d2_ball"] = bound(f, cl, 2, "sparse", ball=True)
    if r["n"] <= 4:
        rec["dense_d3"] = bound(f, None, 3, "dense")
        rec["sparse_d3"] = bound(f, cl, 3, "sparse")
    rows.append(rec)
    print(rec, flush=True)
df = pd.DataFrame(rows)
df["gap_d2"] = df.dense_d2 - df.sparse_d2
df["gap_d2_ball"] = df.dense_d2 - df.sparse_d2_ball
if "dense_d3" in df:
    df["gap_d3"] = df.dense_d3 - df.sparse_d3
df.to_csv(os.path.join(ROOT, "results/sparse_gap.csv"), index=False)
