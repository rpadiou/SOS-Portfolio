"""Reproduce findings E1-E6 of the external review against the v1.0 code path.

Prints measured vs. reported values. E5 at n=15 is not run (the v1.0 dense
builder was reported to be killed for memory); only n=10 is timed.
"""
from __future__ import annotations

import json
import sys
import time
from math import comb

import numpy as np

from sos_portfolio import (
    LasserreRelaxation, SparseLasserreRelaxation, SyntheticMarket,
    build_constraints, build_constraints_from_market, build_objective,
    build_objective_from_market, scipy_optimize, analyze_local_minima,
)
from sos_portfolio.graph_sparsity import ChordalExtension, build_block_adjacency
from sos_portfolio.indexer import SparseIndexer

rows = []


def report(tag, what, measured, reported):
    rows.append((tag, what, measured, reported))
    print(f"{tag:3s} {what:55s} measured={measured}  reported={reported}")


def e1():
    f = build_objective()
    grid = np.linspace(0, 1, 401)
    lam = np.inf
    for x1 in grid:
        for x2 in grid:
            if 0.95 <= x1 + x2 <= 1.0:
                H = np.array([[0.08 - 1.8 * x1 + 12 * x1**2 + 2.9 * x2**2,
                               -0.04 + 5.8 * x1 * x2],
                              [-0.04 + 5.8 * x1 * x2,
                               0.18 + 3 * x2 + 16.2 * x2**2 + 2.9 * x1**2]])
                lam = min(lam, np.linalg.eigvalsh(H)[0])
    report("E1", "min Hessian eigenvalue on K", f"{lam:.4f}", "1.68")
    res = scipy_optimize(f, n_restarts=50, seed=0)
    nm = analyze_local_minima(res["all_results"])["n_distinct_minima"]
    report("E1", "distinct local minima (50 starts)", nm, 1)
    lb = LasserreRelaxation(f, build_constraints(), 2).build_and_solve("CLARABEL")["lower_bound"]
    report("E1", "f_ub / SOS lb", f"{res['f_opt']:.6f} / {lb:.6f}", "0.187371")


def solve_sparse(m, solver="CLARABEL", d=2):
    cl = ChordalExtension(m.adjacency_matrix()).build().maximal_cliques
    r = SparseLasserreRelaxation(build_objective_from_market(m),
                                 build_constraints_from_market(m), cl, d)
    return r, r.build_and_solve(solver)


def e2():
    for (n, nc, s), ref in {(4, 2, 0): 0.034319, (6, 2, 0): 0.018155, (6, 2, 1): 0.001774,
                            (6, 3, 2): 0.017634, (8, 2, 3): 0.005939}.items():
        m = SyntheticMarket.generate(n, nc, s)
        _, rs = solve_sparse(m)
        f = build_objective_from_market(m)
        ub = scipy_optimize(f, 30, 0)
        nm = analyze_local_minima(ub["all_results"])["n_distinct_minima"]
        report("E2", f"({n},{nc},{s}) sparse lb / UB / #min",
               f"{rs['lower_bound']:.6f} / {ub['f_opt']:.6f} / {nm}", ref)


def e3():
    cases = [((6, 2, 0, 2.0, 0.3, 0.05), -22.065857, -2.427636),
             ((6, 2, 1, 4.0, 0.3, 0.05), -174.279652, -5.399794),
             ((6, 3, 2, 6.0, 0.2, 0.05), 0.128596, 0.128596),
             ((8, 2, 3, 5.0, 0.3, 0.05), -173.606348, -6.242960)]
    for (n, nc, s, sk, ku, im), sp_ref, de_ref in cases:
        m = SyntheticMarket.generate(n, nc, s, skew_scale=sk, kurt_base=ku, impact_base=im)
        _, rs = solve_sparse(m)
        f = build_objective_from_market(m)
        rd = LasserreRelaxation(f, build_constraints_from_market(m), 2).build_and_solve("CLARABEL")
        report("E3", f"{(n, nc, s, sk)} sparse lb", f"{rs['lower_bound']:.6f}", sp_ref)
        report("E3", f"{(n, nc, s, sk)} dense lb", f"{rd['lower_bound']:.6f}", de_ref)


def e4():
    for n, nc in [(15, 3), (50, 10)]:
        p = n // nc
        cl = [list(range(c * p, (c + 1) * p)) for c in range(nc)]
        idx = SparseIndexer(n, 2, cl)
        s = comb(p + 2, 2)
        report("E4", f"n={n} sparse unique moments", idx.n_moments, {15: 376, 50: 1251}[n])
        report("E4", f"n={n} sparse PSD entries", nc * s * s, {15: 1323, 50: 4410}[n])
        report("E4", f"n={n} dense unique moments", comb(n + 4, 4), {15: 3876, 50: 316251}[n])
        report("E4", f"n={n} dense PSD entries", comb(n + 2, 2) ** 2, {15: 18496, 50: 1758276}[n])


def e5():
    m = SyntheticMarket.generate(10, 2, 3, skew_scale=5.0, kurt_base=0.3, impact_base=0.05)
    f = build_objective_from_market(m)
    t = time.perf_counter()
    rd = LasserreRelaxation(f, build_constraints_from_market(m), 2).build_and_solve("CLARABEL")
    report("E5", "dense n=10 time (s) / status / lb",
           f"{time.perf_counter() - t:.1f} / {rd['status']} / {rd['lower_bound']:.5f}",
           "11.6 / optimal_inaccurate / -6.79985")
    print("E5  dense n=15 NOT run (reported killed for memory)")


def e6():
    m = SyntheticMarket.generate(50, 10, 0)
    for solver, ref in [("CLARABEL", "lb=-0.005990 flat 10/10 1.85s"),
                        ("SCS", "lb=-0.006045 flat 0/10")]:
        t = time.perf_counter()
        r, res = solve_sparse(m, solver)
        fl = r.flat_extension_check()
        nflat = sum(c["converged"] for c in fl["per_clique"])
        report("E6", f"n=50 {solver}", f"lb={res['lower_bound']:.6f} flat {nflat}/10 "
               f"{time.perf_counter() - t:.2f}s status={res['status']}", ref)


if __name__ == "__main__":
    which = sys.argv[1:] or ["e1", "e2", "e3", "e4", "e5", "e6"]
    for w in which:
        globals()[w]()
    with open("results/review_findings_v1.json", "w") as fh:
        json.dump([dict(tag=a, what=b, measured=str(c), reported=str(d)) for a, b, c, d in rows], fh, indent=1)
