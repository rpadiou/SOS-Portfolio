"""Command-line entry point: solve one instance and report bounds, sizes and certificates.

    python main.py --mode demo
    python main.py --mode sparse --n-assets 15 --n-clusters 3 [--stress] [--kappa-radius 0.02]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from sos_portfolio import (ChordalExtension, MinimizerExtractor, SyntheticMarket, build_objective,
                           build_objective_from_market, build_portfolio_relaxation, certify,
                           make_nonconvex, scipy_optimize)
from sos_portfolio.local_solver import analyze_local_minima


def report(f, pr, n, b_lo, b_hi, solver, starts, robust=False):
    r = pr.solve(solver)
    sz = r["sizes"]
    print(f"solver={r['solver']} status={r['status']}{' (inaccurate)' if r['inaccurate'] else ''} "
          f"iterations={r['iterations']} build={r['build_time']:.2f}s solve={r['solve_time']:.2f}s")
    print(f"sizes: max PSD block={sz['max_block']}  PSD entries={sz['psd_entries']}  "
          f"unique moments={sz['unique_moments']}  PSD blocks={sz['n_psd_blocks']}")
    if r["lower_bound"] is None:
        print("no bound returned")
        return
    lb = r["lower_bound"]
    t = time.perf_counter()
    loc = scipy_optimize(f, starts, 0, b_lo, b_hi)
    tl = time.perf_counter() - t
    mins = analyze_local_minima(loc["results"])
    print(f"lower bound lambda_d = {lb:.8f}")
    print(f"local search: best f = {loc['f_opt']:.8f}  ({starts} starts, {mins['n_distinct_minima']} distinct KKT minima, {tl:.2f}s)")
    print(f"gap (best local - lambda_d) = {loc['f_opt'] - lb:.3e}")
    if robust:
        print("robust objective: flatness/certificate refer to the moment problem with the norm term; skipped")
        return
    ex = MinimizerExtractor(pr.relaxation, r["moments"], n)
    fl = ex.flatness()
    print(f"flat extension (rank tol {ex.rank_tol:g}): {fl['flat']}  "
          f"ranks per asset clique: {[(p['rank_d'], p['rank_d_minus_dK']) for p in fl['per_clique']]}")
    if r["solver"] != "CLARABEL":
        print("note: rank decisions are not reliable at this solver accuracy")
    pts = ex.extract()
    cand = ([] if pts["points"] is None else list(pts["points"])) + [ex.mean_point()]
    c = certify(f, lb, cand, b_lo, b_hi)
    print(f"a posteriori certificate: certified={c['certified']} gap={c['gap']}  (extraction: {pts['reason']})")
    if c["x_hat"] is not None and n <= 12:
        print("x_hat =", np.round(c["x_hat"], 4), " sum =", round(float(c["x_hat"].sum()), 6))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["demo", "sparse"], default="sparse")
    ap.add_argument("--n-assets", type=int, default=15)
    ap.add_argument("--n-clusters", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--structure", choices=["sparse", "dense"], default="sparse")
    ap.add_argument("--budget", choices=["star", "chain"], default="star")
    ap.add_argument("--solver", default="CLARABEL", choices=["CLARABEL", "SCS", "MOSEK"])
    ap.add_argument("--stress", action="store_true", help="non-convex stress instance (make_nonconvex)")
    ap.add_argument("--kappa-radius", type=float, default=0.0, help="Frobenius-ball radius on the kurtosis coefficients")
    ap.add_argument("--starts", type=int, default=50)
    a = ap.parse_args()

    if a.mode == "demo":
        f = build_objective()
        print("2-asset benchmark (strictly convex on K: validation case)")
        pr = build_portfolio_relaxation(f, None, 0.95, 1.0, 2, "dense")
        report(f, pr, 2, 0.95, 1.0, a.solver, a.starts)
        return

    if a.stress:
        m, info = make_nonconvex(a.n_assets, a.n_clusters, a.seed)
        print(f"stress instance: {info}")
    else:
        m = SyntheticMarket.generate(a.n_assets, a.n_clusters, a.seed)
    f = build_objective_from_market(m)
    ce = ChordalExtension(m.adjacency_matrix()).build()
    print(ce.summary())
    pr = build_portfolio_relaxation(
        f, ce.maximal_cliques, m.budget_lower, m.budget_upper, 2, a.structure, budget=a.budget,
        robust_terms=m.kappa_terms() if a.kappa_radius > 0 else None, kappa_radius=a.kappa_radius)
    report(f, pr, a.n_assets, m.budget_lower, m.budget_upper, a.solver, a.starts, robust=a.kappa_radius > 0)


if __name__ == "__main__":
    main()
