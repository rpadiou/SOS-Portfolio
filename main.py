"""
SOS-Portfolio: Production-Grade Sparse & Distributionally Robust SOS Optimizer
================================================================================

Two modes:

  --mode demo     2-asset toy model (original pipeline, backward-compatible)
  --mode sparse   n-asset sparse DR-SOS hierarchy (production, default)

Usage:
    python main.py --mode sparse --n-assets 50 --solver SCS --save-figs
    python main.py --mode demo   --orders 1 2 3 --solver SCS --save-figs
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from src import (
    # Demo mode
    build_objective,
    build_constraints,
    LasserreRelaxation,
    plot_risk_surface,
    plot_convergence,
    plot_moment_matrix_spectrum,
    # Sparse mode
    SyntheticMarket,
    build_objective_from_market,
    build_constraints_from_market,
    SparseLasserreRelaxation,
    ChordalExtension,
    build_block_adjacency,
    build_correlation_graph,
    SparseIndexer,
    MinimizerExtractor,
    scipy_optimize,
    analyze_local_minima,
    generate_monomials,
)


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def hdr(title: str) -> None:
    print(f"\n{'─'*72}")
    print(f"  {title}")
    print(f"{'─'*72}")


def banner(msg: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {msg}")
    print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# Demo mode: 2-asset dense Lasserre hierarchy
# ─────────────────────────────────────────────────────────────────────────────

def run_demo(args: argparse.Namespace) -> None:
    banner("SOS-Portfolio  |  Demo: 2-Asset Dense Lasserre Hierarchy")

    hdr("Step 1: Problem Formulation")
    f = build_objective()
    constraints = build_constraints()
    print(f"  f ∈ R[x₁,x₂], degree={f.degree}, terms={len(f.coeffs)}")
    print(f"  K: {len(constraints)} semialgebraic constraints")

    hdr("Step 2: L-BFGS-B Baseline")
    t0 = time.perf_counter()
    lbfgs = scipy_optimize(f, n_restarts=args.n_restarts)
    print(f"  Time: {time.perf_counter()-t0:.2f}s")
    lma = analyze_local_minima(lbfgs["all_results"])
    print(f"  Distinct minima: {lma['n_distinct_minima']}")
    print(f"  Best f*_UB = {lbfgs['f_opt']:.6f}  at x* = {lbfgs['x_opt']}")

    hdr("Step 3: Dense Lasserre SOS Hierarchy")
    sos_bounds, flat_results, sos_by_d = [], [], {}

    for d in args.orders:
        s = len(generate_monomials(2, d))
        print(f"\n  d={d}  |  M_d: {s}×{s}")
        rel = LasserreRelaxation(f, constraints, order=d)
        t0 = time.perf_counter()
        res = rel.build_and_solve(solver=args.solver, verbose=args.verbose_sdp)
        dt = time.perf_counter() - t0
        lb = res.get("lower_bound")
        sos_bounds.append(lb)
        sos_by_d[d] = res
        fe = rel.flat_extension_check()
        flat_results.append(fe)
        lb_str = f"{lb:.6f}" if lb is not None else "FAILED"
        gap_str = (
            f"{lbfgs['f_opt'] - lb:.2e}" if lb is not None else "N/A"
        )
        flat_str = (
            f"{'✓' if fe.get('converged') else '✗'}  "
            f"rank(M_{d})={fe.get('rank_d')}  rank(M_{d-1})={fe.get('rank_d_minus_1')}"
            if "error" not in fe else fe["error"]
        )
        print(f"  λ_{d} = {lb_str}  |  gap = {gap_str}  |  t = {dt:.2f}s")
        print(f"  Flat ext: {flat_str}")

    hdr("Step 4: Summary")
    _print_summary_table(args.orders, sos_bounds, lbfgs["f_opt"])

    hdr("Step 5: Visualisation")
    save_dir = Path("results") if args.save_figs else None
    if save_dir:
        save_dir.mkdir(exist_ok=True)
    last = sos_by_d.get(args.orders[-1])
    plot_risk_surface(
        f, sos_result=last, lbfgs_result=lbfgs,
        save_path=str(save_dir / "risk_surface.png") if save_dir else None,
    )
    plot_convergence(
        sos_bounds, args.orders, lbfgs_best=lbfgs["f_opt"],
        save_path=str(save_dir / "convergence.png") if save_dir else None,
    )
    plot_moment_matrix_spectrum(
        flat_results, args.orders,
        save_path=str(save_dir / "moment_spectrum.png") if save_dir else None,
    )
    if not save_dir:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        print(f"  Figures → {save_dir.resolve()}/")


# ─────────────────────────────────────────────────────────────────────────────
# Sparse mode: n-asset sparse DR-SOS hierarchy
# ─────────────────────────────────────────────────────────────────────────────

def run_sparse(args: argparse.Namespace) -> None:
    n = args.n_assets
    banner(
        f"SOS-Portfolio  |  Sparse DR-SOS  |  n={n} assets  |  "
        f"solver={args.solver}  |  δ={args.delta_robust}"
    )

    # ── Step 1: Synthetic market data ────────────────────────────────────────
    hdr("Step 1: Synthetic Market Data")
    n_clusters = args.n_clusters
    cluster_size = n // n_clusters
    if n % n_clusters != 0:
        raise ValueError(f"--n-assets ({n}) must be divisible by --n-clusters ({n_clusters})")

    market = SyntheticMarket.generate(
        n=n,
        n_clusters=n_clusters,
        seed=args.seed,
        budget_lower=0.95,
    )
    print(f"  Assets: {n}  |  Clusters: {n_clusters}×{cluster_size}")
    print(f"  Correlated pairs (edges): {len(market.edge_set)}")

    f = build_objective_from_market(market)
    constraints = build_constraints_from_market(market)
    print(f"  Objective: deg={f.degree}, sparse terms={len(f.coeffs)}")
    print(f"  Constraints: {len(constraints)}")

    # ── Step 2: Correlation graph + chordal extension ────────────────────────
    hdr("Step 2: Chordal Sparsity")

    # Block adjacency: within-cluster all-pairs connected (already chordal)
    adj = build_block_adjacency(n, cluster_size)
    chordal = ChordalExtension(adj)
    chordal.build()
    cliques = chordal.maximal_cliques
    print(chordal.summary())
    print(f"  RIP verified: {chordal.verify_rip()}")

    # Complexity comparison
    d = 2
    dense_size = len(generate_monomials(n, d))
    sparse_sizes = [len(generate_monomials(len(c), d)) for c in cliques]
    dense_vars = dense_size ** 2
    sparse_vars = sum(s ** 2 for s in sparse_sizes)
    print(f"\n  Complexity comparison (d={d}):")
    print(f"  {'':30s} {'SDP block':>12}  {'Scalar vars':>14}")
    print(f"  {'Dense  M_d(y)':<30} {dense_size:>6}×{dense_size:<6}  {dense_vars:>14,}")
    print(f"  {'Sparse Σ_k M_d^{{I_k}}(y)':<30} {'Σ'+str(sparse_sizes[:3])+'...':>12}  {sparse_vars:>14,}")
    print(f"  {'Reduction factor':<30} {'':>12}  {dense_vars/max(sparse_vars,1):>14.1f}×")

    # ── Step 3: L-BFGS-B baseline ────────────────────────────────────────────
    hdr("Step 3: L-BFGS-B Baseline")
    t0 = time.perf_counter()
    lbfgs = scipy_optimize(
        f, n_restarts=args.n_restarts, seed=args.seed, budget_lower=0.95
    )
    dt_lbfgs = time.perf_counter() - t0
    lma = analyze_local_minima(lbfgs["all_results"])
    print(f"  n_restarts={args.n_restarts}  |  time={dt_lbfgs:.1f}s")
    print(f"  Distinct minima found: {lma['n_distinct_minima']}")
    print(f"  Best f*_UB = {lbfgs['f_opt']:.6f}")

    # ── Step 4: Sparse (DR-)SOS hierarchy at d=2 ────────────────────────────
    hdr(f"Step 4: Sparse DR-SOS Hierarchy  (d={d}, δ={args.delta_robust})")

    indexer = SparseIndexer(n=n, d=d, cliques=cliques)
    print(indexer.summary())

    relaxation = SparseLasserreRelaxation(
        f=f,
        constraints=constraints,
        cliques=cliques,
        order=d,
        delta_robust=args.delta_robust,
    )

    t0 = time.perf_counter()
    result = relaxation.build_and_solve(solver=args.solver, verbose=args.verbose_sdp)
    dt_sos = time.perf_counter() - t0

    lb = result.get("lower_bound")
    status = result.get("status")
    lb_str = f"{lb:.6f}" if lb is not None else "FAILED"
    gap_str = (
        f"{lbfgs['f_opt'] - lb:.2e}" if lb is not None else "N/A"
    )
    print(f"\n  Status:         {status}")
    print(f"  Lower bound λ_d: {lb_str}")
    print(f"  L-BFGS-B upper:  {lbfgs['f_opt']:.6f}")
    print(f"  Certified gap:   {gap_str}")
    print(f"  Solve time:      {dt_sos:.2f}s")
    print(f"  Moment vars:     {result.get('n_scalar_vars')}")
    print(f"  SDP blocks:      {result.get('n_sdp_blocks')}")

    # ── Step 5: Flat extension + minimiser extraction ────────────────────────
    hdr("Step 5: Flat Extension & Minimiser Extraction")
    fe = relaxation.flat_extension_check()
    print(f"  Global flat extension: {'✓ CERTIFIED' if fe.get('converged') else '✗ NOT YET'}")
    if "per_clique" in fe:
        n_flat = sum(1 for c in fe["per_clique"] if c.get("converged"))
        print(f"  Flat cliques: {n_flat}/{len(fe['per_clique'])}")

    y_vals = result.get("moments")
    moment_matrices = result.get("moment_matrices", {})

    if y_vals is not None and fe.get("converged") and moment_matrices:
        extractor = MinimizerExtractor(
            moment_matrices=moment_matrices,
            cliques=cliques,
            y_vals=y_vals,
            indexer=indexer,
        )
        minimisers = extractor.extract()
        if minimisers is not None:
            print(f"\n  Extracted {minimisers.shape[0]} global minimiser(s):")
            for j, xj in enumerate(minimisers):
                fval = f(xj)
                budget = float(np.sum(xj))
                print(f"    x*[{j}]: f={fval:.6f}  Σx_i={budget:.4f}")
        else:
            print("  Extraction: flat extension certified but rank is > 1.")
    elif y_vals is None:
        print("  Extraction skipped: solver did not return moments.")
    else:
        print("  Extraction skipped: flat extension not certified at d=2.")
        print("  Increase order to d=3 for stronger certificate.")

    # ── Step 6: Complexity reduction table ──────────────────────────────────
    hdr("Step 6: Complexity Reduction Summary")
    _print_complexity_table(n, d, cliques, dense_size, dense_vars, sparse_vars, dt_sos)

    banner("Run complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary_table(
    orders: list, sos_bounds: list, lbfgs_best: float
) -> None:
    print(f"\n  {'Method':<36} {'Lower bound':>14} {'Upper bound':>14}")
    print(f"  {'─'*64}")
    print(
        f"  {'L-BFGS-B (multi-start)':<36} {'—':>14} {lbfgs_best:>14.6f}"
    )
    for d, lb in zip(orders, sos_bounds):
        lb_s = f"{lb:.6f}" if lb is not None else "FAILED"
        print(f"  {f'SOS d={d}':<36} {lb_s:>14} {'≤ f*':>14}")


def _print_complexity_table(
    n: int,
    d: int,
    cliques: list,
    dense_size: int,
    dense_vars: int,
    sparse_vars: int,
    dt_sos: float,
) -> None:
    max_clique = max(len(c) for c in cliques)
    n_cliques = len(cliques)
    max_local = len(generate_monomials(max_clique, d))
    print(
        f"\n  {'Metric':<40} {'Dense':>14} {'Sparse':>14} {'Ratio':>10}"
    )
    print(f"  {'─'*78}")
    print(
        f"  {'Moment matrix size':<40} {dense_size:>12}² {max_local:>12}² "
        f"{dense_size/max_local:>9.1f}×"
    )
    print(
        f"  {'Scalar SDP variables (approx.)':<40} {dense_vars:>14,} "
        f"{sparse_vars:>14,} {dense_vars/max(sparse_vars,1):>9.1f}×"
    )
    print(f"  {'Number of SDP blocks':<40} {'1':>14} {n_cliques:>14}")
    print(f"  {'Max clique size':<40} {n:>14} {max_clique:>14}")
    print(f"  {'Sparse SOS solve time (s)':<40} {'—':>14} {dt_sos:>14.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SOS-Portfolio: Sparse & Distributionally Robust SOS Portfolio Optimizer"
    )
    p.add_argument(
        "--mode", choices=["demo", "sparse"], default="sparse",
        help="'demo' = 2-asset dense Lasserre; 'sparse' = n-asset sparse DR-SOS",
    )
    # Demo-mode args
    p.add_argument("--orders", nargs="+", type=int, default=[1, 2, 3])
    # Sparse-mode args
    p.add_argument("--n-assets", type=int, default=50, dest="n_assets")
    p.add_argument("--n-clusters", type=int, default=10, dest="n_clusters")
    p.add_argument(
        "--delta-robust", type=float, default=0.05, dest="delta_robust",
        help="DR robustness parameter δ (0 = nominal)",
    )
    p.add_argument("--seed", type=int, default=0)
    # Shared args
    p.add_argument(
        "--solver", type=str, default="SCS", choices=["SCS", "CLARABEL", "MOSEK"]
    )
    p.add_argument("--n-restarts", type=int, default=50, dest="n_restarts")
    p.add_argument("--save-figs", action="store_true", dest="save_figs")
    p.add_argument("--verbose-sdp", action="store_true", dest="verbose_sdp")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "demo":
        run_demo(args)
    else:
        run_sparse(args)
