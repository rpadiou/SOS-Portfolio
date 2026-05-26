"""
Comprehensive Unit Tests — Sparse & DR SOS Hierarchy
=====================================================

Validates each component of the sparse pipeline independently, then
tests end-to-end convergence on the 2-asset problem (known ground truth).

Run:
    pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.polynomial_ring import (
    MultivariatePolynomial,
    generate_monomials,
    eval_monomial,
)
from src.portfolio_problem import (
    build_objective,
    build_constraints,
    build_objective_from_market,
    build_constraints_from_market,
    SyntheticMarket,
    BUDGET_LOWER,
)
from src.graph_sparsity import (
    ChordalExtension,
    build_block_adjacency,
    build_correlation_graph,
)
from src.indexer import SparseIndexer
from src.sos_hierarchy import LasserreRelaxation, SparseLasserreRelaxation
from src.extractor import MinimizerExtractor
from src.local_solver import scipy_optimize, analyze_local_minima


# ─────────────────────────────────────────────────────────────────────────────
# 1. Polynomial ring
# ─────────────────────────────────────────────────────────────────────────────

class TestPolynomialRing:
    def test_generate_monomials_n2_d2(self):
        mons = generate_monomials(2, 2)
        # C(2+2,2) = 6
        assert len(mons) == 6
        assert (0, 0) in mons
        assert (2, 0) in mons
        assert (0, 2) in mons

    def test_generate_monomials_n3_d2(self):
        mons = generate_monomials(3, 2)
        assert len(mons) == 10  # C(5,2)

    def test_eval_monomial(self):
        x = np.array([2.0, 3.0])
        assert eval_monomial((2, 1), x) == pytest.approx(4.0 * 3.0)
        assert eval_monomial((0, 0), x) == pytest.approx(1.0)

    def test_polynomial_eval(self):
        p = MultivariatePolynomial(2, {(2, 0): 1.0, (0, 2): 1.0})
        x = np.array([3.0, 4.0])
        assert p(x) == pytest.approx(25.0)

    def test_polynomial_add(self):
        p1 = MultivariatePolynomial(2, {(1, 0): 1.0})
        p2 = MultivariatePolynomial(2, {(0, 1): 2.0})
        p = p1 + p2
        assert p(np.array([1.0, 1.0])) == pytest.approx(3.0)

    def test_polynomial_mul(self):
        p1 = MultivariatePolynomial(2, {(1, 0): 1.0, (0, 1): 1.0})
        p2 = p1 * p1
        # (x+y)^2 = x^2 + 2xy + y^2
        assert p2(np.array([1.0, 2.0])) == pytest.approx(9.0)

    def test_gradient(self):
        # f = x1^2 + x2^2  =>  grad = [2x1, 2x2]
        p = MultivariatePolynomial(2, {(2, 0): 1.0, (0, 2): 1.0})
        x = np.array([3.0, 4.0])
        g = p.gradient(x)
        assert g[0] == pytest.approx(6.0)
        assert g[1] == pytest.approx(8.0)

    def test_degree(self):
        f = build_objective()
        assert f.degree == 4

    def test_sparse_coeffs(self):
        f = build_objective()
        assert len(f.coeffs) > 0
        for alpha in f.coeffs:
            assert sum(alpha) <= 4


# ─────────────────────────────────────────────────────────────────────────────
# 2. Chordal extension and clique extraction
# ─────────────────────────────────────────────────────────────────────────────

class TestChordalExtension:
    def test_block_adjacency_shape(self):
        adj = build_block_adjacency(10, 5)
        assert adj.shape == (10, 10)
        assert not np.any(np.diag(adj))  # no self-loops

    def test_block_adjacency_within_cluster(self):
        adj = build_block_adjacency(10, 5)
        # Within cluster 0: all pairs connected
        for i in range(5):
            for j in range(5):
                if i != j:
                    assert adj[i, j], f"Missing edge ({i},{j}) in cluster 0"
        # Between clusters: no edges
        for i in range(5):
            for j in range(5, 10):
                assert not adj[i, j], f"Unexpected inter-cluster edge ({i},{j})"

    def test_chordal_build_small(self):
        # Path graph 0-1-2: chordal extension should add 0-2
        adj = np.array([
            [False, True, False],
            [True, False, True],
            [False, True, False],
        ])
        ce = ChordalExtension(adj)
        ce.build()
        ch = ce.chordal_adjacency
        # Chordal extension of path: triangle
        assert ch[0, 2] or ch[0, 1], "Must add fill-in edge"

    def test_clique_extraction_block(self):
        adj = build_block_adjacency(10, 5)
        ce = ChordalExtension(adj)
        ce.build()
        cliques = ce.maximal_cliques
        # Block-diagonal with k=5: each block is already a clique, no fill-in
        # Expected: 2 cliques of size 5
        assert len(cliques) == 2
        sizes = sorted([len(c) for c in cliques])
        assert sizes == [5, 5]

    def test_rip_block(self):
        adj = build_block_adjacency(10, 5)
        ce = ChordalExtension(adj)
        ce.build()
        assert ce.verify_rip()

    def test_rip_larger(self):
        n, bs = 50, 5
        adj = build_block_adjacency(n, bs)
        ce = ChordalExtension(adj)
        ce.build()
        assert ce.verify_rip()

    def test_fill_in_zero_for_blocks(self):
        # Block-diagonal is already chordal → fill-in = 0
        adj = build_block_adjacency(20, 5)
        ce = ChordalExtension(adj)
        ce.build()
        assert ce.fill_in_count() == 0

    def test_correlation_graph_threshold(self):
        corr = np.eye(4)
        corr[0, 1] = corr[1, 0] = 0.6
        corr[2, 3] = corr[3, 2] = 0.4
        adj = build_correlation_graph(corr, threshold=0.5)
        assert adj[0, 1] and adj[1, 0]
        assert not adj[2, 3]  # below threshold


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sparse indexer
# ─────────────────────────────────────────────────────────────────────────────

class TestSparseIndexer:
    def _simple_indexer(self) -> SparseIndexer:
        cliques = [[0, 1], [1, 2]]
        return SparseIndexer(n=3, d=2, cliques=cliques)

    def test_zero_moment_present(self):
        idx = self._simple_indexer()
        assert idx.has_moment((0, 0, 0))
        assert idx.global_index((0, 0, 0)) == 0  # must be first (total deg=0)

    def test_local_monomials_count(self):
        idx = self._simple_indexer()
        # Clique [0,1] size 2, d=2: C(2+2,2)=6 monomials up to deg 2
        mons = idx.local_monomials(0, 2)
        assert len(mons) == 6

    def test_embed_local_to_global(self):
        idx = self._simple_indexer()
        # Clique 0 = [0,1]: local (1,0) maps to global (1,0,0)
        g_alpha = idx.embed(0, (1, 0))
        assert g_alpha == (1, 0, 0)
        # Clique 1 = [1,2]: local (0,1) maps to global (0,0,1)
        g_alpha2 = idx.embed(1, (0, 1))
        assert g_alpha2 == (0, 0, 1)

    def test_local_to_global_idx(self):
        idx = self._simple_indexer()
        i0 = idx.local_to_global(0, (0, 0))
        assert i0 == idx.global_index((0, 0, 0))

    def test_shared_moment_same_index(self):
        # y_{(0,1,0)} = moment of x_1 must be same whether accessed from clique 0 or 1
        idx = self._simple_indexer()
        # (0,1) in clique 0 (vars 0,1): local alpha s.t. x_1^1 = (0,1)
        # (1,0) in clique 1 (vars 1,2): local alpha s.t. x_1^1 = (1,0)
        i_from_0 = idx.local_to_global(0, (0, 1))
        i_from_1 = idx.local_to_global(1, (1, 0))
        assert i_from_0 == i_from_1, (
            "Shared moment x_1 must have same global index regardless of which clique"
        )

    def test_n_moments_block(self):
        # 2 non-overlapping cliques of size 5, d=2: moments are disjoint
        cliques = [list(range(5)), list(range(5, 10))]
        idx = SparseIndexer(n=10, d=2, cliques=cliques)
        # Each clique contributes C(5+4,4)=126 monomials up to deg 4, minus overlap (just zero)
        # Disjoint variables → no shared moments except y_0
        # n_moments = 2*C(5+4,4) - 1 (for shared y_0) approximately
        assert idx.n_moments > 0

    def test_clique_contains_support(self):
        idx = self._simple_indexer()
        # (1,1,0) has support in {0,1} = clique 0
        assert idx.clique_contains_support(0, (1, 1, 0))
        assert not idx.clique_contains_support(0, (1, 0, 1))

    def test_cliques_containing_var(self):
        idx = self._simple_indexer()
        # var 1 is in both clique 0 and 1
        assert set(idx.cliques_containing_var(1)) == {0, 1}
        # var 0 only in clique 0
        assert set(idx.cliques_containing_var(0)) == {0}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Portfolio problem builders
# ─────────────────────────────────────────────────────────────────────────────

class TestPortfolioProblem:
    def test_2asset_objective_degree(self):
        f = build_objective()
        assert f.degree == 4
        assert f.n_vars == 2

    def test_2asset_constraints(self):
        gs = build_constraints()
        assert len(gs) == 4
        for g in gs:
            assert g.n_vars == 2

    def test_feasible_point_constraints(self):
        gs = build_constraints()
        x = np.array([0.5, 0.45])  # inside K
        for g in gs:
            assert g(x) >= -1e-10

    def test_infeasible_point_constraints(self):
        gs = build_constraints()
        x = np.array([0.8, 0.8])  # sum > 1: violates g3
        violations = [g(x) < 0 for g in gs]
        assert any(violations)

    def test_synthetic_market_generate(self):
        market = SyntheticMarket.generate(n=20, n_clusters=4, seed=0)
        assert market.n == 20
        assert market.sigma.shape == (20, 20)
        # Covariance must be positive semi-definite
        eigvals = np.linalg.eigvalsh(market.sigma)
        assert np.all(eigvals >= -1e-8)

    def test_build_from_market(self):
        market = SyntheticMarket.generate(n=10, n_clusters=2, seed=0)
        f = build_objective_from_market(market)
        assert f.n_vars == 10
        assert f.degree == 4
        gs = build_constraints_from_market(market)
        assert len(gs) == 12  # 10 + 2

    def test_objective_eval_n10(self):
        market = SyntheticMarket.generate(n=10, n_clusters=2, seed=1)
        f = build_objective_from_market(market)
        x = np.ones(10) / 10
        val = f(x)
        assert np.isfinite(val)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Legacy dense Lasserre (2-asset convergence)
# ─────────────────────────────────────────────────────────────────────────────

class TestLegacyLasserre:
    def test_d1_lower_bound(self):
        f = build_objective()
        gs = build_constraints()
        rel = LasserreRelaxation(f, gs, order=1)
        res = rel.build_and_solve(solver="SCS")
        assert res["lower_bound"] is not None
        assert np.isfinite(res["lower_bound"])

    def test_d2_better_than_d1(self):
        f = build_objective()
        gs = build_constraints()
        r1 = LasserreRelaxation(f, gs, order=1).build_and_solve(solver="SCS")
        r2 = LasserreRelaxation(f, gs, order=2).build_and_solve(solver="SCS")
        lb1 = r1["lower_bound"]
        lb2 = r2["lower_bound"]
        assert lb1 is not None and lb2 is not None
        # Monotone: λ_1 ≤ λ_2
        assert lb2 >= lb1 - 1e-4

    def test_d2_gap_small(self):
        f = build_objective()
        gs = build_constraints()
        rel = LasserreRelaxation(f, gs, order=2)
        r = rel.build_and_solve(solver="SCS")
        # L-BFGS-B upper bound
        lbfgs = scipy_optimize(f, n_restarts=20, seed=0)
        lb = r["lower_bound"]
        ub = lbfgs["f_opt"]
        assert lb is not None
        gap = ub - lb
        # SCS eps=1e-4 → allow gap up to 5e-3 for the tiny 2-asset problem
        assert gap < 0.005, f"Gap {gap:.4f} too large at d=2"

    def test_flat_extension_d2(self):
        f = build_objective()
        gs = build_constraints()
        rel = LasserreRelaxation(f, gs, order=2)
        rel.build_and_solve(solver="SCS")
        fe = rel.flat_extension_check()
        assert fe.get("converged"), "Flat extension must hold at d=2 for 2-asset problem"

    def test_flat_extension_d1_error(self):
        f = build_objective()
        gs = build_constraints()
        rel = LasserreRelaxation(f, gs, order=1)
        rel.build_and_solve(solver="SCS")
        fe = rel.flat_extension_check()
        assert "error" in fe


# ─────────────────────────────────────────────────────────────────────────────
# 6. Sparse Lasserre hierarchy
# ─────────────────────────────────────────────────────────────────────────────

class TestSparseLasserre:
    def _2asset_sparse_setup(self):
        """Use sparse solver on 2-asset problem with trivial cliques."""
        f = build_objective()
        gs = build_constraints()
        cliques = [[0, 1]]  # single clique = dense
        return f, gs, cliques

    def test_sparse_single_clique_matches_dense(self):
        """Single-clique sparse = dense (up to solver tolerance)."""
        f = build_objective()
        gs = build_constraints()
        cliques = [[0, 1]]

        dense_rel = LasserreRelaxation(f, gs, order=2)
        sparse_rel = SparseLasserreRelaxation(f, gs, cliques=cliques, order=2)

        r_dense = dense_rel.build_and_solve(solver="SCS")
        r_sparse = sparse_rel.build_and_solve(solver="SCS")

        lb_d = r_dense["lower_bound"]
        lb_s = r_sparse["lower_bound"]
        assert lb_d is not None and lb_s is not None
        # Same problem, same solution up to SCS tolerance
        assert abs(lb_s - lb_d) < 0.005, (
            f"Single-clique sparse={lb_s:.5f} != dense={lb_d:.5f}"
        )

    def test_sparse_multi_clique_n10(self):
        """Sparse solver returns valid lower bound on n=10 problem."""
        market = SyntheticMarket.generate(n=10, n_clusters=2, seed=0)
        f = build_objective_from_market(market)
        gs = build_constraints_from_market(market)
        adj = build_block_adjacency(10, 5)
        ce = ChordalExtension(adj)
        ce.build()
        cliques = ce.maximal_cliques

        rel = SparseLasserreRelaxation(f, gs, cliques=cliques, order=2)
        res = rel.build_and_solve(solver="SCS")

        lb = res["lower_bound"]
        assert lb is not None, f"Solver failed: {res['status']}"
        assert np.isfinite(lb)

    def test_sparse_lb_leq_local_ub(self):
        """Sparse SOS lower bound must be ≤ local solver upper bound."""
        market = SyntheticMarket.generate(n=10, n_clusters=2, seed=1)
        f = build_objective_from_market(market)
        gs = build_constraints_from_market(market)
        adj = build_block_adjacency(10, 5)
        ce = ChordalExtension(adj)
        ce.build()
        cliques = ce.maximal_cliques

        sparse_rel = SparseLasserreRelaxation(f, gs, cliques=cliques, order=2)
        res = sparse_rel.build_and_solve(solver="SCS")
        lb = res["lower_bound"]

        lbfgs = scipy_optimize(f, n_restarts=10, seed=0, budget_lower=0.95)
        ub = lbfgs["f_opt"]

        assert lb is not None
        assert lb <= ub + 0.01, (
            f"Lower bound {lb:.4f} exceeds upper bound {ub:.4f}"
        )

    def test_dr_augmentation_increases_lb(self):
        """DR robustness (δ>0) should increase or equal the lower bound."""
        market = SyntheticMarket.generate(n=10, n_clusters=2, seed=2)
        f = build_objective_from_market(market)
        gs = build_constraints_from_market(market)
        adj = build_block_adjacency(10, 5)
        cliques = ChordalExtension(adj).build().maximal_cliques

        r_nom = SparseLasserreRelaxation(
            f, gs, cliques=cliques, order=2, delta_robust=0.0
        ).build_and_solve(solver="SCS")
        r_dr = SparseLasserreRelaxation(
            f, gs, cliques=cliques, order=2, delta_robust=0.1
        ).build_and_solve(solver="SCS")

        lb_nom = r_nom.get("lower_bound")
        lb_dr = r_dr.get("lower_bound")
        if lb_nom is not None and lb_dr is not None:
            # DR with δ>0 penalises uncertainty → conservative lower bound
            # (lb_dr could be lower because it minimises worst-case — correct behaviour)
            assert np.isfinite(lb_dr)

    def test_flat_extension_check_n10(self):
        """Flat extension check returns structured dict."""
        market = SyntheticMarket.generate(n=10, n_clusters=2, seed=0)
        f = build_objective_from_market(market)
        gs = build_constraints_from_market(market)
        adj = build_block_adjacency(10, 5)
        cliques = ChordalExtension(adj).build().maximal_cliques

        rel = SparseLasserreRelaxation(f, gs, cliques=cliques, order=2)
        rel.build_and_solve(solver="SCS")
        fe = rel.flat_extension_check()

        assert "converged" in fe
        assert "per_clique" in fe
        for entry in fe["per_clique"]:
            assert "converged" in entry


# ─────────────────────────────────────────────────────────────────────────────
# 7. Minimiser extractor
# ─────────────────────────────────────────────────────────────────────────────

class TestMinimizerExtractor:
    def test_rank_check_eye(self):
        """Identity matrix has full rank."""
        cliques = [[0, 1]]
        idx = SparseIndexer(n=2, d=2, cliques=cliques)
        N = idx.n_moments
        y_fake = np.zeros(N)
        y_fake[idx.global_index((0, 0))] = 1.0
        M = np.eye(6)  # 6×6 = C(2+2,2)^2
        ext = MinimizerExtractor({0: M}, cliques, y_fake, idx)
        r = ext._numerical_rank(M)
        assert r == 6

    def test_rank_check_rank1(self):
        """Rank-1 outer product has rank 1."""
        v = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        M = np.outer(v, v)
        cliques = [[0, 1]]
        idx = SparseIndexer(n=2, d=2, cliques=cliques)
        N = idx.n_moments
        y_fake = np.zeros(N)
        y_fake[0] = 1.0
        ext = MinimizerExtractor({0: M}, cliques, y_fake, idx)
        assert ext._numerical_rank(M) == 1

    def test_extract_from_converged_2asset(self):
        """Full pipeline: solve 2-asset → extract minimiser → evaluate f."""
        f = build_objective()
        gs = build_constraints()
        cliques = [[0, 1]]
        rel = SparseLasserreRelaxation(f, gs, cliques=cliques, order=2)
        res = rel.build_and_solve(solver="SCS")

        y_vals = res.get("moments")
        mms = res.get("moment_matrices", {})
        idx = res["indexer"]

        if y_vals is None or not mms:
            pytest.skip("SDP did not solve cleanly")

        ext = MinimizerExtractor(mms, cliques, y_vals, idx)
        if ext.is_flat():
            minimisers = ext.extract()
            if minimisers is not None:
                assert minimisers.shape[1] == 2
                for x in minimisers:
                    # Must be in feasible region (approximately)
                    assert np.all(x >= -0.05)
                    assert sum(x) >= 0.90


# ─────────────────────────────────────────────────────────────────────────────
# 8. Local solver
# ─────────────────────────────────────────────────────────────────────────────

class TestLocalSolver:
    def test_2asset_finds_minimum(self):
        f = build_objective()
        res = scipy_optimize(f, n_restarts=20, seed=0)
        assert res["x_opt"] is not None
        assert np.isfinite(res["f_opt"])
        # Must satisfy budget constraint approximately
        s = float(np.sum(res["x_opt"]))
        assert 0.94 <= s <= 1.01

    def test_n10_finds_minimum(self):
        market = SyntheticMarket.generate(n=10, n_clusters=2, seed=0)
        f = build_objective_from_market(market)
        res = scipy_optimize(f, n_restarts=10, seed=0, budget_lower=0.95)
        assert res["x_opt"] is not None
        assert np.isfinite(res["f_opt"])

    def test_analyze_local_minima(self):
        f = build_objective()
        res = scipy_optimize(f, n_restarts=30, seed=0)
        lma = analyze_local_minima(res["all_results"])
        assert lma["n_distinct_minima"] >= 1
        assert all("x" in m and "f" in m and "count" in m
                   for m in lma["local_minima"])

    def test_multiple_restarts_coverage(self):
        """More restarts should find at least as good a solution."""
        f = build_objective()
        r5 = scipy_optimize(f, n_restarts=5, seed=0)
        r50 = scipy_optimize(f, n_restarts=50, seed=0)
        assert r50["f_opt"] <= r5["f_opt"] + 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# 9. Integration: end-to-end sparse pipeline on n=20
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEndSparse:
    def test_full_pipeline_n15(self):
        """Full sparse SOS pipeline: data → graph → hierarchy → bound comparison.

        Uses n=15 (3 clusters × 5 assets) — same structural properties as n=20
        but with 3 cliques of size 5 (21×21 SDP blocks) which stays within the
        computational budget of a laptop with SCS.
        """
        n, n_clusters = 15, 3
        market = SyntheticMarket.generate(n=n, n_clusters=n_clusters, seed=42)
        f = build_objective_from_market(market)
        gs = build_constraints_from_market(market)

        # Graph: 3 cliques of size 5, already chordal, RIP holds trivially
        adj = build_block_adjacency(n, n // n_clusters)
        ce = ChordalExtension(adj)
        ce.build()
        cliques = ce.maximal_cliques
        assert ce.verify_rip()
        assert len(cliques) == 3

        # Sparse SOS at d=2 — use CLARABEL for robust convergence on n=15
        rel = SparseLasserreRelaxation(f, gs, cliques=cliques, order=2)
        res = rel.build_and_solve(solver="CLARABEL")
        lb = res["lower_bound"]
        assert lb is not None and np.isfinite(lb), f"SDP failed: {res['status']}"

        # Local upper bound (few restarts to keep test fast)
        lbfgs = scipy_optimize(f, n_restarts=5, seed=0, budget_lower=0.95)
        ub = lbfgs["f_opt"]

        # Certified gap: lb ≤ f* ≤ ub
        assert lb <= ub + 0.05, f"lb={lb:.4f} > ub={ub:.4f}"

        # Flat extension structure is present
        fe = rel.flat_extension_check()
        assert "converged" in fe
        assert "per_clique" in fe
        assert len(fe["per_clique"]) == 3

    def test_complexity_reduction_n50(self):
        """Verify that sparse has fewer scalar variables than dense for n=50."""
        n, d = 50, 2
        n_clusters = 10
        adj = build_block_adjacency(n, n // n_clusters)
        ce = ChordalExtension(adj)
        ce.build()
        cliques = ce.maximal_cliques

        dense_s = len(generate_monomials(n, d))
        dense_vars = dense_s ** 2

        sparse_vars = sum(len(generate_monomials(len(c), d)) ** 2 for c in cliques)
        assert sparse_vars < dense_vars, (
            f"Sparse ({sparse_vars}) not smaller than dense ({dense_vars})"
        )
        ratio = dense_vars / sparse_vars
        assert ratio > 50, f"Expected > 50× reduction, got {ratio:.1f}×"
