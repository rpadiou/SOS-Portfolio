from math import comb

import pytest

from sos_portfolio import (ChordalExtension, SyntheticMarket, build_objective_from_market,
                           build_portfolio_relaxation, scipy_optimize)
from sos_portfolio.sos_hierarchy import dense_sizes


@pytest.mark.parametrize("n,eq", [(3, False), (5, False), (5, True)])
def test_closed_form_dense_sizes_match_builder(n, eq):
    m = SyntheticMarket.generate(n, 1, 0)
    f = build_objective_from_market(m)
    pr = build_portfolio_relaxation(f, None, 1.0 if eq else 0.95, 1.0, 2, "dense")
    pr.relaxation.build()
    assert pr.relaxation.sizes() == dense_sizes(n, 2, True, eq)


def test_n50_index_counts_and_solves():
    m = SyntheticMarket.generate(50, 10, 0)
    f = build_objective_from_market(m)
    cl = ChordalExtension(m.adjacency_matrix()).build().maximal_cliques
    assert len(cl) == 10 and all(len(c) == 5 for c in cl)
    star = build_portfolio_relaxation(f, cl, 0.95, 1.0, 2, "sparse", budget="chain")
    # x-moments supported in one clique: 10 * (C(9,4) - 1) + 1 = 1251 (the v1 count); the rest are
    # auxiliary sector-sum moments (s_k, p_k) introduced to impose the budget pointwise
    x_only = 10 * (comb(9, 4) - 1) + 1
    assert x_only == 1251
    r = star.solve("CLARABEL")
    assert r["status"] == "optimal" and r["sizes"]["unique_moments"] > x_only
    assert r["sizes"]["max_moment_block"] == 21           # 5 assets -> C(7,2)
    assert comb(52, 2) == 1326 and dense_sizes(50)["max_block"] == 1326
    ub = scipy_optimize(f, 10, 0)["f_opt"]
    assert r["lower_bound"] <= ub + 1e-6 and ub - r["lower_bound"] < 1e-4
