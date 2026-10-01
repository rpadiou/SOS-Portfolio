"""E3: the sparse bound must not be far below the dense bound on non-convex instances.

In v1.0 the budget constraint was replaced by a constraint on E[sum x], which
decoupled the cliques. Marked xfail(strict) until the fix of Phase 1.1 lands.
"""
import pytest

from sos_portfolio import (LasserreRelaxation, SparseLasserreRelaxation, SyntheticMarket,
                           build_constraints_from_market, build_objective_from_market)
from sos_portfolio.graph_sparsity import ChordalExtension

CASES = [(6, 2, 0, 2.0, 0.3, 0.05), (6, 2, 1, 4.0, 0.3, 0.05), (8, 2, 3, 5.0, 0.3, 0.05)]


@pytest.mark.xfail(strict=True, reason="E3: budget relaxed to a mean constraint")
@pytest.mark.parametrize("case", CASES)
def test_sparse_bound_close_to_dense(case):
    n, nc, seed, sk, ku, im = case
    m = SyntheticMarket.generate(n, nc, seed, skew_scale=sk, kurt_base=ku, impact_base=im)
    f = build_objective_from_market(m)
    g = build_constraints_from_market(m)
    cl = ChordalExtension(m.adjacency_matrix()).build().maximal_cliques
    lb_sp = SparseLasserreRelaxation(f, g, cl, 2).build_and_solve("CLARABEL")["lower_bound"]
    lb_de = LasserreRelaxation(f, g, 2).build_and_solve("CLARABEL")["lower_bound"]
    assert lb_de - lb_sp <= 1e-2 * max(1.0, abs(lb_de))
