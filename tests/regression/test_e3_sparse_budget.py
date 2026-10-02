"""E3: the sparse bound must be a *usable* relaxation of the original problem.

v1.0 replaced the budget by E[sum x] in [b_lo, b_hi], which decoupled the cliques
(sparse lb -22.07 vs dense -2.43 on the first instance below).
"""
import numpy as np
import pytest

from sos_portfolio import (ChordalExtension, SyntheticMarket, build_objective_from_market,
                           build_portfolio_relaxation, scipy_optimize)
from util import cliques_of, random_sparse_poly

E3 = [((6, 2, 0, 2.0, 0.3, 0.05), -2.427636), ((6, 2, 1, 4.0, 0.3, 0.05), -5.399794),
      ((6, 3, 2, 6.0, 0.2, 0.05), 0.128596), ((8, 2, 3, 5.0, 0.3, 0.05), -6.242960)]


def bounds(f, cl, d=2):
    sp = build_portfolio_relaxation(f, cl, 0.95, 1.0, d, "sparse").solve()["lower_bound"]
    de = build_portfolio_relaxation(f, None, 0.95, 1.0, d, "dense").solve()["lower_bound"]
    return sp, de


@pytest.mark.parametrize("case,ref", E3)
def test_sparse_matches_dense_and_local_on_e3_instances(case, ref):
    n, nc, seed, sk, ku, im = case
    m = SyntheticMarket.generate(n, nc, seed, skew_scale=sk, kurt_base=ku, impact_base=im)
    f = build_objective_from_market(m)
    cl = ChordalExtension(m.adjacency_matrix()).build().maximal_cliques
    sp, de = bounds(f, cl)
    ub = scipy_optimize(f, 30, 0)["f_opt"]
    assert de == pytest.approx(ref, abs=1e-4)
    assert sp <= de + 1e-5                       # sparse relaxes dense
    assert de - sp <= 1e-3 * max(1.0, abs(de))   # and is tight here
    assert sp <= ub + 1e-6 and de <= ub + 1e-6


# With overlapping cliques the budget clique closes a cycle: the relaxation is valid, the RIP warning is expected.
@pytest.mark.filterwarnings("ignore:the cliques do not admit")
@pytest.mark.parametrize("kind,n,seed", [("chain", 7, 0), ("chain", 7, 1), ("star", 6, 0), ("random", 8, 2),
                                         ("random", 8, 5), ("chain", 5, 3)])
def test_overlapping_cliques_sparse_is_valid_relaxation_of_dense(kind, n, seed):
    cl = cliques_of(kind, n, seed)
    f = random_sparse_poly(n, cl, seed)
    sp, de = bounds(f, cl)
    ub = scipy_optimize(f, 40, seed)["f_opt"]
    assert sp <= de + 1e-5
    assert de <= ub + 1e-6
    rng = np.random.RandomState(seed)
    X = rng.dirichlet(np.ones(n) * 0.5, size=1000) * rng.uniform(0.95, 1.0, size=(1000, 1))
    assert min(f(x) for x in X) >= sp - 1e-6     # lb <= f on 1000 feasible points


@pytest.mark.parametrize("budget", ["star", "chain"])
def test_budget_variants_agree(budget):
    m = SyntheticMarket.generate(9, 3, 1, skew_scale=3.0, kurt_base=0.3, impact_base=0.05)
    f = build_objective_from_market(m)
    cl = ChordalExtension(m.adjacency_matrix()).build().maximal_cliques
    star = build_portfolio_relaxation(f, cl, 0.95, 1.0, 2, "sparse", budget="star").solve()["lower_bound"]
    other = build_portfolio_relaxation(f, cl, 0.95, 1.0, 2, "sparse", budget=budget).solve()["lower_bound"]
    de = build_portfolio_relaxation(f, None, 0.95, 1.0, 2, "dense").solve()["lower_bound"]
    ub_ = scipy_optimize(f, 30, 0)["f_opt"]
    assert other <= ub_ + 1e-5
    print(budget, star, other, de)
    assert other <= de + 1e-5


def test_equality_budget():
    m = SyntheticMarket.generate(6, 2, 0, skew_scale=2.0, kurt_base=0.3, impact_base=0.05)
    f = build_objective_from_market(m)
    cl = ChordalExtension(m.adjacency_matrix()).build().maximal_cliques
    sp = build_portfolio_relaxation(f, cl, 1.0, 1.0, 2, "sparse").solve()
    de = build_portfolio_relaxation(f, None, 1.0, 1.0, 2, "dense").solve()
    ub = scipy_optimize(f, 30, 0, b_lo=1.0, b_hi=1.0)["f_opt"]
    assert sp["status"] == de["status"] == "optimal"
    assert sp["lower_bound"] <= de["lower_bound"] + 1e-5
    assert de["lower_bound"] <= ub + 1e-5
