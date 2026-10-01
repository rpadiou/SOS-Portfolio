"""Empirical utilities, tested on a *synthetic* fixture (heavy-tailed returns), not on market data."""
import numpy as np
import pytest

from sos_portfolio import build_portfolio_relaxation, certify, MinimizerExtractor
from sos_portfolio import empirical as em


@pytest.fixture(scope="module")
def R():
    rng = np.random.RandomState(0)
    L = np.linalg.cholesky(0.3 * np.ones((6, 6)) + 0.7 * np.eye(6))
    return (rng.standard_t(4, size=(400, 6)) @ L.T) * 0.01


def test_polynomial_matches_direct_objective(R):
    Rc = em.prepare_window(R)
    sect = [0, 0, 0, 1, 1, 1]
    for blocks in (em.blocks_of(6, None), em.blocks_of(6, sect)):
        lam = em.lambdas(Rc, blocks, (1, 1, 1))
        f = em.polynomial(Rc, blocks, lam)
        x = np.random.RandomState(1).dirichlet(np.ones(6))
        assert f(x) == pytest.approx(em.objective(x, Rc, blocks, lam)[0], rel=1e-9)
        np.testing.assert_allclose(f.gradient(x), em.objective(x, Rc, blocks, lam)[1], rtol=1e-7, atol=1e-10)
        m = em.model_moments(np.full(6, 1 / 6), Rc, blocks)
        assert all(lam[k] > 0 for k in (2, 3, 4))


def test_sos_certifies_the_estimated_polynomial_with_equality_budget(R):
    Rc = em.prepare_window(R)
    blocks = em.blocks_of(6, [0, 0, 0, 1, 1, 1])
    lam = em.lambdas(Rc, blocks, (1, 1, 1))
    f = em.polynomial(Rc, blocks, lam)
    pr = build_portfolio_relaxation(f, blocks, 1.0, 1.0, 2, "sparse")
    r = pr.solve("CLARABEL")
    xl = em.solve_local(Rc, blocks, lam)
    assert r["status"] == "optimal"
    assert r["lower_bound"] <= f(xl) + 1e-6
    ex = MinimizerExtractor(pr.relaxation, r["moments"], 6)
    c = certify(f, r["lower_bound"], [ex.mean_point(), xl], 1.0, 1.0)
    assert c["certified"]


@pytest.mark.parametrize("fn", [em.min_variance, em.erc, em.min_cvar, em.hrp])
def test_baselines_are_long_only_fully_invested(fn, R):
    w = fn(R)
    assert w.min() >= -1e-9 and w.sum() == pytest.approx(1.0)


def test_erc_equalises_risk_contributions(R):
    w = em.erc(R)
    S = em.ledoit_wolf_cov(R)
    rc = w * (S @ w)
    assert rc.max() / rc.min() < 1.01


def test_min_variance_beats_equal_weight_in_sample(R):
    S = em.ledoit_wolf_cov(R)
    w = em.min_variance(R)
    assert w @ S @ w <= np.full(6, 1 / 6) @ S @ np.full(6, 1 / 6) + 1e-12


def test_sharpe_test_identical_series_and_different_series():
    rng = np.random.RandomState(0)
    a = rng.randn(1000) * 0.01 + 0.0004
    assert em.sharpe_diff_test(a, a.copy(), n_boot=200)["p_value"] > 0.9
    b = rng.randn(1000) * 0.01 - 0.0012
    assert em.sharpe_diff_test(a, b, n_boot=300)["p_value"] < 0.05


def test_block_bootstrap_interval_covers_zero_for_equal_series():
    rng = np.random.RandomState(1)
    a = rng.randn(800)
    d, lo, hi = em.block_bootstrap_diff(a, a, np.std, n=200)
    assert d == 0 and lo == 0 == hi
