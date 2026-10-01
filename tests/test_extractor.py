import numpy as np
import pytest

from sos_portfolio import (ChordalExtension, MinimizerExtractor, MomentRelaxation, MultivariatePolynomial,
                           build_portfolio_relaxation, certify, scipy_optimize)
from sos_portfolio.extractor import numerical_ranks


def moments_of(atoms, weights, rel):
    ix = rel.indexer
    y = np.zeros(ix.n_moments)
    for k in range(len(ix.cliques)):
        for m in ix.local_monomials(k, 2 * ix.d, full=True):
            e = np.zeros(ix.n, dtype=int)
            e[ix.cliques[k]] = m
            y[ix.index(e)] = sum(w * np.prod(a ** e) for w, a in zip(weights, atoms))
    return y


@pytest.mark.parametrize("n,r", [(2, 2), (3, 2), (3, 3)])
def test_henrion_lasserre_recovers_known_atoms(n, r):
    rng = np.random.RandomState(n * 10 + r)
    atoms = rng.uniform(0.05, 0.9, size=(r, n))
    w = rng.dirichlet(np.ones(r))
    rel = MomentRelaxation(n, MultivariatePolynomial(n, {(1,) + (0,) * (n - 1): 1.0}), [list(range(n))], [], 2)
    y = moments_of(atoms, w, rel)
    ex = MinimizerExtractor(rel, y, n)
    assert ex.flatness()["flat"]
    ca = ex.clique_atoms(0)
    assert ca["rank"] == r
    got = sorted(map(tuple, np.round(ca["atoms"], 6)))
    want = sorted(map(tuple, np.round(atoms, 6)))
    np.testing.assert_allclose(got, want, atol=1e-6)
    np.testing.assert_allclose(sorted(ca["weights"]), sorted(w), atol=1e-6)


def test_merge_over_overlapping_cliques_matches_atoms_on_shared_variables():
    n = 5
    cl = [[0, 1, 2], [2, 3, 4]]
    atoms = np.array([[0.1, 0.2, 0.3, 0.15, 0.25], [0.4, 0.05, 0.3, 0.1, 0.15]])   # share x2 = 0.3 -> ambiguous
    atoms2 = np.array([[0.1, 0.2, 0.3, 0.15, 0.25], [0.4, 0.05, 0.6, 0.1, 0.15]])
    w = np.array([0.5, 0.5])
    f = MultivariatePolynomial(n, {(1, 0, 0, 0, 0): 1.0})
    rel = MomentRelaxation(n, f, cl, [], 2)
    ex = MinimizerExtractor(rel, moments_of(atoms2, w, rel), n)
    pts = ex.extract()["points"]
    want = sorted(map(tuple, np.round(atoms2, 6)))
    assert sorted(map(tuple, np.round(pts, 6))) == want
    # ambiguous merge (same x2 in both atoms) returns all combinations, never a wrong unique answer
    ex2 = MinimizerExtractor(rel, moments_of(atoms, w, rel), n)
    pts2 = ex2.extract()["points"]
    assert len(pts2) >= 2
    for a in atoms:
        assert any(np.allclose(a, p, atol=1e-6) for p in pts2)


def test_rank_detection_threshold_and_gap():
    M = np.diag([1.0, 0.5, 1e-9, 1e-10])
    r = numerical_ranks(M, 1e-5)
    assert r["rank_thr"] == 2 and r["rank_gap"] == 2


def test_failed_extraction_is_explicit_when_not_flat():
    n = 2
    rel = MomentRelaxation(n, MultivariatePolynomial(n, {(1, 0): 1.0}), [[0, 1]], [], 2)
    # moments of the uniform measure on [0,1]^2: rank full, not flat -> extraction must not pretend
    y = np.zeros(rel.indexer.n_moments)
    for m in rel.indexer.local_monomials(0, 4):
        y[rel.indexer.index(m)] = np.prod([1.0 / (a + 1) for a in m])
    ex = MinimizerExtractor(rel, y, n)
    assert not ex.flatness()["flat"]


def test_certificate_rejects_infeasible_candidate():
    f = MultivariatePolynomial(2, {(2, 0): 1.0, (0, 2): 1.0})
    c = certify(f, 0.0, [np.array([0.9, 0.9])], 0.95, 1.0, polish_candidates=False)
    assert not c["certified"] and c["reason"] == "no feasible candidate"
