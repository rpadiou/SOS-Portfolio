import numpy as np
import pytest

from sos_portfolio import MultivariatePolynomial, generate_monomials


def test_monomial_count():
    assert len(generate_monomials(3, 2)) == 10
    assert len(generate_monomials(4, 4)) == 70


def test_eval_gradient_hessian_against_finite_differences():
    p = MultivariatePolynomial(3, {(2, 1, 0): 1.5, (0, 0, 4): -2.0, (1, 1, 1): 0.7, (0, 3, 0): 1.0, (0, 0, 0): 2.0})
    x = np.array([0.3, 0.6, 0.9])
    h = 1e-6
    g = np.array([(p(x + h * e) - p(x - h * e)) / (2 * h) for e in np.eye(3)])
    np.testing.assert_allclose(p.gradient(x), g, rtol=1e-6, atol=1e-8)
    H = np.array([(p.gradient(x + h * e) - p.gradient(x - h * e)) / (2 * h) for e in np.eye(3)])
    np.testing.assert_allclose(p.hessian(x), H, rtol=1e-5, atol=1e-7)


def test_ring_operations_and_embed():
    p = MultivariatePolynomial(2, {(1, 0): 1.0, (0, 1): 1.0})
    sq = p * p
    assert sq.coeffs == {(2, 0): 1.0, (1, 1): 2.0, (0, 2): 1.0}
    e = p.embed(4, [1, 3])
    assert e((0, 2.0, 0, 3.0)) == pytest.approx(5.0)
    assert sq.degree == 2 and p.support() == {0, 1}
