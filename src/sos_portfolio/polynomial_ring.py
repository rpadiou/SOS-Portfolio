"""
Polynomial Ring Module — SOS-Portfolio
=======================================

Implements the algebraic engine for multivariate polynomials in the ring
R[x_1, ..., x_n], providing the foundation for the Lasserre SOS hierarchy.

Mathematical Framework
----------------------
A polynomial p ∈ R[x_1, ..., x_n] of degree d is represented as:

    p(x) = Σ_{|α| ≤ d} c_α · x^α

where α = (α_1, ..., α_n) ∈ N^n is a multi-index, |α| = Σ α_i,
and x^α = x_1^{α_1} · ... · x_n^{α_n}.

Gram (SOS) Representation
--------------------------
A polynomial p is SOS (Sum-of-Squares) if and only if there exists a PSD
matrix Q ≽ 0 such that:

    p(x) = v(x)^T Q v(x)

where v(x) is the vector of monomials up to degree ⌊deg(p)/2⌋.
"""

import numpy as np
from itertools import product
from typing import Dict, List, Tuple, Optional
from functools import lru_cache


# Type alias: multi-index → coefficient
Monomial = Tuple[int, ...]
Coefficients = Dict[Monomial, float]


def generate_monomials(n_vars: int, degree: int) -> List[Monomial]:
    """
    Enumerate all monomials x^α with |α| ≤ degree in n_vars variables.

    Produces the canonical monomial basis of the truncated polynomial space:

        M_{n,d} = { α ∈ N^n : |α| = Σ α_i ≤ d }

    The dimension is C(n + d, d) = (n+d)! / (n! d!).

    Uses a recursive composition generator — O(output size) not O((d+1)^n).
    """
    monomials: List[Monomial] = []

    def _gen(n: int, remaining: int, current: List[int]) -> None:
        if n == 1:
            monomials.append(tuple(current + [remaining]))
            return
        for k in range(remaining + 1):
            current.append(k)
            _gen(n - 1, remaining - k, current)
            current.pop()

    for total_deg in range(degree + 1):
        _gen(n_vars, total_deg, [])
    return monomials


def eval_monomial(alpha: Monomial, x: np.ndarray) -> float:
    """
    Evaluate the monomial x^α = x_1^{α_1} · ... · x_n^{α_n}.

    Parameters
    ----------
    alpha : Monomial
        Multi-index (α_1, ..., α_n).
    x : np.ndarray
        Point in R^n.

    Returns
    -------
    float
        Value x^α.
    """
    return float(np.prod([xi ** ai for xi, ai in zip(x, alpha)]))


def eval_monomial_vector(monomials: List[Monomial], x: np.ndarray) -> np.ndarray:
    """
    Evaluate the monomial vector v(x) = [x^{α_1}, x^{α_2}, ..., x^{α_s}]^T.

    Parameters
    ----------
    monomials : List[Monomial]
        Ordered list of multi-indices.
    x : np.ndarray
        Evaluation point in R^n.

    Returns
    -------
    np.ndarray
        Vector v(x) ∈ R^s.
    """
    return np.array([eval_monomial(alpha, x) for alpha in monomials])


class MultivariatePolynomial:
    """
    Represents an element of the polynomial ring R[x_1, ..., x_n].

    A polynomial is stored as a sparse dictionary mapping multi-indices
    to real coefficients:  p = { α : c_α | c_α ≠ 0 }

    Supports ring operations (+, -, *) and evaluation.

    Attributes
    ----------
    n_vars : int
        Number of variables n.
    coeffs : Coefficients
        Sparse coefficient dictionary { α → c_α }.
    degree : int
        Total degree deg(p) = max{ |α| : c_α ≠ 0 }.
    """

    def __init__(self, n_vars: int, coeffs: Optional[Coefficients] = None):
        """
        Construct p ∈ R[x_1, ..., x_n].

        Parameters
        ----------
        n_vars : int
            Ambient dimension n.
        coeffs : dict, optional
            Coefficient dictionary { α → c_α }.
        """
        self.n_vars = n_vars
        self.coeffs: Coefficients = {}
        if coeffs:
            for alpha, c in coeffs.items():
                if abs(c) > 1e-14:
                    self.coeffs[tuple(alpha)] = float(c)

    @property
    def degree(self) -> int:
        """Total degree: deg(p) = max |α| over all terms with c_α ≠ 0."""
        if not self.coeffs:
            return 0
        return max(sum(alpha) for alpha in self.coeffs)

    def __call__(self, x: np.ndarray) -> float:
        """
        Evaluate p at point x ∈ R^n.

        p(x) = Σ_α c_α · x^α

        Parameters
        ----------
        x : np.ndarray
            Evaluation point, shape (n,).

        Returns
        -------
        float
            Scalar value p(x).
        """
        return sum(c * eval_monomial(alpha, x) for alpha, c in self.coeffs.items())

    def __add__(self, other: "MultivariatePolynomial") -> "MultivariatePolynomial":
        result = MultivariatePolynomial(self.n_vars, dict(self.coeffs))
        for alpha, c in other.coeffs.items():
            result.coeffs[alpha] = result.coeffs.get(alpha, 0.0) + c
            if abs(result.coeffs[alpha]) < 1e-14:
                del result.coeffs[alpha]
        return result

    def __mul__(self, other) -> "MultivariatePolynomial":
        if isinstance(other, (int, float)):
            return MultivariatePolynomial(
                self.n_vars, {alpha: other * c for alpha, c in self.coeffs.items()}
            )
        result_coeffs: Coefficients = {}
        for a1, c1 in self.coeffs.items():
            for a2, c2 in other.coeffs.items():
                alpha = tuple(a1[i] + a2[i] for i in range(self.n_vars))
                result_coeffs[alpha] = result_coeffs.get(alpha, 0.0) + c1 * c2
        return MultivariatePolynomial(self.n_vars, result_coeffs)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the gradient ∇p(x) ∈ R^n.

        ∂p/∂x_i = Σ_α c_α · α_i · x^{α - e_i}

        Parameters
        ----------
        x : np.ndarray
            Evaluation point.

        Returns
        -------
        np.ndarray
            Gradient vector ∇p(x).
        """
        grad = np.zeros(self.n_vars)
        for alpha, c in self.coeffs.items():
            for i in range(self.n_vars):
                if alpha[i] > 0:
                    new_alpha = list(alpha)
                    new_alpha[i] -= 1
                    grad[i] += c * alpha[i] * eval_monomial(tuple(new_alpha), x)
        return grad

    def __repr__(self) -> str:
        terms = []
        for alpha, c in sorted(self.coeffs.items(), key=lambda kv: (sum(kv[0]), kv[0])):
            mono_str = " ".join(f"x{i+1}^{a}" if a > 1 else f"x{i+1}"
                                for i, a in enumerate(alpha) if a > 0) or "1"
            terms.append(f"{c:+.4f}·{mono_str}")
        return " ".join(terms) if terms else "0"


def gram_matrix_to_polynomial(Q: np.ndarray, monomials: List[Monomial],
                               n_vars: int) -> MultivariatePolynomial:
    """
    Reconstruct the polynomial p from its Gram decomposition.

    Given a PSD matrix Q and monomial vector v(x), compute:

        p(x) = v(x)^T Q v(x) = Σ_{i,j} Q_{ij} · x^{α_i + α_j}

    This is the fundamental link between the algebraic SOS certificate
    and its polynomial representation.

    Parameters
    ----------
    Q : np.ndarray
        Gram matrix, shape (s, s), must be PSD for p to be SOS.
    monomials : List[Monomial]
        Monomial basis v(x).
    n_vars : int
        Number of variables.

    Returns
    -------
    MultivariatePolynomial
        The polynomial p = v^T Q v.
    """
    coeffs: Coefficients = {}
    s = len(monomials)
    for i in range(s):
        for j in range(s):
            if abs(Q[i, j]) > 1e-14:
                alpha = tuple(monomials[i][k] + monomials[j][k] for k in range(n_vars))
                coeffs[alpha] = coeffs.get(alpha, 0.0) + Q[i, j]
    return MultivariatePolynomial(n_vars, coeffs)
