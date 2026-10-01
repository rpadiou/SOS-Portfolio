"""Sparse multivariate polynomials with numpy-backed evaluation."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

Monomial = Tuple[int, ...]
Coefficients = Dict[Monomial, float]


def generate_monomials(n_vars: int, degree: int) -> List[Monomial]:
    """All exponents alpha in N^n_vars with |alpha| <= degree, graded-lex order."""
    out: List[Monomial] = []

    def gen(n: int, remaining: int, current: List[int]) -> None:
        if n == 1:
            out.append(tuple(current + [remaining]))
            return
        for k in range(remaining + 1):
            current.append(k)
            gen(n - 1, remaining - k, current)
            current.pop()

    for total in range(degree + 1):
        gen(n_vars, total, [])
    return out


def eval_monomial(alpha: Monomial, x: np.ndarray) -> float:
    return float(np.prod(np.power(np.asarray(x, dtype=float), np.asarray(alpha))))


class MultivariatePolynomial:
    """p(x) = sum_alpha c_alpha x^alpha, stored as {alpha: c_alpha}."""

    def __init__(self, n_vars: int, coeffs: Optional[Coefficients] = None):
        self.n_vars = n_vars
        self.coeffs: Coefficients = {}
        for alpha, c in (coeffs or {}).items():
            if abs(c) > 1e-14:
                self.coeffs[tuple(int(a) for a in alpha)] = float(c)
        self._arrays: Optional[Tuple[np.ndarray, np.ndarray]] = None

    @property
    def degree(self) -> int:
        return max((sum(a) for a in self.coeffs), default=0)

    def support(self) -> set:
        """Indices of variables appearing in at least one monomial."""
        return {i for a in self.coeffs for i, e in enumerate(a) if e > 0}

    def _arr(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._arrays is None:
            if self.coeffs:
                E = np.array(list(self.coeffs.keys()), dtype=int)
                c = np.array(list(self.coeffs.values()), dtype=float)
            else:
                E = np.zeros((0, self.n_vars), dtype=int)
                c = np.zeros(0)
            self._arrays = (E, c)
        return self._arrays

    def __call__(self, x: np.ndarray) -> float:
        E, c = self._arr()
        x = np.asarray(x, dtype=float)
        return float(c @ np.prod(np.power(x[None, :], E), axis=1))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        E, c = self._arr()
        x = np.asarray(x, dtype=float)
        g = np.zeros(self.n_vars)
        for i in range(self.n_vars):
            m = E[:, i] > 0
            if not m.any():
                continue
            Ei = E[m].copy()
            w = c[m] * Ei[:, i]
            Ei[:, i] -= 1
            g[i] = w @ np.prod(np.power(x[None, :], Ei), axis=1)
        return g

    def hessian(self, x: np.ndarray) -> np.ndarray:
        E, c = self._arr()
        x = np.asarray(x, dtype=float)
        n = self.n_vars
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                Eij = E.copy()
                w = c * Eij[:, i]
                Eij[:, i] -= 1
                w = w * Eij[:, j]
                Eij[:, j] -= 1
                m = (Eij >= 0).all(axis=1) & (w != 0)
                if m.any():
                    H[i, j] = H[j, i] = w[m] @ np.prod(np.power(x[None, :], Eij[m]), axis=1)
        return H

    def __add__(self, other: "MultivariatePolynomial") -> "MultivariatePolynomial":
        out = dict(self.coeffs)
        for a, c in other.coeffs.items():
            out[a] = out.get(a, 0.0) + c
        return MultivariatePolynomial(self.n_vars, out)

    def __mul__(self, other) -> "MultivariatePolynomial":
        if isinstance(other, (int, float)):
            return MultivariatePolynomial(self.n_vars, {a: other * c for a, c in self.coeffs.items()})
        out: Coefficients = {}
        for a1, c1 in self.coeffs.items():
            for a2, c2 in other.coeffs.items():
                a = tuple(u + v for u, v in zip(a1, a2))
                out[a] = out.get(a, 0.0) + c1 * c2
        return MultivariatePolynomial(self.n_vars, out)

    __rmul__ = __mul__

    def embed(self, n_total: int, var_map: Optional[List[int]] = None) -> "MultivariatePolynomial":
        """Re-index variables: variable i becomes var_map[i] (default identity) in n_total variables."""
        var_map = list(range(self.n_vars)) if var_map is None else var_map
        out: Coefficients = {}
        for a, c in self.coeffs.items():
            b = [0] * n_total
            for i, e in enumerate(a):
                if e:
                    b[var_map[i]] = e
            out[tuple(b)] = c
        return MultivariatePolynomial(n_total, out)

    def __repr__(self) -> str:
        terms = []
        for a, c in sorted(self.coeffs.items(), key=lambda kv: (sum(kv[0]), kv[0])):
            m = " ".join(f"x{i + 1}^{e}" if e > 1 else f"x{i + 1}" for i, e in enumerate(a) if e) or "1"
            terms.append(f"{c:+.4g}*{m}")
        return " ".join(terms) if terms else "0"


def linear_poly(n_vars: int, coef: Dict[int, float], const: float = 0.0) -> MultivariatePolynomial:
    """const + sum_i coef[i] x_i."""
    d: Coefficients = {}
    if const:
        d[(0,) * n_vars] = const
    for i, c in coef.items():
        a = [0] * n_vars
        a[i] = 1
        d[tuple(a)] = d.get(tuple(a), 0.0) + c
    return MultivariatePolynomial(n_vars, d)
