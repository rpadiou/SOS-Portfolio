"""Degree-4 portfolio risk polynomials and synthetic markets.

f(x) = x' Sigma x + sum_i eta_i x_i^3 + sum_{i,j} kappa_ij x_i^2 x_j^2
       + sum_{i,j} gamma_ij x_i^2 x_j^2

over K = {x >= 0, b_lo <= sum x <= b_hi}. The coefficients are those of a *model*:
kappa is not the fourth co-moment tensor of a portfolio, and the synthetic markets
below are not calibrated to data.

The gamma terms (named `impact_*` in the code) are a stylised concentration penalty on the
weights. They are quartic so that f has degree four, and they are larger on concentrated
positions. They are not calibrated and not derived from an execution model; the identifiers
keep the name `impact` for that reason only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .polynomial_ring import MultivariatePolynomial, Monomial

BUDGET_LOWER = 0.95
BUDGET_UPPER = 1.0

# 2-asset benchmark (effective f has a strictly convex Hessian on K: validation case only)
_SIGMA_2 = np.array([[0.04, -0.02], [-0.02, 0.09]])
_KURTOSIS_DIAG_2 = np.array([0.8, 1.2])
_KURTOSIS_CROSS_2 = 0.6
_SKEWNESS_2 = np.array([-0.3, 0.5])
_IMPACT_DIAG_2 = np.array([0.2, 0.15])
_IMPACT_CROSS_2 = 0.25


def build_objective() -> MultivariatePolynomial:
    """Objective of the 2-asset benchmark."""
    c: Dict[Monomial, float] = {
        (2, 0): _SIGMA_2[0, 0], (0, 2): _SIGMA_2[1, 1], (1, 1): 2 * _SIGMA_2[0, 1],
        (3, 0): _SKEWNESS_2[0], (0, 3): _SKEWNESS_2[1],
        (4, 0): _KURTOSIS_DIAG_2[0] + _IMPACT_DIAG_2[0],
        (0, 4): _KURTOSIS_DIAG_2[1] + _IMPACT_DIAG_2[1],
        (2, 2): 2 * _KURTOSIS_CROSS_2 + _IMPACT_CROSS_2,
    }
    return MultivariatePolynomial(2, c)


@dataclass
class SyntheticMarket:
    """Synthetic coefficients with a block (cluster) interaction graph.

    sigma is a covariance matrix; kurtosis_*/impact_* and skewness are the
    coefficients of the degree-3/4 terms. edge_set holds the pairs (i<j) that
    carry cross terms; the objective is correlatively sparse w.r.t. it.
    """

    n: int
    n_clusters: int
    cluster_size: int
    sigma: np.ndarray
    kurtosis_diag: np.ndarray
    kurtosis_cross: Dict[Tuple[int, int], float]
    skewness: np.ndarray
    impact_diag: np.ndarray
    impact_cross: Dict[Tuple[int, int], float]
    edge_set: List[Tuple[int, int]]
    budget_lower: float = BUDGET_LOWER
    budget_upper: float = BUDGET_UPPER

    @classmethod
    def generate(
        cls,
        n: int = 50,
        n_clusters: int = 10,
        seed: int = 0,
        budget_lower: float = BUDGET_LOWER,
        rho_within: float = 0.45,
        rho_noise: float = 0.05,
        vol_mean: float = 0.20,
        vol_std: float = 0.05,
        kurt_base: float = 0.5,
        skew_scale: float = 0.3,
        impact_base: float = 0.15,
    ) -> "SyntheticMarket":
        if n % n_clusters:
            raise ValueError("n must be divisible by n_clusters")
        rng = np.random.RandomState(seed)
        cs = n // n_clusters
        vols = np.clip(np.abs(rng.normal(vol_mean, vol_std, n)), 0.05, 0.60)
        corr = np.eye(n)
        edges: List[Tuple[int, int]] = []
        for c in range(n_clusters):
            for i in range(c * cs, (c + 1) * cs):
                for j in range(i + 1, (c + 1) * cs):
                    rho = float(np.clip(rho_within + rng.uniform(-rho_noise, rho_noise), 0.05, 0.95))
                    corr[i, j] = corr[j, i] = rho
                    edges.append((i, j))
        D = np.diag(vols)
        sigma = D @ corr @ D
        ev = np.linalg.eigvalsh(sigma)
        if ev.min() < 1e-6:
            sigma += (1e-4 - ev.min()) * np.eye(n)
        kd = kurt_base + rng.uniform(0.0, kurt_base, n)
        kc = {e: kurt_base * rng.uniform(0.2, 1.0) for e in edges}
        sign = rng.choice([-1.0, 1.0], size=n_clusters)
        skew = np.array([sign[i // cs] * skew_scale * rng.uniform(0.5, 1.5) for i in range(n)])
        idg = impact_base + rng.uniform(0.0, impact_base, n)
        ic = {e: impact_base * rng.uniform(0.1, 0.8) for e in edges}
        return cls(n, n_clusters, cs, sigma, kd, kc, skew, idg, ic, edges, budget_lower)

    def adjacency_matrix(self) -> np.ndarray:
        adj = np.zeros((self.n, self.n), dtype=bool)
        for i, j in self.edge_set:
            adj[i, j] = adj[j, i] = True
        return adj

    def kappa_terms(self) -> List[Tuple[Monomial, float]]:
        """Monomials multiplied by the uncertain kurtosis coefficients and their
        Frobenius weights: kappa is a symmetric matrix, so x_i^4 has weight 1 and
        x_i^2 x_j^2 (i<j) weight sqrt(2)."""
        n = self.n
        out: List[Tuple[Monomial, float]] = []
        for i in range(n):
            a = [0] * n
            a[i] = 4
            out.append((tuple(a), 1.0))
        for (i, j) in self.kurtosis_cross:
            a = [0] * n
            a[i] = a[j] = 2
            out.append((tuple(a), float(np.sqrt(2.0))))
        return out


def build_objective_from_market(market: SyntheticMarket) -> MultivariatePolynomial:
    """Degree-4 polynomial in x (only monomials supported in one cluster)."""
    n = market.n
    coeffs: Dict[Monomial, float] = {}

    def add(support: Dict[int, int], val: float) -> None:
        a = [0] * n
        for i, e in support.items():
            a[i] = e
        coeffs[tuple(a)] = coeffs.get(tuple(a), 0.0) + val

    for i in range(n):
        add({i: 2}, market.sigma[i, i])
        add({i: 3}, market.skewness[i])
        add({i: 4}, market.kurtosis_diag[i] + market.impact_diag[i])
    for i, j in market.edge_set:
        add({i: 1, j: 1}, 2.0 * market.sigma[i, j])
        add({i: 2, j: 2}, 2.0 * market.kurtosis_cross[(i, j)] + market.impact_cross[(i, j)])
    return MultivariatePolynomial(n, coeffs)


def is_feasible(x: np.ndarray, b_lo: float, b_hi: float, tol: float = 1e-9) -> bool:
    s = float(np.sum(x))
    return bool(np.all(x >= -tol) and b_lo - tol <= s <= b_hi + tol)


def get_feasible_grid(n_points: int = 100):
    x = np.linspace(0, 1, n_points)
    X1, X2 = np.meshgrid(x, x)
    return X1, X2, (X1 + X2 <= BUDGET_UPPER) & (X1 + X2 >= BUDGET_LOWER)


def evaluate_objective_grid(f: MultivariatePolynomial, n_points: int = 100):
    X1, X2, _ = get_feasible_grid(n_points)
    Z = np.array([[f(np.array([a, b])) for a, b in zip(r1, r2)] for r1, r2 in zip(X1, X2)])
    return X1, X2, Z


def make_nonconvex(
    n: int = 6,
    n_clusters: int = 2,
    seed: int = 0,
    min_minima: int = 2,
    skew_scales: Sequence[float] = (2.0, 3.0, 4.0, 6.0),
    kurt_base: float = 0.3,
    impact_base: float = 0.05,
    n_starts: int = 60,
) -> Tuple[SyntheticMarket, Dict]:
    """Stress-test instance with at least `min_minima` distinct local minima.

    Strong skewness and low kurtosis make the degree-3 terms dominate and the
    objective non-convex. The scale is increased along `skew_scales` until a
    multi-start run (`n_starts` SLSQP starts) finds `min_minima` distinct KKT
    points. These are *uncalibrated stress regimes*, not realistic markets.
    Raises RuntimeError if no scale in the grid qualifies.
    """
    from .local_solver import analyze_local_minima, scipy_optimize

    for sk in skew_scales:
        m = SyntheticMarket.generate(n, n_clusters, seed, skew_scale=sk, kurt_base=kurt_base,
                                     impact_base=impact_base)
        f = build_objective_from_market(m)
        out = scipy_optimize(f, n_starts, seed, m.budget_lower, m.budget_upper)
        mins = analyze_local_minima(out["results"])
        if mins["n_distinct_minima"] >= min_minima:
            return m, {"skew_scale": sk, "n_local_minima": mins["n_distinct_minima"],
                       "minima_values": [mm["f"] for mm in mins["local_minima"]]}
    raise RuntimeError(f"no skew_scale in {tuple(skew_scales)} gives {min_minima} local minima (seed={seed})")
