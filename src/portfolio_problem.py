"""
Portfolio Optimization Problem — SOS-Portfolio
================================================

Defines degree-4 polynomial risk objectives for n-asset portfolios.

Supports two modes:
  - n=2 (demo): fixed parameters from the original toy model.
  - n≥3 (production): synthetic structured market data via SyntheticMarket.

Objective
---------
f(x) = x^T Σ x                               (Markowitz variance, deg 2)
     + Σ_i η_i x_i^3                          (skewness penalty, deg 3)
     + Σ_{(i,j)∈E} κ_{ij} x_i^2 x_j^2        (pairwise kurtosis, deg 4)
     + Σ_i κ_i x_i^4                          (excess kurtosis, deg 4)
     + Σ_{(i,j)∈E} γ_{ij} x_i^2 x_j^2        (Almgren-Chriss impact, deg 4)

E ⊆ {(i,j) : i<j} is the asset correlation graph (sparse for n large).

Feasible Set K (Semialgebraic)
------------------------------
g_i(x) = x_i ≥ 0          for i = 0,...,n-1   (no short-selling)
g_lo(x) = Σ x_i - ℓ ≥ 0                       (minimum investment)
g_hi(x) = 1 - Σ x_i ≥ 0                       (budget ceiling)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .polynomial_ring import MultivariatePolynomial, Monomial


# ── Legacy 2-asset parameters (backward compatibility) ─────────────────────

_SIGMA_2 = np.array([[0.04, -0.02], [-0.02, 0.09]])
_KURTOSIS_DIAG_2 = np.array([0.8, 1.2])
_KURTOSIS_CROSS_2 = 0.6
_SKEWNESS_2 = np.array([-0.3, 0.5])
_IMPACT_DIAG_2 = np.array([0.2, 0.15])
_IMPACT_CROSS_2 = 0.25
BUDGET_LOWER = 0.95


def build_objective() -> MultivariatePolynomial:
    """2-asset objective (original demo — backward compatible)."""
    return _build_objective_n2()


def build_constraints() -> List[MultivariatePolynomial]:
    """2-asset semialgebraic constraints (backward compatible)."""
    return _build_constraints_n(2, BUDGET_LOWER)


def _build_objective_n2() -> MultivariatePolynomial:
    coeffs: Dict[Monomial, float] = {}
    coeffs[(2, 0)] = _SIGMA_2[0, 0]
    coeffs[(0, 2)] = _SIGMA_2[1, 1]
    coeffs[(1, 1)] = 2.0 * _SIGMA_2[0, 1]
    coeffs[(3, 0)] = _SKEWNESS_2[0]
    coeffs[(0, 3)] = _SKEWNESS_2[1]
    coeffs[(4, 0)] = _KURTOSIS_DIAG_2[0] + _IMPACT_DIAG_2[0]
    coeffs[(0, 4)] = _KURTOSIS_DIAG_2[1] + _IMPACT_DIAG_2[1]
    coeffs[(2, 2)] = 2.0 * _KURTOSIS_CROSS_2 + _IMPACT_CROSS_2
    return MultivariatePolynomial(n_vars=2, coeffs=coeffs)


# ── Synthetic n-asset market ────────────────────────────────────────────────

@dataclass
class SyntheticMarket:
    """
    Synthetic structured market parameters for n assets.

    The correlation structure is block-diagonal: n_clusters clusters of
    cluster_size assets each, with within-cluster correlations drawn from
    [rho_min, rho_max] and near-zero inter-cluster correlations.

    Attributes
    ----------
    n : int
        Number of assets.
    n_clusters : int
        Number of industry clusters.
    cluster_size : int
        Assets per cluster (n = n_clusters × cluster_size).
    sigma : np.ndarray, shape (n, n)
        Covariance matrix.
    kurtosis_diag : np.ndarray, shape (n,)
        Diagonal kurtosis coefficients κ_i.
    kurtosis_cross : Dict[(int,int), float]
        Off-diagonal pairwise kurtosis for correlated pairs.
    skewness : np.ndarray, shape (n,)
        Skewness penalty coefficients η_i.
    impact_diag : np.ndarray, shape (n,)
        Diagonal Almgren-Chriss impact γ_i.
    impact_cross : Dict[(int,int), float]
        Pairwise impact for correlated pairs.
    edge_set : List[Tuple[int,int]]
        Edges (i,j) with i<j in the correlation graph.
    budget_lower : float
        Minimum total invested weight.
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
    budget_lower: float = 0.95

    @classmethod
    def generate(
        cls,
        n: int = 50,
        n_clusters: int = 10,
        seed: int = 0,
        budget_lower: float = 0.95,
        rho_within: float = 0.45,
        rho_noise: float = 0.05,
        vol_mean: float = 0.20,
        vol_std: float = 0.05,
        kurt_base: float = 0.5,
        skew_scale: float = 0.3,
        impact_base: float = 0.15,
    ) -> "SyntheticMarket":
        """
        Generate structured synthetic market data.

        Parameters
        ----------
        n : int
            Number of assets. Must be divisible by n_clusters.
        n_clusters : int
            Number of clusters (industries).
        seed : int
            NumPy random seed.
        rho_within : float
            Mean pairwise within-cluster correlation.
        rho_noise : float
            Noise on within-cluster correlations.
        vol_mean, vol_std : float
            Mean and std-dev of per-asset annualised volatility.
        kurt_base : float
            Base level of kurtosis coefficients.
        skew_scale : float
            Amplitude of skewness coefficients (signed).
        impact_base : float
            Base level of market impact coefficients.
        """
        assert n % n_clusters == 0, "n must be divisible by n_clusters"
        rng = np.random.RandomState(seed)
        cluster_size = n // n_clusters

        # Per-asset volatilities
        vols = np.abs(rng.normal(vol_mean, vol_std, n))
        vols = np.clip(vols, 0.05, 0.60)

        # Build block-diagonal correlation matrix
        corr = np.eye(n)
        edges: List[Tuple[int, int]] = []
        for c in range(n_clusters):
            start, end = c * cluster_size, (c + 1) * cluster_size
            for i in range(start, end):
                for j in range(i + 1, end):
                    rho = rho_within + rng.uniform(-rho_noise, rho_noise)
                    rho = float(np.clip(rho, 0.05, 0.95))
                    corr[i, j] = rho
                    corr[j, i] = rho
                    edges.append((i, j))

        # Covariance: Σ = D^{1/2} C D^{1/2}, D = diag(σ_i^2)
        D = np.diag(vols)
        sigma_raw = D @ corr @ D

        # Ensure positive definiteness by regularisation
        eigvals = np.linalg.eigvalsh(sigma_raw)
        if eigvals.min() < 1e-6:
            sigma_raw += (1e-4 - eigvals.min()) * np.eye(n)

        # Kurtosis: diagonal + within-cluster cross terms
        kurtosis_diag = kurt_base + rng.uniform(0.0, kurt_base, n)
        kurtosis_cross: Dict[Tuple[int, int], float] = {}
        for i, j in edges:
            kurtosis_cross[(i, j)] = kurt_base * rng.uniform(0.2, 1.0)

        # Skewness: signed, cluster-consistent
        cluster_sign = rng.choice([-1.0, 1.0], size=n_clusters)
        skewness = np.array([
            cluster_sign[i // cluster_size] * skew_scale * rng.uniform(0.5, 1.5)
            for i in range(n)
        ])

        # Impact: diagonal + within-cluster cross
        impact_diag = impact_base + rng.uniform(0.0, impact_base, n)
        impact_cross: Dict[Tuple[int, int], float] = {}
        for i, j in edges:
            impact_cross[(i, j)] = impact_base * rng.uniform(0.1, 0.8)

        return cls(
            n=n,
            n_clusters=n_clusters,
            cluster_size=cluster_size,
            sigma=sigma_raw,
            kurtosis_diag=kurtosis_diag,
            kurtosis_cross=kurtosis_cross,
            skewness=skewness,
            impact_diag=impact_diag,
            impact_cross=impact_cross,
            edge_set=edges,
            budget_lower=budget_lower,
        )

    def adjacency_matrix(self) -> np.ndarray:
        """Binary adjacency from the edge set."""
        adj = np.zeros((self.n, self.n), dtype=bool)
        for i, j in self.edge_set:
            adj[i, j] = True
            adj[j, i] = True
        return adj


# ── Polynomial builders ─────────────────────────────────────────────────────

def build_objective_from_market(market: SyntheticMarket) -> MultivariatePolynomial:
    """
    Construct the sparse degree-4 polynomial objective from market data.

    Sparsity pattern: only monomials whose support lies within a single
    cluster are non-zero, ensuring each term appears in at least one clique.

    Parameters
    ----------
    market : SyntheticMarket

    Returns
    -------
    MultivariatePolynomial
        f ∈ R[x_0,...,x_{n-1}] of degree 4.
    """
    n = market.n
    coeffs: Dict[Monomial, float] = {}

    def add(alpha: Monomial, val: float) -> None:
        if abs(val) > 1e-14:
            coeffs[alpha] = coeffs.get(alpha, 0.0) + val

    zero_n = [0] * n

    # ── Variance: σ_{ij} x_i x_j ────────────────────────────────────────────
    for i in range(n):
        a = list(zero_n); a[i] = 2
        add(tuple(a), market.sigma[i, i])
    for i, j in market.edge_set:
        a = list(zero_n); a[i] = 1; a[j] = 1
        add(tuple(a), 2.0 * market.sigma[i, j])

    # ── Skewness: η_i x_i^3 ─────────────────────────────────────────────────
    for i in range(n):
        a = list(zero_n); a[i] = 3
        add(tuple(a), market.skewness[i])

    # ── Diagonal kurtosis: κ_i x_i^4 ────────────────────────────────────────
    for i in range(n):
        a = list(zero_n); a[i] = 4
        add(tuple(a), market.kurtosis_diag[i])

    # ── Pairwise kurtosis: κ_{ij} x_i^2 x_j^2 ──────────────────────────────
    for (i, j), k_ij in market.kurtosis_cross.items():
        a = list(zero_n); a[i] = 2; a[j] = 2
        add(tuple(a), 2.0 * k_ij)

    # ── Diagonal impact: γ_i x_i^4 ──────────────────────────────────────────
    for i in range(n):
        a = list(zero_n); a[i] = 4
        add(tuple(a), market.impact_diag[i])

    # ── Pairwise impact: γ_{ij} x_i^2 x_j^2 ────────────────────────────────
    for (i, j), g_ij in market.impact_cross.items():
        a = list(zero_n); a[i] = 2; a[j] = 2
        add(tuple(a), g_ij)

    return MultivariatePolynomial(n_vars=n, coeffs=coeffs)


def build_constraints_from_market(
    market: SyntheticMarket,
) -> List[MultivariatePolynomial]:
    """
    Build the semialgebraic constraint polynomials for n assets.

    Returns n+2 constraints:
      - g_i = x_i ≥ 0 for i = 0,...,n-1
      - g_lo = Σ x_i - budget_lower ≥ 0
      - g_hi = 1 - Σ x_i ≥ 0

    Parameters
    ----------
    market : SyntheticMarket

    Returns
    -------
    List[MultivariatePolynomial]
        Constraint polynomials.
    """
    return _build_constraints_n(market.n, market.budget_lower)


def _build_constraints_n(
    n: int, budget_lower: float
) -> List[MultivariatePolynomial]:
    zero_n = tuple([0] * n)
    constraints: List[MultivariatePolynomial] = []

    # g_i = x_i ≥ 0
    for i in range(n):
        alpha: List[int] = [0] * n
        alpha[i] = 1
        constraints.append(MultivariatePolynomial(n, {tuple(alpha): 1.0}))

    # g_lo = Σ x_i - budget_lower ≥ 0
    lo_coeffs: Dict[Monomial, float] = {zero_n: -budget_lower}
    for i in range(n):
        alpha = [0] * n
        alpha[i] = 1
        lo_coeffs[tuple(alpha)] = 1.0
    constraints.append(MultivariatePolynomial(n, lo_coeffs))

    # g_hi = 1 - Σ x_i ≥ 0
    hi_coeffs: Dict[Monomial, float] = {zero_n: 1.0}
    for i in range(n):
        alpha = [0] * n
        alpha[i] = 1
        hi_coeffs[tuple(alpha)] = hi_coeffs.get(tuple(alpha), 0.0) - 1.0
    constraints.append(MultivariatePolynomial(n, hi_coeffs))

    return constraints


def get_feasible_grid(n_points: int = 100):
    """2-asset feasible grid (backward compatible with visualization)."""
    x1 = np.linspace(0, 1, n_points)
    x2 = np.linspace(0, 1, n_points)
    X1, X2 = np.meshgrid(x1, x2)
    mask = (X1 >= 0) & (X2 >= 0) & (X1 + X2 <= 1.0) & (X1 + X2 >= BUDGET_LOWER)
    return X1, X2, mask


def evaluate_objective_grid(f: MultivariatePolynomial, n_points: int = 100):
    """2-asset objective grid evaluation (backward compatible)."""
    X1, X2, _ = get_feasible_grid(n_points)
    Z = np.zeros_like(X1)
    for i in range(n_points):
        for j in range(n_points):
            Z[i, j] = f(np.array([X1[i, j], X2[i, j]]))
    return X1, X2, Z
