"""
Local Solver Baseline — SOS-Portfolio
=======================================

L-BFGS-B local optimisation baseline. Supports both the 2-asset toy model
(backward compatible) and n-asset portfolios.

The Failure Mode
-----------------
For non-convex degree-4 objectives, L-BFGS-B satisfies first-order KKT
conditions at a stationary point but cannot certify global optimality.
Multiple restarts provide a probabilistic upper bound f_UB ≥ f*.
The SOS hierarchy provides a certified lower bound λ_d ≤ f*.
Together: λ_d ≤ f* ≤ f_UB with certified gap = f_UB - λ_d.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import LinearConstraint, minimize, OptimizeResult
from typing import Dict, List, Optional, Tuple

from .polynomial_ring import MultivariatePolynomial


def scipy_optimize(
    f: MultivariatePolynomial,
    n_restarts: int = 50,
    seed: int = 42,
    budget_lower: float = 0.95,
) -> Dict:
    """
    Minimise f over the portfolio simplex using L-BFGS-B with SLSQP fallback.

    Supports arbitrary n ≥ 2. The budget constraint Σ x_i ∈ [budget_lower, 1]
    is handled via a penalty term (L-BFGS-B box-only) or linear constraints
    (SLSQP). Both are run; the better result is returned.

    Parameters
    ----------
    f : MultivariatePolynomial
        Objective polynomial of degree 4.
    n_restarts : int
        Number of random initialisations.
    seed : int
        Random seed.
    budget_lower : float
        Minimum total invested weight Σ x_i ≥ budget_lower.

    Returns
    -------
    Dict with keys:
        x_opt, f_opt, all_results, n_restarts, n_vars
    """
    n = f.n_vars
    rng = np.random.RandomState(seed)
    PENALTY = 1e5

    def penalised_obj(x: np.ndarray) -> float:
        val = f(x)
        s = float(np.sum(x))
        val += PENALTY * max(0.0, budget_lower - s) ** 2
        val += PENALTY * max(0.0, s - 1.0) ** 2
        return val

    def penalised_grad(x: np.ndarray) -> np.ndarray:
        grad = f.gradient(x)
        s = float(np.sum(x))
        if s < budget_lower:
            dpb = 2.0 * PENALTY * (budget_lower - s)
            grad -= dpb * np.ones(n)
        elif s > 1.0:
            dpb = 2.0 * PENALTY * (s - 1.0)
            grad += dpb * np.ones(n)
        return grad

    bounds_lbfgs = [(0.0, 1.0)] * n

    # SLSQP constraints (exact)
    lin_cons = LinearConstraint(
        np.ones((1, n)),
        lb=np.array([budget_lower]),
        ub=np.array([1.0]),
    )
    bounds_slsqp = [(0.0, 1.0)] * n

    all_results: List[Dict] = []
    best_val = np.inf
    best_x: Optional[np.ndarray] = None

    for i in range(n_restarts):
        # Dirichlet sample on simplex then scale to [budget_lower, 1]
        raw = rng.dirichlet(np.ones(n))
        target = rng.uniform(budget_lower, 1.0)
        x0 = raw * target
        x0 = np.clip(x0, 0.0, 1.0)

        # L-BFGS-B with penalty
        res_l = minimize(
            fun=penalised_obj,
            x0=x0,
            jac=penalised_grad,
            method="L-BFGS-B",
            bounds=bounds_lbfgs,
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
        )
        # Project to feasibility
        x_l = np.clip(res_l.x, 0.0, 1.0)
        s_l = float(np.sum(x_l))
        if s_l > 1e-9:
            x_l = x_l * np.clip(s_l, budget_lower, 1.0) / s_l

        f_l = f(x_l)

        # SLSQP with exact constraints (fewer iterations, but higher quality)
        res_s = minimize(
            fun=f,
            x0=x0,
            jac=f.gradient,
            method="SLSQP",
            bounds=bounds_slsqp,
            constraints=lin_cons,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        x_s = np.clip(res_s.x, 0.0, 1.0)
        f_s = f(x_s)

        # Keep the better of L-BFGS-B and SLSQP
        if f_l <= f_s:
            x_best, f_best = x_l, f_l
        else:
            x_best, f_best = x_s, f_s

        all_results.append({"x": x_best.copy(), "f": f_best, "success": True})

        if f_best < best_val:
            best_val = f_best
            best_x = x_best.copy()

    return {
        "x_opt": best_x,
        "f_opt": best_val,
        "all_results": all_results,
        "n_restarts": n_restarts,
        "n_vars": n,
    }


def analyze_local_minima(
    all_results: List[Dict], tol: float = 1e-3
) -> Dict:
    """
    Cluster L-BFGS-B results to identify distinct local minima.

    Two results are the same local minimum if ‖x_a - x_b‖₂ < tol.

    Parameters
    ----------
    all_results : List[Dict]
        From scipy_optimize['all_results'].
    tol : float
        Distance threshold for clustering.

    Returns
    -------
    Dict with keys: n_distinct_minima, local_minima (sorted by f value).
    """
    minima: List[Dict] = []

    for res in all_results:
        found = False
        for m in minima:
            if np.linalg.norm(res["x"] - m["x"]) < tol:
                m["count"] += 1
                if res["f"] < m["f"]:
                    m["f"] = res["f"]
                    m["x"] = res["x"].copy()
                found = True
                break
        if not found:
            minima.append({"x": res["x"].copy(), "f": res["f"], "count": 1})

    minima.sort(key=lambda m: m["f"])
    return {"n_distinct_minima": len(minima), "local_minima": minima}
