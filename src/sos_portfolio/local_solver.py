"""Multi-start local solver (SLSQP with exact linear constraints): upper bounds on f*.

A point only counts as an upper bound if it satisfies the constraints to `feas_tol`.
A local run is reported as a KKT point only if the projected-gradient residual
||x - P_K(x - grad f(x))|| is below `kkt_tol`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from .polynomial_ring import MultivariatePolynomial


def project_to_K(z: np.ndarray, b_lo: float, b_hi: float) -> np.ndarray:
    """Euclidean projection onto {x >= 0, b_lo <= sum x <= b_hi} (bisection on the shift)."""
    x = np.maximum(z, 0.0)
    s = x.sum()
    if b_lo <= s <= b_hi:
        return x
    target = b_lo if s < b_lo else b_hi
    lo, hi = float(z.min() - target - 1.0), float(z.max())
    for _ in range(200):
        tau = 0.5 * (lo + hi)
        if np.maximum(z - tau, 0.0).sum() > target:
            lo = tau
        else:
            hi = tau
    return np.maximum(z - 0.5 * (lo + hi), 0.0)


def kkt_residual(f: MultivariatePolynomial, x: np.ndarray, b_lo: float, b_hi: float) -> float:
    return float(np.linalg.norm(x - project_to_K(x - f.gradient(x), b_lo, b_hi)))


def feasibility_violation(x: np.ndarray, b_lo: float, b_hi: float) -> float:
    s = float(np.sum(x))
    return float(max(0.0, -x.min(), b_lo - s, s - b_hi))


def make_starts(n: int, n_starts: int, b_lo: float, b_hi: float, rng: np.random.RandomState) -> List[np.ndarray]:
    """1/N, simplex vertices, then Dirichlet samples with random total weight."""
    starts = [np.full(n, 0.5 * (b_lo + b_hi) / n)]
    starts += [b_hi * np.eye(n)[i] for i in range(min(n, n_starts // 4))]
    while len(starts) < n_starts:
        starts.append(rng.dirichlet(np.ones(n)) * rng.uniform(b_lo, b_hi))
    return starts[:n_starts]


def scipy_optimize(
    f: MultivariatePolynomial,
    n_starts: int = 50,
    seed: int = 42,
    b_lo: float = 0.95,
    b_hi: float = 1.0,
    n_perturb: int = 0,
    perturb_scale: float = 0.05,
    feas_tol: float = 1e-8,
    kkt_tol: float = 1e-5,
    starts: Optional[Sequence[np.ndarray]] = None,
    maxiter: int = 500,
) -> Dict:
    """Run SLSQP from several starts; `n_perturb` extra starts are perturbations of the best point.

    Returns a dict with `results` (one entry per run: x, f, feasible, converged,
    n_iter, kkt_res, is_kkt), and `x_opt`/`f_opt` = best *feasible* run (None if none).
    """
    n = f.n_vars
    rng = np.random.RandomState(seed)
    A = LinearConstraint(np.ones((1, n)), lb=b_lo, ub=b_hi)
    bounds = [(0.0, 1.0)] * n
    results: List[Dict] = []

    def run(x0: np.ndarray) -> Dict:
        r = minimize(f, x0, jac=f.gradient, method="SLSQP", bounds=bounds, constraints=A,
                     options={"maxiter": maxiter, "ftol": 1e-14})
        x = np.clip(r.x, 0.0, 1.0)
        viol = feasibility_violation(x, b_lo, b_hi)
        res = kkt_residual(f, x, b_lo, b_hi)
        return {"x": x, "f": float(f(x)), "feasible": viol <= feas_tol, "violation": viol,
                "converged": bool(r.success), "n_iter": int(r.nit), "kkt_res": res,
                "is_kkt": bool(res <= kkt_tol and viol <= feas_tol)}

    for x0 in (starts if starts is not None else make_starts(n, n_starts, b_lo, b_hi, rng)):
        results.append(run(np.asarray(x0, dtype=float)))

    def best() -> Optional[Dict]:
        feas = [r for r in results if r["feasible"]]
        return min(feas, key=lambda r: r["f"]) if feas else None

    for _ in range(n_perturb):
        b = best()
        if b is None:
            break
        x0 = project_to_K(b["x"] + perturb_scale * rng.randn(n), b_lo, b_hi)
        results.append(run(x0))

    b = best()
    return {"results": results, "x_opt": None if b is None else b["x"],
            "f_opt": None if b is None else b["f"], "n_starts": len(results), "n_vars": n}


def analyze_local_minima(results: Sequence[Dict], tol_f: float = 1e-6, tol_x: float = 5e-3) -> Dict:
    """Cluster KKT runs: same minimum iff |df| <= tol_f*max(1,|f|) and ||dx||_inf <= tol_x."""
    minima: List[Dict] = []
    for r in results:
        if not r.get("is_kkt"):
            continue
        for m in minima:
            if abs(r["f"] - m["f"]) <= tol_f * max(1.0, abs(m["f"])) and np.max(np.abs(r["x"] - m["x"])) <= tol_x:
                m["count"] += 1
                if r["f"] < m["f"]:
                    m["f"], m["x"] = r["f"], r["x"].copy()
                break
        else:
            minima.append({"x": r["x"].copy(), "f": r["f"], "count": 1})
    minima.sort(key=lambda m: m["f"])
    return {"n_distinct_minima": len(minima), "local_minima": minima}


def polish(f: MultivariatePolynomial, x0: np.ndarray, b_lo: float, b_hi: float, feas_tol: float = 1e-8) -> np.ndarray:
    """Local refinement (SLSQP) of a candidate; returns x0 projected onto K if the run is infeasible."""
    x0 = project_to_K(np.asarray(x0, dtype=float), b_lo, b_hi)
    out = scipy_optimize(f, b_lo=b_lo, b_hi=b_hi, starts=[x0], feas_tol=feas_tol)
    return out["x_opt"] if out["x_opt"] is not None and out["f_opt"] <= f(x0) else x0
