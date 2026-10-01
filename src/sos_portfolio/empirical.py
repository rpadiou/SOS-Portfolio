"""Real-data utilities for Experiment C: estimated co-moment polynomial, baseline allocators, walk-forward
metrics and inference. See experiments/PROTOCOL_C.md.

Optional dependencies (extra `data`): pandas, scikit-learn.
"""

from __future__ import annotations

from math import factorial
from typing import Dict, List, Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from .polynomial_ring import MultivariatePolynomial, generate_monomials

SIGNS = {2: 1.0, 3: -1.0, 4: 1.0}     # F = l2 m2 - l3 m3 + l4 m4


# model
def prepare_window(R: np.ndarray) -> np.ndarray:
    """Centre the window and rescale so the average asset volatility is 1."""
    Rc = R - R.mean(axis=0, keepdims=True)
    return Rc / Rc.std(axis=0).mean()


def blocks_of(n: int, sectors: Optional[Sequence[int]]) -> List[List[int]]:
    if sectors is None:
        return [list(range(n))]
    sectors = np.asarray(sectors)
    return [list(np.where(sectors == s)[0]) for s in np.unique(sectors)]


def model_moments(x: np.ndarray, Rc: np.ndarray, blocks: Sequence[Sequence[int]]) -> Dict[int, float]:
    """m_k(x) summed over blocks (block b uses only the weights of its own assets)."""
    out = {k: 0.0 for k in (2, 3, 4)}
    for b in blocks:
        p = Rc[:, b] @ x[b]
        for k in out:
            out[k] += float(np.mean(p ** k))
    return out


def lambdas(Rc: np.ndarray, blocks, w: Tuple[float, float, float]) -> Dict[int, float]:
    """l_k = w_k / mean_i |m_k(e_i)|: each term contributes w_k on a single-asset portfolio (average over assets).

    (Normalising at equal weights, as first planned, divides by a diversified third moment that is close to
    zero and gives coefficients spanning five orders of magnitude; the asset-level scale is well conditioned.)
    """
    n = Rc.shape[1]
    m = {k: np.mean([abs(model_moments(np.eye(n)[i], Rc, blocks)[k]) for i in range(n)]) for k in (2, 3, 4)}
    return {k: (w[i] / max(m[k], 1e-300)) for i, k in enumerate((2, 3, 4))}


def objective(x: np.ndarray, Rc, blocks, lam) -> Tuple[float, np.ndarray]:
    """F(x) and its gradient, straight from the returns (no monomial expansion)."""
    val, g = 0.0, np.zeros_like(x)
    for b in blocks:
        Rb = Rc[:, b]
        p = Rb @ x[b]
        for k in (2, 3, 4):
            c = SIGNS[k] * lam[k]
            val += c * float(np.mean(p ** k))
            g[b] += c * k * (Rb.T @ (p ** (k - 1))) / len(p)
    return val, g


def polynomial(Rc: np.ndarray, blocks, lam) -> MultivariatePolynomial:
    """Monomial expansion of F: coefficient of x^alpha is sign * l_k * k!/prod(alpha!) * mean_t prod_i r_ti^alpha_i."""
    n = Rc.shape[1]
    coeffs: Dict[Tuple[int, ...], float] = {}
    for b in blocks:
        Rb = Rc[:, b]
        for a in generate_monomials(len(b), 4):
            k = sum(a)
            if k < 2:
                continue
            mult = factorial(k) / np.prod([factorial(e) for e in a])
            mean = float(np.mean(np.prod(Rb ** np.array(a)[None, :], axis=1)))
            full = [0] * n
            for j, e in enumerate(a):
                full[b[j]] = e
            coeffs[tuple(full)] = coeffs.get(tuple(full), 0.0) + SIGNS[k] * lam[k] * mult * mean
    return MultivariatePolynomial(n, coeffs)


def solve_local(Rc, blocks, lam, n_starts: int = 20, seed: int = 0, cap: float = 1.0) -> np.ndarray:
    """Multi-start SLSQP for min F over {0 <= x <= cap, sum x = 1}; returns the best feasible point."""
    n = Rc.shape[1]
    rng = np.random.RandomState(seed)
    starts = [np.full(n, 1.0 / n)] + [rng.dirichlet(np.ones(n)) for _ in range(n_starts - 1)]
    best, bx = np.inf, np.full(n, 1.0 / n)
    for x0 in starts:
        r = minimize(lambda x: objective(x, Rc, blocks, lam), x0, jac=True, method="SLSQP", bounds=[(0, cap)] * n,
                     constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1.0, "jac": lambda x: np.ones(n)}],
                     options={"maxiter": 300, "ftol": 1e-12})
        x = np.clip(r.x, 0, cap)
        x = x / x.sum()
        if x.max() > cap + 1e-9:
            continue
        v = objective(x, Rc, blocks, lam)[0]
        if v < best:
            best, bx = v, x
    return bx


# baselines
def ledoit_wolf_cov(R: np.ndarray) -> np.ndarray:
    from sklearn.covariance import LedoitWolf
    return LedoitWolf().fit(R).covariance_


def min_variance(R: np.ndarray, cap: float = 1.0) -> np.ndarray:
    S = ledoit_wolf_cov(R)
    n = S.shape[0]
    x = cp.Variable(n)
    cp.Problem(cp.Minimize(cp.quad_form(x, cp.psd_wrap(S * 1e4))), [x >= 0, x <= cap, cp.sum(x) == 1]).solve(solver=cp.CLARABEL)
    w = np.clip(x.value, 0, None)
    return w / w.sum()


def erc(R: np.ndarray) -> np.ndarray:
    """Equal risk contribution (long-only): min 0.5 x'Sx - (1/n) sum log x, then normalise."""
    S = ledoit_wolf_cov(R) * 1e4
    n = S.shape[0]
    y = cp.Variable(n, pos=True)
    cp.Problem(cp.Minimize(0.5 * cp.quad_form(y, cp.psd_wrap(S)) - cp.sum(cp.log(y)) / n)).solve(solver=cp.CLARABEL)
    w = np.asarray(y.value)
    return w / w.sum()


def min_cvar(R: np.ndarray, alpha: float = 0.95, cap: float = 1.0) -> np.ndarray:
    """Rockafellar-Uryasev LP on the window scenarios."""
    T, n = R.shape
    x, z, u = cp.Variable(n), cp.Variable(), cp.Variable(T)
    cons = [x >= 0, x <= cap, cp.sum(x) == 1, u >= 0, u >= -R @ x - z]
    cp.Problem(cp.Minimize(z + cp.sum(u) / ((1 - alpha) * T)), cons).solve(solver=cp.CLARABEL)
    w = np.clip(x.value, 0, None)
    return w / w.sum()


def hrp(R: np.ndarray) -> np.ndarray:
    """Hierarchical Risk Parity (Lopez de Prado 2016): single-linkage order, recursive bisection on inverse variance."""
    C = np.corrcoef(R, rowvar=False)
    S = np.cov(R, rowvar=False)
    D = np.sqrt(np.clip(0.5 * (1 - C), 0, None))
    order = list(leaves_list(linkage(squareform(D, checks=False), method="single")))
    w = np.ones(S.shape[0])
    groups = [order]
    while groups:
        g = groups.pop()
        if len(g) < 2:
            continue
        h = len(g) // 2
        a, b = g[:h], g[h:]

        def cvar(ix):
            Sx = S[np.ix_(ix, ix)]
            iv = 1.0 / np.diag(Sx)
            iv /= iv.sum()
            return float(iv @ Sx @ iv)

        va, vb = cvar(a), cvar(b)
        fa = 1.0 - va / (va + vb)
        w[a] *= fa
        w[b] *= 1.0 - fa
        groups += [a, b]
    return w / w.sum()


# performance and inference
def performance(r_net: np.ndarray, r_gross: np.ndarray) -> Dict[str, float]:
    ann = 252
    cum = np.cumsum(r_net)
    dd = np.max(np.maximum.accumulate(cum) - cum)
    q = np.quantile(r_gross, 0.05)
    mu, sd = r_net.mean(), r_net.std(ddof=1)
    z = (r_gross - r_gross.mean()) / r_gross.std(ddof=1)
    return {"ann_vol": float(r_gross.std(ddof=1) * np.sqrt(ann)), "cvar5": float(-r_gross[r_gross <= q].mean()),
            "max_drawdown": float(dd), "skew": float(np.mean(z ** 3)), "ann_return_net": float(mu * ann),
            "sharpe_net": float(mu / sd * np.sqrt(ann))}


def circular_block_indices(T: int, block: int, rng: np.random.RandomState) -> np.ndarray:
    n_blocks = int(np.ceil(T / block))
    starts = rng.randint(0, T, n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]) % T
    return idx.ravel()[:T]


def block_bootstrap_diff(a: np.ndarray, b: np.ndarray, stat, block: int = 21, n: int = 2000, seed: int = 0):
    """stat(a) - stat(b) with a circular-block-bootstrap 95% interval (same resampled dates for both series)."""
    rng = np.random.RandomState(seed)
    T = len(a)
    d0 = stat(a) - stat(b)
    bs = np.empty(n)
    for i in range(n):
        ix = circular_block_indices(T, block, rng)
        bs[i] = stat(a[ix]) - stat(b[ix])
    return float(d0), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def _hac_se(Y: np.ndarray, kernel_lag: Optional[int] = None) -> np.ndarray:
    """Parzen-kernel HAC covariance of the mean of the (T, 4) influence series, as in Ledoit-Wolf (2008)."""
    T = Y.shape[0]
    Yc = Y - Y.mean(axis=0)
    L = kernel_lag or int(np.floor(4 * (T / 100) ** (2 / 9)))
    G = Yc.T @ Yc / T
    for j in range(1, L + 1):
        w = 1 - 6 * (j / (L + 1)) ** 2 + 6 * (j / (L + 1)) ** 3 if j / (L + 1) <= 0.5 else 2 * (1 - j / (L + 1)) ** 3
        Gj = Yc[j:].T @ Yc[:-j] / T
        G += w * (Gj + Gj.T)
    return G


def sharpe_diff_test(r1: np.ndarray, r2: np.ndarray, block: int = 5, n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Ledoit-Wolf (2008) studentised circular-block bootstrap test of H0: SR1 = SR2 (two-sided p-value)."""
    def diff_and_se(a, b):
        Y = np.column_stack([a, b, a ** 2, b ** 2])
        m = Y.mean(axis=0)
        v1, v2 = m[2] - m[0] ** 2, m[3] - m[1] ** 2
        d = m[0] / np.sqrt(v1) - m[1] / np.sqrt(v2)
        grad = np.array([m[2] / v1 ** 1.5, -m[3] / v2 ** 1.5, -0.5 * m[0] / v1 ** 1.5, 0.5 * m[1] / v2 ** 1.5])
        se = np.sqrt(max(grad @ _hac_se(Y) @ grad / len(a), 1e-300))
        return d, se

    d0, se0 = diff_and_se(r1, r2)
    rng = np.random.RandomState(seed)
    T = len(r1)
    ts = np.empty(n_boot)
    for i in range(n_boot):
        ix = circular_block_indices(T, block, rng)
        db, seb = diff_and_se(r1[ix], r2[ix])
        ts[i] = abs(db - d0) / seb
    return {"sharpe_diff_daily": float(d0), "se": float(se0), "p_value": float((np.sum(ts >= abs(d0) / se0) + 1) / (n_boot + 1))}


def scale_polynomial(f: MultivariatePolynomial, cap: float) -> MultivariatePolynomial:
    """Polynomial of u = x / cap (u in [0, 1]); keeps moments O(1) in the SDP, which is essential for conditioning."""
    return MultivariatePolynomial(f.n_vars, {a: c * cap ** sum(a) for a, c in f.coeffs.items()})
