"""Experiment B: robustness to kurtosis-coefficient estimation error. See PROTOCOL_B.md."""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MARKETS = [  # (name, n, clusters, seed, params)
    ("n6-mixed-s0", 6, 2, 0, dict(skew_scale=1.0, kurt_base=0.5, impact_base=0.15)),
    ("n6-stress-s1", 6, 2, 1, dict(skew_scale=2.0, kurt_base=0.3, impact_base=0.05)),
    ("n10-mixed-s0", 10, 2, 0, dict(skew_scale=1.0, kurt_base=0.5, impact_base=0.15)),
    ("n10-stress-s1", 10, 2, 1, dict(skew_scale=2.0, kurt_base=0.3, impact_base=0.05)),
]
RHOS = [0.05, 0.1, 0.2, 0.5]
GAMMAS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
SHRINK = [0.25, 0.5, 0.75, 1.0]
T_OBS, N_BOOT, N_STARTS = 20, 500, 20


class Model:
    def __init__(self, name, n, nc, seed, params):
        from sos_portfolio import SyntheticMarket
        self.name, self.n = name, n
        self.m = SyntheticMarket.generate(n, nc, seed, **params)
        self.edges = list(self.m.kurtosis_cross.keys())
        self.theta = np.concatenate([self.m.kurtosis_diag, [self.m.kurtosis_cross[e] for e in self.edges]])
        self.w = np.concatenate([np.ones(n), np.full(len(self.edges), np.sqrt(2.0))])
        E = []
        for i in range(n):
            a = [0] * n; a[i] = 4; E.append(a)
        for i, j in self.edges:
            a = [0] * n; a[i] = a[j] = 2; E.append(a)
        self.E = np.array(E)
        self.b_lo, self.b_hi = self.m.budget_lower, self.m.budget_upper

    def wnorm(self, v):
        return float(np.linalg.norm(self.w * v))

    def poly(self, theta):
        from sos_portfolio import build_objective_from_market
        mk = copy.copy(self.m)
        mk.kurtosis_diag = np.array(theta[: self.n])
        mk.kurtosis_cross = {e: float(t) for e, t in zip(self.edges, theta[self.n:])}
        return build_objective_from_market(mk)

    def z(self, x):
        return self.w * np.prod(x[None, :] ** self.E, axis=1)

    def dz_norm_grad(self, x):
        z0 = np.prod(x[None, :] ** self.E, axis=1)
        nz = np.linalg.norm(self.w * z0)
        J = np.zeros((len(z0), self.n))
        for i in range(self.n):
            Ei = self.E.copy(); c = Ei[:, i].astype(float); Ei[:, i] = np.maximum(Ei[:, i] - 1, 0)
            J[:, i] = c * np.prod(x[None, :] ** Ei, axis=1)
        return nz, (J.T @ (self.w ** 2 * z0)) / max(nz, 1e-300)

    def solve_local(self, f, delta, seed, starts=N_STARTS):
        from scipy.optimize import LinearConstraint, minimize
        from sos_portfolio.local_solver import make_starts, feasibility_violation
        rng = np.random.RandomState(seed)
        A = LinearConstraint(np.ones((1, self.n)), self.b_lo, self.b_hi)

        def fun(x):
            nz, g = self.dz_norm_grad(x)
            return f(x) + delta * nz, f.gradient(x) + delta * g

        best, bx = np.inf, None
        for x0 in make_starts(self.n, starts, self.b_lo, self.b_hi, rng):
            r = minimize(fun, x0, jac=True, method="SLSQP", bounds=[(0, 1)] * self.n, constraints=A,
                         options={"maxiter": 300, "ftol": 1e-13})
            x = np.clip(r.x, 0, 1)
            if feasibility_violation(x, self.b_lo, self.b_hi) <= 1e-8 and fun(x)[0] < best:
                best, bx = fun(x)[0], x
        return bx, best

    def sdp_bound(self, f, delta):
        from sos_portfolio import ChordalExtension, build_portfolio_relaxation
        cl = ChordalExtension(self.m.adjacency_matrix()).build().maximal_cliques
        r = build_portfolio_relaxation(f, cl, self.b_lo, self.b_hi, 2, "sparse", robust_terms=self.m.kappa_terms(),
                                       kappa_radius=delta).solve("CLARABEL", max_threads=1)
        return r["lower_bound"]


def metrics(M, x, ftrue, fx_true, r, theta_hat_poly):
    xs = x / x.sum()
    return {"regret": float(ftrue(x) - fx_true), "neff": float(1 / np.sum(xs ** 2)),
            "worst": float(theta_hat_poly(x) + r * np.linalg.norm(M.z(x))),
            "true_val": float(ftrue(x))}


def run_cell(args):
    mi, rho, n_draws, seed0, sdp_checks = args
    M = Model(*MARKETS[mi])
    ftrue = M.poly(M.theta)
    xs_true, fs_true = M.solve_local(ftrue, 0.0, 12345, 100)
    rng = np.random.RandomState(seed0 + 1000 * mi + int(rho * 100))
    tn = M.wnorm(M.theta)
    p = len(M.theta)
    rows = []
    for d in range(n_draws):
        g = rng.randn(p)
        eps = (g / M.w) * (rho * tn / np.linalg.norm(g))               # ||eps||_W = rho ||theta||_W exactly
        th_hat = M.theta + eps
        r = rho * tn
        sig_obs = np.sqrt(T_OBS) * rho * tn / np.sqrt(p) / M.w
        D = rng.randn(T_OBS, p) * sig_obs
        D -= D.mean(axis=0)
        obs = th_hat[None, :] + D
        idx = rng.randint(0, T_OBS, size=(N_BOOT, T_OBS))
        boot = obs[idx].mean(axis=1)
        sigma_boot = M.wnorm(boot.std(axis=0))
        fhat = M.poly(th_hat)
        methods = [("nominal", 0.0, th_hat)]
        methods += [(f"robust_g{g_}", g_ * tn, th_hat) for g_ in GAMMAS]
        methods += [(f"rule_c{c}", c * sigma_boot, th_hat) for c in (1, 2)]
        tgt = np.concatenate([np.full(M.n, th_hat[: M.n].mean()), np.full(p - M.n, th_hat[M.n:].mean())])
        methods += [(f"shrink_s{s}", 0.0, (1 - s) * th_hat + s * tgt) for s in SHRINK]
        for name, delta, th_use in methods:
            fuse = fhat if th_use is th_hat else M.poly(th_use)
            x, val = M.solve_local(fuse, delta, seed0 + d)
            row = {"market": M.name, "rho": rho, "draw": d, "method": name, "delta": float(delta),
                   "dist": float(0.5 * np.abs(x - xs_true).sum()), "x": x.tolist(),
                   **metrics(M, x, ftrue, fs_true, r, fhat)}
            if d < sdp_checks and name.startswith(("robust", "nominal")):
                lb = M.sdp_bound(fuse, delta)
                row["sdp_lb"], row["local_val"] = lb, float(val)
                row["sdp_certified"] = bool(lb is not None and val - lb <= 1e-4 * max(abs(lb), 1e-3))
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--markets", type=int, nargs="*", default=list(range(len(MARKETS))))
    ap.add_argument("--rhos", type=float, nargs="*", default=RHOS)
    ap.add_argument("--sdp-checks", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(ROOT, "results/exp_b.jsonl"))
    a = ap.parse_args()
    cells = [(mi, rho, a.draws, 7, a.sdp_checks) for mi in a.markets for rho in a.rhos]
    t0 = time.time()
    with open(a.out, "w") as fh, mp.get_context("spawn").Pool(a.workers) as pool:
        for k, rows in enumerate(pool.imap_unordered(run_cell, cells), 1):
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"cell {k}/{len(cells)} done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
