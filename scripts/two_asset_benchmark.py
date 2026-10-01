"""2-asset benchmark: convexity check, hierarchy bounds, term ablation -> results/two_asset.json.

The benchmark is a validation case (strictly convex on K). The ablation removes one
coefficient group at a time and reports the minimiser, to test the v1.0 claim that the
60.6/34.4 allocation is driven by asymmetric kurtosis.
"""
import json

import numpy as np

import sos_portfolio.portfolio_problem as pp
from sos_portfolio import MultivariatePolynomial, build_objective, build_portfolio_relaxation, scipy_optimize

B_LO, B_HI = 0.95, 1.0


def min_hessian_eig(f, m=400):
    lam, arg = np.inf, None
    for s in np.linspace(B_LO, B_HI, 11):
        for t in np.linspace(0, s, m + 1):
            x = np.array([t, s - t])
            e = np.linalg.eigvalsh(f.hessian(x))[0]
            if e < lam:
                lam, arg = e, x
    return float(lam), arg.tolist()


def solve(f):
    out = {}
    for d in (2, 3):
        r = build_portfolio_relaxation(f, None, B_LO, B_HI, d, "dense").solve("CLARABEL")
        out[f"lb_d{d}"] = r["lower_bound"]
        out[f"time_d{d}"] = r["solve_time"]
    loc = scipy_optimize(f, 50, 0, B_LO, B_HI)
    out["f_local"], out["x_local"] = loc["f_opt"], loc["x_opt"].tolist()
    return out


def ablate(keep):
    c = build_objective().coeffs
    groups = {"variance": [(2, 0), (0, 2), (1, 1)], "skewness": [(3, 0), (0, 3)],
              "quartic_diag": [(4, 0), (0, 4)], "quartic_cross": [(2, 2)]}
    kept = {a: v for g, al in groups.items() if g in keep for a in al for v in [c[a]]}
    return MultivariatePolynomial(2, kept)


def main():
    f = build_objective()
    lam, at = min_hessian_eig(f)
    res = {"coefficients": {str(k): v for k, v in f.coeffs.items()},
           "min_hessian_eigenvalue_on_K": lam, "argmin": at, "full": solve(f), "ablation": {}}
    allg = ["variance", "skewness", "quartic_diag", "quartic_cross"]
    for g in allg:
        res["ablation"][f"without_{g}"] = solve(ablate([h for h in allg if h != g]))
    with open("results/two_asset.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "coefficients"}, indent=1))


if __name__ == "__main__":
    main()
