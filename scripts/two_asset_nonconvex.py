"""Non-convex 2-asset stress instance -> results/two_asset_nonconvex.json.

Picks the smallest seed for which `make_nonconvex(2, 1, seed)` yields >= 2 local minima that are
also found as local minima on a fine grid, then records: Hessian spectrum over K, local minima and
their basins (100 starts), hierarchy bounds for d=2,3, flat-extension spectra and the extracted minimiser.
"""
import itertools
import json

import numpy as np

from sos_portfolio import (MinimizerExtractor, build_objective_from_market, build_portfolio_relaxation,
                           certify, make_nonconvex, scipy_optimize)
from sos_portfolio.local_solver import analyze_local_minima, make_starts

B_LO, B_HI = 0.95, 1.0


def grid_minima(f, m=200):
    pts = {}
    for i, j in itertools.product(range(m + 1), repeat=2):
        x = np.array([i / m, j / m])
        if B_LO - 1e-12 <= x.sum() <= B_HI + 1e-12:
            pts[(i, j)] = f(x)
    out = []
    for (i, j), v in pts.items():
        nb = [pts[(i + a, j + b)] for a in (-1, 0, 1) for b in (-1, 0, 1) if (a or b) and (i + a, j + b) in pts]
        if all(v <= u + 1e-12 for u in nb):
            out.append((i / m, j / m, v))
    return out


def main():
    for seed in range(200):
        try:
            mk, info = make_nonconvex(2, 1, seed, min_minima=2, n_starts=60)
        except RuntimeError:
            continue
        f = build_objective_from_market(mk)
        gm = grid_minima(f)
        if len(gm) >= 2:
            break
    else:
        raise SystemExit("no instance found")
    eigs = [np.linalg.eigvalsh(f.hessian(np.array([t, s - t]))) for s in np.linspace(B_LO, B_HI, 11)
            for t in np.linspace(0, s, 201)]
    eigs = np.array(eigs)
    loc = scipy_optimize(f, starts=make_starts(2, 100, B_LO, B_HI, np.random.RandomState(0)), b_lo=B_LO, b_hi=B_HI)
    mins = analyze_local_minima(loc["results"])
    basin = [{"x": m["x"].tolist(), "f": m["f"], "share": m["count"] / sum(mm["count"] for mm in mins["local_minima"])}
             for m in mins["local_minima"]]
    out = {"seed": seed, "skew_scale": info["skew_scale"], "coefficients": {str(k): v for k, v in f.coeffs.items()},
           "min_hessian_eig": float(eigs[:, 0].min()), "max_hessian_eig": float(eigs[:, 1].max()),
           "grid_minima": gm, "local_minima": basin, "hierarchy": {}}
    for d in (2, 3):
        pr = build_portfolio_relaxation(f, None, B_LO, B_HI, d, "dense")
        r = pr.solve("CLARABEL")
        rec = {"lb": r["lower_bound"], "status": r["status"], "time": r["solve_time"]}
        ex = MinimizerExtractor(pr.relaxation, r["moments"], 2)
        fl = ex.flatness()
        rec.update(flat=fl["flat"], ranks=[(p["rank_d"], p["rank_d_minus_dK"]) for p in fl["per_clique"]],
                   sv_d=fl["per_clique"][0]["sv_d"].tolist(), sv_prev=fl["per_clique"][0]["sv_prev"].tolist())
        pts = ex.extract()
        cert = certify(f, r["lower_bound"], [] if pts["points"] is None else list(pts["points"]) + [ex.mean_point()], B_LO, B_HI)
        rec.update(certified=cert["certified"], gap=cert["gap"], x_hat=None if cert["x_hat"] is None else cert["x_hat"].tolist())
        out["hierarchy"][f"d{d}"] = rec
    json.dump(out, open("results/two_asset_nonconvex.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("coefficients",)}, indent=1)[:2500])


if __name__ == "__main__":
    main()
