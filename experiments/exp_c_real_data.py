"""Experiment C, step 1: walk-forward allocations on real data. See PROTOCOL_C.md (frozen before running).

    python experiments/exp_c_real_data.py [--out results/exp_c_alloc.jsonl] [--limit 3] [--burnin]

One JSON line per rebalance date with the weights of every strategy and solver diagnostics.
Performance metrics are computed by experiments/summarize_exp_c.py from these weights.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOW = 756
EVAL_START = "2015-01-01"
U10 = ["AAPL", "MSFT", "JNJ", "PFE", "JPM", "BAC", "PG", "KO", "HD", "MCD"]
W4 = (0.0, 1.0, 3.0)
CAP = 0.25          # 0 <= x_i <= CAP for every strategy (amendment 1 of PROTOCOL_C, before any run)


def load():
    px = pd.read_parquet(os.path.join(ROOT, "data/raw/prices.parquet"))
    uni = pd.read_csv(os.path.join(ROOT, "data/universe.csv"))
    rets = np.log(px).diff().dropna()
    return rets, uni


def rebalance_indices(rets: pd.DataFrame, start: str, end: str | None = None):
    d = rets.index
    first = pd.Series(np.arange(len(d)), index=d).groupby([d.year, d.month]).first()
    idx = [i for i in first.values if d[i] >= pd.Timestamp(start) and i >= WINDOW and (end is None or d[i] < pd.Timestamp(end))]
    return idx


def solve_model(Rw, sectors, w4, structure, tag, cert_cfg):
    """Local + SOS solve of the estimated polynomial for one window. structure: 'dense' (single clique) or 'sparse' (sectors)."""
    from sos_portfolio import MinimizerExtractor, build_portfolio_relaxation, certify
    from sos_portfolio import empirical as em
    n = Rw.shape[1]
    Rc = em.prepare_window(Rw)
    blocks = em.blocks_of(n, None if structure == "dense" else sectors)
    w = (1.0, 1.0, w4)
    lam = em.lambdas(Rc, blocks, w)
    t = time.perf_counter()
    xl = em.solve_local(Rc, blocks, lam, cap=CAP)
    t_loc = time.perf_counter() - t
    f = em.scale_polynomial(em.polynomial(Rc, blocks, lam), CAP)       # polynomial of u = x / CAP
    t = time.perf_counter()
    pr = build_portfolio_relaxation(f, None if structure == "dense" else blocks, 1.0 / CAP, 1.0 / CAP, 2, structure,
                                    budget="star" if len(blocks) <= 12 else "chain")
    r = pr.solve("CLARABEL", max_threads=1)
    out = {"local": {"w": xl.tolist(), "time": t_loc, "obj": em.objective(xl, Rc, blocks, lam)[0]}}
    if r["lower_bound"] is None:
        out["sos"] = {"w": xl.tolist(), "time": time.perf_counter() - t, "status": r["status"], "lb": None,
                      "certified": False, "fallback": True}
        return out, Rc, lam
    ex = MinimizerExtractor(pr.relaxation, r["moments"], n)
    pts = ex.extract()
    cand = ([] if pts["points"] is None else list(pts["points"])) + [ex.mean_point()]
    c = certify(f, r["lower_bound"], cand, 1.0 / CAP, 1.0 / CAP, eps_abs=cert_cfg[0], eps_rel=cert_cfg[1])
    xs = CAP * c["x_hat"] if c["x_hat"] is not None else xl
    xs = np.clip(xs, 0, None)
    xs = xs / xs.sum()
    out["sos"] = {"w": xs.tolist(), "time": time.perf_counter() - t, "status": r["status"], "inaccurate": r["inaccurate"],
                  "lb": r["lower_bound"], "certified": bool(c["certified"]), "gap": c["gap"],
                  "obj": em.objective(xs, Rc, blocks, lam)[0], "flat": bool(ex.flatness()["flat"]),
                  "fallback": c["x_hat"] is None}
    return out, Rc, lam


def job(args):
    from sos_portfolio import empirical as em
    i, rets_values, cols, sector_of = args
    cols = list(cols)
    R_all = rets_values[i - WINDOW:i]
    res = {"i": int(i)}
    for uname, tickers in (("U10", U10), ("U37", cols)):
        ix = [cols.index(t) for t in tickers]
        Rw = R_all[:, ix]
        sectors = np.array([sector_of[t] for t in tickers])
        S = {}
        for name, fn in (("EW", lambda R: np.full(R.shape[1], 1.0 / R.shape[1])),
                         ("MinVar_LW", lambda R: em.min_variance(R, CAP)), ("ERC_LW", em.erc), ("HRP", em.hrp),
                         ("MinCVaR95", lambda R: em.min_cvar(R, cap=CAP))):
            t = time.perf_counter()
            S[name] = {"w": fn(Rw).tolist(), "time": time.perf_counter() - t}
        for w4 in W4:
            if uname == "U10":
                o, Rc, lam = solve_model(Rw, sectors, w4, "dense", "full", (1e-6, 1e-4))
                S[f"Poly_full_local_w4={w4:g}"] = o["local"]
                S[f"Poly_full_SOS_w4={w4:g}"] = o["sos"]
            else:
                o, Rc, lam = solve_model(Rw, sectors, w4, "sparse", "sector", (1e-6, 1e-4))
                S[f"Poly_sector_local_w4={w4:g}"] = o["local"]
                S[f"Poly_sector_SOS_w4={w4:g}"] = o["sos"]
                # price of sparsity: full 37-asset model at the sector-model solutions, and its own local solution
                full_blocks = em.blocks_of(Rw.shape[1], None)
                lam_f = em.lambdas(Rc, full_blocks, (1.0, 1.0, w4))
                t = time.perf_counter()
                xf = em.solve_local(Rc, full_blocks, lam_f, cap=CAP)
                S[f"Poly_full_local_w4={w4:g}"] = {"w": xf.tolist(), "time": time.perf_counter() - t}
                for key, x in (("sector_SOS", o["sos"]["w"]), ("sector_local", o["local"]["w"]), ("full_local", xf)):
                    S[f"fullmodel_obj_at_{key}_w4={w4:g}"] = {"obj": em.objective(np.asarray(x), Rc, full_blocks, lam_f)[0]}
        res[uname] = S
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results/exp_c_alloc.jsonl"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--burnin", action="store_true", help="pipeline check on burn-in dates only (2013-2014)")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    rets, uni = load()
    sector_of = dict(zip(uni.ticker, pd.factorize(uni.sector)[0]))
    idx = rebalance_indices(rets, "2013-01-01", "2015-01-01") if a.burnin else rebalance_indices(rets, EVAL_START)
    if a.limit:
        idx = idx[: a.limit]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["i"] for l in open(a.out)}
    todo = [i for i in idx if i not in done]
    args = [(i, rets.values, list(rets.columns), sector_of) for i in todo]
    print(f"{len(todo)} rebalances", flush=True)
    t0 = time.time()
    with open(a.out, "a") as fh, mp.get_context("spawn").Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(job, args), 1):
            r["date"] = str(rets.index[r["i"]].date())
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"{k}/{len(todo)} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
