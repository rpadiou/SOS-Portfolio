"""Experiment A: multi-start local search versus the order-2 moment-SOS bound. See PROTOCOL.md.

    python experiments/exp_a_local_vs_global.py [--config experiments/configs/exp_a.yaml]
        [--only-n 6 8] [--seeds 3] [--out results/exp_a.jsonl]

Appends one JSON line per instance (resumable: finished instances are skipped).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import warnings

import numpy as np
import yaml

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compute(job: dict) -> dict:
    from sos_portfolio import (ChordalExtension, MinimizerExtractor, SyntheticMarket, build_objective_from_market,
                               build_portfolio_relaxation, certify, scipy_optimize)
    from sos_portfolio.local_solver import analyze_local_minima

    n, nc, seed, p, cfg = job["n"], job["nc"], job["seed"], job["params"], job["cfg"]
    m = SyntheticMarket.generate(n, nc, seed, **p)
    f = build_objective_from_market(m)
    cl = ChordalExtension(m.adjacency_matrix()).build().maximal_cliques
    row = {k: job[k] for k in ("key", "regime", "n", "nc", "seed")}
    row.update({f"p_{k}": v for k, v in p.items()})
    opts = {"max_threads": 1}

    budget = "star" if len(cl) <= 12 else "chain"
    pr = build_portfolio_relaxation(f, cl, m.budget_lower, m.budget_upper, 2, "sparse", budget=budget)
    t = time.perf_counter()
    rs = pr.solve("CLARABEL", **opts)
    row.update(lb_sparse=rs["lower_bound"], status_sparse=rs["status"], inaccurate_sparse=rs["inaccurate"],
               t_sparse=time.perf_counter() - t, iters_sparse=rs["iterations"],
               max_block_sparse=rs["sizes"]["max_block"], moments_sparse=rs["sizes"]["unique_moments"])
    row.update(lb_dense=None, status_dense=None, t_dense=None)
    if n <= cfg["dense_max_n"]:
        t = time.perf_counter()
        rd = build_portfolio_relaxation(f, None, m.budget_lower, m.budget_upper, 2, "dense").solve("CLARABEL", **opts)
        row.update(lb_dense=rd["lower_bound"], status_dense=rd["status"], t_dense=time.perf_counter() - t)

    rng = np.random.RandomState(cfg["seed_local"] + seed)
    starts = [rng.dirichlet(np.ones(n)) * rng.uniform(m.budget_lower, m.budget_upper)
              for _ in range(cfg["n_local_starts"])]
    t = time.perf_counter()
    loc = scipy_optimize(f, starts=starts, b_lo=m.budget_lower, b_hi=m.budget_upper)
    row["t_local"] = time.perf_counter() - t
    res = loc["results"]
    fr = np.array([r["f"] if r["feasible"] else np.inf for r in res])
    for K in cfg["K_values"]:
        row[f"ub_K{K}"] = float(fr[:K].min())
    row["n_minima"] = analyze_local_minima(res)["n_distinct_minima"]
    row["n_infeasible_runs"] = int(np.sum(~np.isfinite(fr)))

    lb = rs["lower_bound"]
    if lb is not None:
        ex = MinimizerExtractor(pr.relaxation, rs["moments"], n)
        fl = ex.flatness()
        pts = ex.extract()
        cand = ([] if pts["points"] is None else list(pts["points"])) + [ex.mean_point()]
        if loc["x_opt"] is not None:
            pass  # the local best is deliberately NOT a candidate: the certificate must come from the moments
        c = certify(f, lb, cand, m.budget_lower, m.budget_upper, eps_abs=1e-6, eps_rel=1e-4)
        row.update(flat=bool(fl["flat"]), extraction=pts["reason"], certified=c["certified"],
                   cert_gap=c["gap"], f_hat=c["f_hat"])
        ref = min(row["ub_K100"], c["f_hat"] if c["certified"] and c["f_hat"] is not None else np.inf)
    else:
        row.update(flat=None, extraction="no bound", certified=False, cert_gap=None, f_hat=None)
        ref = row["ub_K100"]
    row["f_cert"] = float(ref)
    tau = max(1e-6, 1e-4 * abs(ref))
    row["tau"] = tau
    row["p_single_fail"] = float(np.mean(fr - ref > tau))
    for K in cfg["K_values"]:
        row[f"fail_K{K}"] = bool(row[f"ub_K{K}"] - ref > tau)
    row["gap_abs"] = None if lb is None else float(row["ub_K100"] - lb)
    row["gap_rel"] = None if lb is None else float((row["ub_K100"] - lb) / max(abs(lb), 1e-3))
    return row


def jobs_from(cfg: dict, only_n=None, seeds=None):
    out = []
    for regime, spec in cfg["regimes"].items():
        for n, nc in cfg["sizes"].items():
            if only_n and n not in only_n:
                continue
            for ci, cell in enumerate(spec["cells"]):
                for seed in range(seeds if seeds is not None else spec["seeds"]):
                    out.append({"key": f"{regime}|{n}|{ci}|{seed}", "regime": regime, "n": n, "nc": nc,
                                "seed": seed, "params": cell, "cfg": {k: cfg[k] for k in
                                ("seed_local", "n_local_starts", "K_values", "dense_max_n")}})
    return sorted(out, key=lambda j: -j["n"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "experiments/configs/exp_a.yaml"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results/exp_a.jsonl"))
    ap.add_argument("--only-n", type=int, nargs="*")
    ap.add_argument("--seeds", type=int)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["key"] for l in open(a.out)}
    todo = [j for j in jobs_from(cfg, a.only_n, a.seeds) if j["key"] not in done]
    print(f"{len(todo)} instances to run ({len(done)} done)", flush=True)
    t0 = time.time()
    with open(a.out, "a") as fh, mp.get_context("spawn").Pool(cfg["workers"]) as pool:
        for i, row in enumerate(pool.imap_unordered(compute, todo, chunksize=1), 1):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if i % 25 == 0:
                print(f"{i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
