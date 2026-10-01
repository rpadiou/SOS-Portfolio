"""Re-solve a fixed sample of experiment A, B and C instances with tighter tolerances and a second solver.

For each sampled case the stored SDP result (CLARABEL, tolerance 1e-7) is compared with
  clarabel_1e-10  CLARABEL with all tolerances at 1e-10
  scs_1e-8        SCS with eps = 1e-8 (first-order, independent of the interior-point code; small cases only)
on the lower bound, the a posteriori certificate, the flatness test and, per experiment, the quantities that enter
the conclusions: the failure flags of K-start local search (A), the certification decision of the order-2 check (B),
the SOS allocation (C). Sample seed: 2026. Output: results/solver_precision.csv. Run with PYTHONPATH=src.
With --dd-only, re-solves the experiment A instances where the sparse bound is more than 1e-3 below the dense one
(results/solver_precision_dd.csv).
With --small, compares the bound with an independent minimum on 2- and 3-asset instances (results/small_instance_offsets.csv).
"""
import json
import multiprocessing as mp
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
SEED = 2026
VARIANTS = [("clarabel_1e-10", "CLARABEL", dict(tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)),
            ("scs_1e-8", "SCS", dict(eps=1e-8, max_iters=200000))]


def solve(pr, variant):
    name, solver, opts = variant
    return pr.solve(solver, max_threads=1, **opts) if solver == "CLARABEL" else pr.solve(solver, **opts)


# experiment A
def job_a(row):
    from sos_portfolio import (ChordalExtension, MinimizerExtractor, SyntheticMarket, build_objective_from_market,
                               build_portfolio_relaxation, certify)
    p = {k: row["p_" + k] for k in ("skew_scale", "kurt_base", "impact_base")}
    m = SyntheticMarket.generate(row["n"], row["nc"], row["seed"], **p)
    f = build_objective_from_market(m)
    cl = ChordalExtension(m.adjacency_matrix()).build().maximal_cliques
    pr = build_portfolio_relaxation(f, cl, m.budget_lower, m.budget_upper, 2, "sparse", budget="star" if len(cl) <= 12 else "chain")
    out = []
    for v in VARIANTS:
        if v[1] == "SCS" and row["n"] > 15:
            continue
        r = solve(pr, v)
        rec = {"exp": "A", "id": row["key"], "variant": v[0], "lb": r["lower_bound"], "lb_ref": row["lb_sparse"],
               "status": r["status"], "certified_ref": bool(row["certified"]), "flat_ref": bool(row["flat"]),
               "scale": max(abs(row["lb_sparse"]), 1e-3)}
        if r["lower_bound"] is not None:
            ex = MinimizerExtractor(pr.relaxation, r["moments"], row["n"])
            pts = ex.extract()
            cand = ([] if pts["points"] is None else list(pts["points"])) + [ex.mean_point()]
            c = certify(f, r["lower_bound"], cand, m.budget_lower, m.budget_upper, eps_abs=1e-6, eps_rel=1e-4)
            ref = min(row["ub_K100"], c["f_hat"] if c["certified"] and c["f_hat"] is not None else np.inf)
            tau = max(1e-6, 1e-4 * abs(ref))
            flips = sum(bool((row[f"ub_K{K}"] - ref > tau) != row[f"fail_K{K}"]) for K in (1, 5, 20, 100))
            rec.update(certified=bool(c["certified"]), flat=bool(ex.flatness()["flat"]), changed_flags=flips)
        out.append(rec)
    return out


def sample_a(rng):
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "results/exp_a.jsonl"))]
    df = pd.DataFrame(rows)
    df["absgap"] = df.cert_gap.abs()
    pick = []
    pick += list(df[df.inaccurate_sparse & (df.n <= 30)].sample(8, random_state=SEED).index)       # solver stalls
    pick += list(df[(~df.certified) & (df.n <= 30) & ~df.index.isin(pick)].sample(6, random_state=SEED).index)   # uncertified
    pick += list(df[(df.n <= 30) & ~df.index.isin(pick)].nsmallest(8, "absgap").index)             # tightest gaps
    pick += list(df[(df.n <= 30) & ~df.index.isin(pick)].sample(5, random_state=SEED).index)       # random
    pick += list(df[(df.n == 50) & ~df.index.isin(pick)].sample(3, random_state=SEED).index)       # large
    return [rows[i] for i in pick]


# experiment B
def job_b(case):
    import exp_b_robust as B
    from sos_portfolio import ChordalExtension, build_portfolio_relaxation
    mi, rho, d, method, delta, local_val, lb_ref, cert_ref = case
    M = B.Model(*B.MARKETS[mi])
    rng = np.random.RandomState(7 + 1000 * mi + int(rho * 100))
    tn, p = M.wnorm(M.theta), len(M.theta)
    th_hat = None
    for k in range(d + 1):                                           # replay the random stream of the cell
        g = rng.randn(p)
        eps = (g / M.w) * (rho * tn / np.linalg.norm(g))
        th_hat = M.theta + eps
        sig = np.sqrt(B.T_OBS) * rho * tn / np.sqrt(p) / M.w
        rng.randn(B.T_OBS, p)
        rng.randint(0, B.T_OBS, size=(B.N_BOOT, B.T_OBS))
    f = M.poly(th_hat)
    cl = ChordalExtension(M.m.adjacency_matrix()).build().maximal_cliques
    pr = build_portfolio_relaxation(f, cl, M.b_lo, M.b_hi, 2, "sparse", robust_terms=M.m.kappa_terms(), kappa_radius=delta)
    out = []
    for v in VARIANTS:
        r = solve(pr, v)
        lb = r["lower_bound"]
        cert = bool(lb is not None and local_val - lb <= 1e-4 * max(abs(lb), 1e-3))
        out.append({"exp": "B", "id": f"{B.MARKETS[mi][0]}|{rho}|{d}|{method}", "variant": v[0], "lb": lb, "lb_ref": lb_ref,
                    "status": r["status"], "certified": cert, "certified_ref": bool(cert_ref), "scale": max(abs(lb_ref), 1.0)})
    return out


def sample_b():
    import exp_b_robust as B
    df = pd.read_csv(os.path.join(ROOT, "results/exp_b.csv"))
    df = df[df.sdp_lb.notna()].reset_index(drop=True)
    df["sdp_certified"] = df.sdp_certified.astype(bool)
    unc = df[~df.sdp_certified].sample(15, random_state=SEED)
    cer = df[df.sdp_certified & (df.method != "nominal")].sample(15, random_state=SEED)
    cases = []
    names = [m[0] for m in B.MARKETS]
    for _, r in pd.concat([unc, cer]).iterrows():
        cases.append((names.index(r.market), float(r.rho), int(r.draw), r.method, float(r.delta), float(r.local_val), float(r.sdp_lb), bool(r.sdp_certified)))
    return cases


# experiment C
def job_c(case):
    import exp_c_real_data as C
    from sos_portfolio import MinimizerExtractor, build_portfolio_relaxation, certify
    from sos_portfolio import empirical as em
    U, i, w4 = case
    rets, uni = C.load()
    sector_of = dict(zip(uni.ticker, pd.factorize(uni.sector)[0]))
    cols = list(rets.columns)
    tick = C.U10 if U == "U10" else cols
    Rw = rets.values[i - C.WINDOW:i][:, [cols.index(t) for t in tick]]
    n, dense = len(tick), U == "U10"
    Rc = em.prepare_window(Rw)
    blocks = em.blocks_of(n, None if dense else np.array([sector_of[t] for t in tick]))
    f = em.scale_polynomial(em.polynomial(Rc, blocks, em.lambdas(Rc, blocks, (1, 1, w4))), C.CAP)
    key = ("Poly_full_SOS_w4=" if dense else "Poly_sector_SOS_w4=") + f"{w4:g}"
    stored = [json.loads(l) for l in open(os.path.join(ROOT, "results/exp_c_alloc.jsonl")) if f'"i": {i},' in l][0][U][key]
    pr = build_portfolio_relaxation(f, None if dense else blocks, 1 / C.CAP, 1 / C.CAP, 2, "dense" if dense else "sparse",
                                    budget="star" if len(blocks) <= 12 else "chain")
    out = []
    for v in VARIANTS:
        if v[1] == "SCS" and not dense:
            continue
        r = solve(pr, v)
        rec = {"exp": "C", "id": f"{U}|{i}|w4={w4:g}", "variant": v[0], "lb": r["lower_bound"], "lb_ref": stored["lb"], "status": r["status"],
               "certified_ref": bool(stored["certified"]), "flat_ref": bool(stored["flat"]), "scale": max(1.0, abs(stored["lb"]))}
        if r["lower_bound"] is not None:
            ex = MinimizerExtractor(pr.relaxation, r["moments"], n)
            pts = ex.extract()
            cand = ([] if pts["points"] is None else list(pts["points"])) + [ex.mean_point()]
            c = certify(f, r["lower_bound"], cand, 1 / C.CAP, 1 / C.CAP, eps_abs=1e-6, eps_rel=1e-4)
            xs = C.CAP * c["x_hat"] if c["x_hat"] is not None else np.array(stored["w"])
            xs = np.clip(xs, 0, None)
            xs = xs / xs.sum()
            rec.update(certified=bool(c["certified"]), flat=bool(ex.flatness()["flat"]),
                       weight_l1=float(0.5 * np.abs(xs - np.array(stored["w"])).sum()))
        out.append(rec)
    return out


def sample_c():
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "results/exp_c_alloc.jsonl"))]
    recs = [(r["i"], w, r["U10"][f"Poly_full_SOS_w4={w}"]) for r in rows for w in "013"]
    df = pd.DataFrame([{"i": i, "w4": float(w), "status": s["status"], "absgap": abs(s["gap"]), "excess": s["lb"] - s["obj"]} for i, w, s in recs])
    pick = list(df[df.status == "optimal_inaccurate"].sample(8, random_state=SEED).index)
    pick += list(df[~df.index.isin(pick)].nsmallest(6, "absgap").index)
    pick += list(df[~df.index.isin(pick)].nlargest(6, "excess").index)
    pick += list(df[~df.index.isin(pick)].sample(4, random_state=SEED).index)
    cases = [("U10", int(df.loc[k, "i"]), float(df.loc[k, "w4"])) for k in pick]
    cases += [("U37", i, 0.0) for i in (3545, 3588, 3630)]
    cases += [("U37", i, 1.0) for i in (3545, 3630)]
    return cases


def dd_check():
    """Re-solve the experiment A instances where the sparse bound is more than 1e-3 below the dense bound."""
    from sos_portfolio import ChordalExtension, SyntheticMarket, build_objective_from_market, build_portfolio_relaxation
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "results/exp_a.jsonl"))]
    out = []
    for r in [r for r in rows if r["lb_dense"] is not None and r["lb_dense"] - r["lb_sparse"] > 1e-3]:
        p = {k: r["p_" + k] for k in ("skew_scale", "kurt_base", "impact_base")}
        m = SyntheticMarket.generate(r["n"], r["nc"], r["seed"], **p)
        f = build_objective_from_market(m)
        cl = ChordalExtension(m.adjacency_matrix()).build().maximal_cliques
        rec = {"id": r["key"], "dd_stored": r["lb_dense"] - r["lb_sparse"]}
        for v in VARIANTS:
            sp = solve(build_portfolio_relaxation(f, cl, m.budget_lower, m.budget_upper, 2, "sparse"), v)
            de = solve(build_portfolio_relaxation(f, None, m.budget_lower, m.budget_upper, 2, "dense"), v)
            rec[f"dd_{v[0]}"] = de["lower_bound"] - sp["lower_bound"]
        out.append(rec)
    pd.DataFrame(out).to_csv(os.path.join(ROOT, "results/solver_precision_dd.csv"), index=False)


def small_check():
    """Bound against an independent minimum on the small instances of tests/test_bound_soundness.py."""
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from sos_portfolio import build_portfolio_relaxation
    from test_bound_soundness import exact_min, small_instances
    out = []
    for name, f, n, lo, hi, m in small_instances():
        fs = exact_min(f, n, lo, hi, m)
        pr = build_portfolio_relaxation(f, None, lo, hi, 2, "dense")
        rec = {"instance": name, "n": n, "f_star": fs}
        for label, opts in (("default", {}), ("tight", dict(tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10))):
            r = pr.solve("CLARABEL", max_threads=1, **opts)
            rec[f"lb_minus_fstar_{label}"] = r["lower_bound"] - fs
            rec[f"status_{label}"] = r["status"]
        out.append(rec)
    pd.DataFrame(out).to_csv(os.path.join(ROOT, "results/small_instance_offsets.csv"), index=False)


def run(args):
    kind, case = args
    return {"A": job_a, "B": job_b, "C": job_c}[kind](case)


if __name__ == "__main__":
    if "--small" in sys.argv:
        small_check()
        sys.exit(0)
    if "--dd-only" in sys.argv:
        dd_check()
        sys.exit(0)
    rng = np.random.RandomState(SEED)
    jobs = [("A", r) for r in sample_a(rng)] + [("B", c) for c in sample_b()] + [("C", c) for c in sample_c()]
    jobs.sort(key=lambda j: -(j[1]["n"] if j[0] == "A" else 0))     # slow instances first
    print(len(jobs), "cases", flush=True)
    res = []
    with mp.get_context("spawn").Pool(6) as pool:
        for k, out in enumerate(pool.imap_unordered(run, jobs), 1):
            res += out
            print(k, flush=True)
    df = pd.DataFrame(res)
    df["d_lb_rel"] = (df.lb - df.lb_ref).abs() / df.scale
    df.to_csv(os.path.join(ROOT, "results/solver_precision.csv"), index=False)
