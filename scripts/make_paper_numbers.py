"""Write paper/numbers.tex (macros) and paper/tables.tex (tables) from results/.

Every number in the paper comes from here. Run from the repository root:
    python scripts/make_paper_numbers.py
"""
import json
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper")
M = {}


def pct(x, d=0):
    return f"{100 * x:.{d}f}\\%"


def sci(x, d=1):
    if abs(x) < 1e-12:      # numerical noise
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    return f"\\ensuremath{{{m:.{d}f}\\times10^{{{e}}}}}"


def num(x, d=2):
    return f"{x:.{d}f}"


def put(name, val):
    v = str(val)
    if re.match(r"^-\d", v):
        v = "\\ensuremath{-}" + v[1:]
    M[name] = v


def sci_cell(x, d=1):
    return sci(x, d) if abs(x) > 0 else "0"


# experiment A
a = pd.read_csv(os.path.join(RES, "exp_a.csv"))
put("aN", len(a))
put("aCert", pct(a.certified.mean(), 1))
put("aFlat", pct(a.flat.mean(), 1))
put("aInacc", int(a.inaccurate_sparse.sum()))
put("aUncert", int((~a.certified).sum()))
a["multi"] = a.n_minima > 1
for r, k in (("calibrated", "Cal"), ("mixed", "Mix"), ("stress", "Str")):
    put("aMulti" + k, pct(a[a.regime == r].multi.mean()))
    g = a[a.regime == r]
    for n, w in ((2, "Two"), (4, "Four"), (10, "Ten"), (15, "Fifteen"), (30, "Thirty"), (50, "Fifty")):
        put(f"aPf{k}{w}", num(g[g.n == n].p_single_fail.mean()))
        put(f"aMulti{k}{w}", pct(g[g.n == n].multi.mean()))
for k, w in ((1, "One"), (5, "Five"), (20, "Twenty"), (100, "Hundred")):
    put("aFail" + w, pct(a[f"fail_K{k}"].mean(), 1 if k == 100 else 0))
    put("aBigFail" + w, pct(a[a.n >= 30][f"fail_K{k}"].mean(), 1 if k == 100 else 0))
put("aCalFiftyCert", pct(a[(a.regime == "calibrated") & (a.n == 50)].certified.mean()))
dd = a.dd.dropna()
put("aDDn", len(dd))
put("aDDmed", sci(dd.median()))
put("aDDmax", sci(dd.max()))
put("aDDover", int((dd > 1e-3).sum()))
put("aDDmin", sci(dd.min()))
put("aCalFiftyFailHundred", pct(a[(a.regime == "calibrated") & (a.n == 50)].fail_K100.mean()))
put("aExcessMax", sci((a.lb_sparse - a.ub_K100).max()))
for r, k in (("calibrated", "Cal"), ("stress", "Str")):
    g = a[(a.regime == r) & (a.n == 50)]
    put("aTimeSos" + k, f"{g.t_sparse.median():.0f}")
    put("aTimeLoc" + k, f"{g.t_local.median():.0f}")

# two assets
t1 = json.load(open(os.path.join(RES, "two_asset.json")))
put("tHess", num(t1["min_hessian_eigenvalue_on_K"]))
t2 = json.load(open(os.path.join(RES, "two_asset_nonconvex.json")))
put("ncHess", num(t2["min_hessian_eig"]))
put("ncShare", pct(min(m["share"] for m in t2["local_minima"])))
put("ncLb", num(t2["hierarchy"]["d2"]["lb"], 4))
put("ncBestLocal", num(min(m["f"] for m in t2["local_minima"]), 4))
put("ncWorstLocal", num(max(m["f"] for m in t2["local_minima"]), 4))
put("ncRankDThree", t2["hierarchy"]["d3"]["ranks"][0][0])
put("ncRankDThreePrev", t2["hierarchy"]["d3"]["ranks"][0][1])
put("ncGapTwo", sci(abs(t2["hierarchy"]["d2"]["gap"])))

# dense baseline and complexity
db = pd.read_csv(os.path.join(RES, "dense_baseline.csv"))
for n, w in ((10, "Ten"), (12, "Twelve"), (15, "Fifteen")):
    d = db[(db.n == n) & (db.structure == "dense")].iloc[0]
    s = db[(db.n == n) & (db.structure == "sparse")].iloc[0]
    put(f"db{w}Time", f"{d.wall_s:.0f}" if d.wall_s >= 10 else f"{d.wall_s:.1f}")
    put(f"db{w}Mem", f"{d.peak_rss_mb / 1000:.1f}")
    put(f"db{w}SparseTime", num(s.wall_s))
ex = pd.read_csv(os.path.join(RES, "dense_extrapolation.csv"))
for n, w in ((30, "Thirty"), (50, "Fifty")):
    t = ex[(ex.quantity == "time") & (ex.n == n)].iloc[0]
    m = ex[(ex.quantity == "peak memory") & (ex.n == n)].iloc[0]
    put(f"dbExtTime{w}", sci(t.estimate, 1))
    put(f"dbExtTimeLo{w}", sci(t.pi95_low, 1))
    put(f"dbExtTimeHi{w}", sci(t.pi95_high, 1))
    put(f"dbExtMem{w}", f"{m.estimate / 1000:.0f}")
put("dbExpTime", num(ex[ex.quantity == "time"].exponent_on_block_size.iloc[0]))
cx = pd.read_csv(os.path.join(RES, "complexity.csv"))
d50 = cx[(cx.n == 50) & (cx.structure == "dense")].iloc[0]
for b, w in (("star", "Star"), ("chain", "Chain")):
    s = cx[(cx.n == 50) & (cx.budget == b)].iloc[0]
    put(f"cxPsd{w}", f"{d50.psd_entries / s.psd_entries:.0f}")
    put(f"cxMom{w}", f"{d50.unique_moments / s.unique_moments:.0f}")
    put(f"cxBlk{w}", f"{d50.moment_block_entries / s.moment_block_entries:.0f}")
    put(f"cxTime{w}", f"{s.wall_s:.0f}")
    put(f"cxBlock{w}", int(s.max_block))
put("cxBlockDense", int(d50.max_block))
rf = json.load(open(os.path.join(RES, "review_findings_v1.json")))
rfd = {(r["tag"], r["what"]): r["measured"] for r in rf}
put("vOneSparse", num(float(rfd[("E3", "(6, 2, 0, 2.0) sparse lb")]), 1))
put("vOneDense", num(float(rfd[("E3", "(6, 2, 0, 2.0) dense lb")]), 2))
put("vTwoSparse", num(float(rfd[("E3", "(6, 2, 1, 4.0) sparse lb")]), 0))
put("vTwoDense", num(float(rfd[("E3", "(6, 2, 1, 4.0) dense lb")]), 1))
put("vRatioPsd", f"{1758276 / 4410:.0f}")
put("vRatioMom", f"{316251 / 1251:.0f}")

# experiment B
b = pd.read_csv(os.path.join(RES, "exp_b.csv"))
put("bDraws", int(b.draw.nunique()))
put("bCells", int(b.groupby(["market", "rho"]).ngroups))
put("bMethods", int(b.method.nunique()))
chk = b[b.sdp_lb.notna()]
put("bSdpN", len(chk))
put("bSdpNom", pct(chk[chk.method == "nominal"].sdp_certified.mean()))
put("bSdpRob", pct(chk[chk.method != "nominal"].sdp_certified.mean()))
greg = b.groupby(["market", "rho", "method"]).regret.mean()


def breg(mk, rho, meth):
    return greg[(mk, rho, meth)]


for mk, k in (("n10-mixed-s0", "Ten"), ("n6-mixed-s0", "Six")):
    for rho, r in ((0.05, "A"), (0.2, "B"), (0.5, "C")):
        for meth, w in (("nominal", "Nom"), ("robust_g0.1", "Rob"), ("rule_c1", "Rule"), ("shrink_s0.5", "Shr"), ("robust_g1.0", "Big")):
            put(f"b{k}{r}{w}", sci(breg(mk, rho, meth)))
put("bRatioBigA", f"{breg('n10-mixed-s0', 0.05, 'robust_g1.0') / breg('n10-mixed-s0', 0.05, 'nominal'):.0f}")
put("bRatioShrC", num(breg("n10-mixed-s0", 0.5, "shrink_s0.75") / breg("n10-mixed-s0", 0.5, "nominal")))
put("bRatioRobC", num(breg("n10-mixed-s0", 0.5, "robust_g0.1") / breg("n10-mixed-s0", 0.5, "nominal")))
for gm_, w in (("robust_g0.1", "GOne"), ("robust_g1.0", "GBig"), ("robust_g0.02", "GSmall"), ("robust_g0.5", "GHalf")):
    put(f"bSdp{w}", pct(chk[chk.method == gm_].sdp_certified.mean()))
nm_ = b[b.method == "nominal"]
put("bStressMin", pct(nm_[nm_.market.str.contains("stress")].assign(z=lambda d: d.regret < 1e-9).groupby(["market", "rho"]).z.mean().min()))
put("bStrBig", num(breg("n10-stress-s1", 0.5, "robust_g1.0")))
put("bStrShr", num(breg("n10-stress-s1", 0.5, "shrink_s0.5"), 3))
put("bStressNominal", sci(greg[("n10-stress-s1", 0.5, "nominal")]))

# experiment C
cm = pd.read_csv(os.path.join(RES, "exp_c_metrics.csv")).set_index(["universe", "strategy"])
ct = pd.read_csv(os.path.join(RES, "exp_c_tests.csv"))
cmod = pd.read_csv(os.path.join(RES, "exp_c_model.csv"))
rows = [json.loads(l) for l in open(os.path.join(RES, "exp_c_alloc.jsonl"))]
put("cDates", len(rows))
for w, k in ((0, "Zero"), (1, "One"), (3, "Three")):
    r = cmod[(cmod.universe == "U37") & (cmod.model == f"Poly_sector_w4={w}")].iloc[0]
    put(f"cSub{k}", pct(r["frac_local_suboptimal(>1e-4 rel)"]))
    put(f"cMaxExc{k}", num(r.max_rel_excess_local, 1))
    q = cmod[(cmod.universe == "U10") & (cmod.model == f"Poly_full_w4={w}")].iloc[0]
    put(f"cSubTen{k}", pct(q["frac_local_suboptimal(>1e-4 rel)"]))
put("cCertAll", pct(cmod.frac_certified.min()))
put("cTimeTen", f"{cm.loc[('U10', 'Poly_full_SOS_w4=1'), 'time_per_rebal_s']:.0f}")
put("cTimeSector", f"{cm.loc[('U37', 'Poly_sector_SOS_w4=1'), 'time_per_rebal_s']:.0f}")
put("cTimeLocal", f"{cm.loc[('U37', 'Poly_sector_local_w4=1'), 'time_per_rebal_s']:.0f}")
sp = {}
for w4 in (0, 1, 3):
    x = np.array([r["U37"][f"fullmodel_obj_at_sector_SOS_w4={w4}"]["obj"] for r in rows])
    y = np.array([r["U37"][f"fullmodel_obj_at_full_local_w4={w4}"]["obj"] for r in rows])
    k = {0: "Zero", 1: "One", 3: "Three"}[w4]
    put(f"cSpSec{k}", num(x.mean()))
    put(f"cSpFull{k}", num(y.mean()))
    put(f"cSpWorse{k}", pct((x > y + 1e-9).mean()))


def trow(u, a_, b_):
    r = ct[(ct.universe == u) & (ct.A == a_) & (ct.B == b_)].iloc[0]
    return r


for u, a_, tag in (("U10", "Poly_full_SOS_w4=1", "Ten"), ("U37", "Poly_sector_SOS_w4=1", "Sec")):
    for bn, w in (("EW", "Ew"), ("MinVar_LW", "Mv"), ("HRP", "Hrp"), ("ERC_LW", "Erc")):
        r = trow(u, a_, bn)
        put(f"c{tag}{w}Vol", f"{100 * r.d_vol:+.1f}")
        put(f"c{tag}{w}VolLo", f"{100 * r.vol_lo:+.1f}")
        put(f"c{tag}{w}VolHi", f"{100 * r.vol_hi:+.1f}")
        put(f"c{tag}{w}Net", f"{100 * r.d_netret:+.1f}")
        put(f"c{tag}{w}NetLo", f"{100 * r.netret_lo:+.1f}")
        put(f"c{tag}{w}NetHi", f"{100 * r.netret_hi:+.1f}")
        put(f"c{tag}{w}P", num(r.sharpe_p))
r = trow("U37", "Poly_sector_SOS_w4=1", "Poly_sector_local_w4=1")
put("cSecLocVol", f"{100 * r.d_vol:+.2f}")
put("cSecLocVolLo", f"{100 * r.vol_lo:+.2f}")
put("cSecLocVolHi", f"{100 * r.vol_hi:+.2f}")
put("cTests", len(ct))
put("cNetZero", int(((ct.netret_lo <= 0) & (ct.netret_hi >= 0)).sum()))
put("cCvarZero", int(((ct.cvar_lo <= 0) & (ct.cvar_hi >= 0)).sum()))
pmin = ct.sharpe_p.min()
pmax = ct.sharpe_p.max()
put("cPMin", num(pmin))
put("cPMax", num(pmax))

# bound accuracy (X-C5)
bt = pd.read_csv(os.path.join(RES, "bound_tolerance.csv"))
put("xN", len(bt))
put("xDefMax", sci(max(bt["excess_tol1e-07"].max(), 0)))
put("xTightMax", sci(bt["excess_tol1e-09"].max()))
put("xMinTight", sci(bt["excess_tol1e-09"].min()))
put("xDefMaxU", sci(cmod.lb_excess_max.max()))

# environment
import platform
from importlib import metadata

put("verPython", platform.python_version())
for pkg, k in (("cvxpy", "Cvxpy"), ("clarabel", "Clarabel"), ("numpy", "Numpy"), ("scipy", "Scipy"), ("pandas", "Pandas")):
    put("ver" + k, metadata.version(pkg))
cpu = [l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")]
put("cpuModel", cpu[0].replace("(R)", "").replace("(TM)", ""))
put("cpuThreads", len(cpu))

# write macros
with open(os.path.join(OUT, "numbers.tex"), "w") as fh:
    fh.write("% generated by scripts/make_paper_numbers.py; do not edit\n")
    for k, v in M.items():
        assert k.isalpha(), k
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# tables
T = {}


def table(caption, label, header, rows_, spec):
    T[label.split(":")[1]] = ("\\begin{table}[" + ("!h" if label == "tab:E" else "t") + "]\n\\centering" + ("\\scriptsize" if label == "tab:E" else "\\small") + "\n\\caption{" + caption + "}\\label{" + label + "}\n"
             "\\begin{tabular}{" + spec + "}\n\\toprule\n" + header + " \\\\\n\\midrule\n"
             + " \\\\\n".join(rows_) + " \\\\\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


# Table A
rowsA = []
for r in ("calibrated", "mixed", "stress"):
    for n in (4, 10, 15, 30, 50):
        g = a[(a.regime == r) & (a.n == n)]
        rowsA.append(f"{r} & {n} & {len(g)} & {pct(g.multi.mean())} & {num(g.p_single_fail.mean())} & {pct(g.fail_K20.mean())} & "
                     f"{pct(g.fail_K100.mean())} & {pct(g.certified.mean())} & {g.t_sparse.median():.1f} & {g.t_local.median():.1f}")
table("Experiment A, selected cells. Multi-min: share of instances with several distinct local minima among 100 runs; $p_1$: mean "
      "probability that one start misses the reference optimum; fail $K$: share of instances where the best of $K$ starts misses it; "
      "cert.: share certified a posteriori at $d=2$; times are medians of the sparse SDP and of 100 local runs, in seconds.",
      "tab:A", "regime & $n$ & inst. & multi-min & $p_1$ & fail 20 & fail 100 & cert. & SDP (s) & 100 local (s)", rowsA, "lrrrrrrrrr")

# Table B
rowsB = []
for mk, nm in (("n10-mixed-s0", "10"), ("n6-mixed-s0", "6")):
    for rho in (0.05, 0.2, 0.5):
        v = [g_ for g_ in (breg(mk, rho, m_) for m_ in ("nominal", "robust_g0.1", "robust_g1.0", "rule_c1", "shrink_s0.5", "shrink_s1.0"))]
        rowsB.append(f"{nm} & {rho} & " + " & ".join(sci_cell(x_) for x_ in v))
table("Experiment B, mean true regret over %d draws, mixed markets. Nominal: $\\delta=0$; ball: $\\delta=\\gamma\\|\\theta\\|_W$ with "
      "$\\gamma=0.1$ and $1$; rule: $\\delta=c\\,\\hat\\sigma$ with $c=1$; shrinkage towards the structured target with $s=0.5$ and $1$." % b.draw.nunique(),
      "tab:B", "$n$ & $\\rho$ & nominal & ball 0.1 & ball 1 & rule & shrink 0.5 & shrink 1", rowsB, "rrrrrrrr")

# Table C
rowsC = []
for u, names in (("U10", ["EW", "MinVar_LW", "ERC_LW", "HRP", "MinCVaR95", "Poly_full_local_w4=1", "Poly_full_SOS_w4=1"]),
                 ("U37", ["EW", "MinVar_LW", "ERC_LW", "HRP", "MinCVaR95", "Poly_full_local_w4=1", "Poly_sector_local_w4=1", "Poly_sector_SOS_w4=1"])):
    for s in names:
        r = cm.loc[(u, s)]
        lab = s.replace("_", "\\_").replace("w4=1", "").replace("\\_ ", "").rstrip("\\_")
        rowsC.append(f"{u} & {lab} & {100 * r.ann_vol:.1f} & {100 * r.cvar5:.2f} & {100 * r.max_drawdown:.0f} & {r.turnover:.2f} & "
                     f"{r.eff_N:.1f} & {100 * r.net_ret_5bp:.1f} & {r.sharpe_5bp:.2f}")
table("Experiment C, out of sample 2015--2025 (%d monthly rebalances). Annualised volatility, daily 5\\%% CVaR, maximum drawdown (\\%%), "
      "mean one-way turnover per rebalance, effective number of assets, annualised return net of 5 bp, Sharpe ratio net of 5 bp. "
      "$w_4=1$ for the polynomial models." % len(rows),
      "tab:C", "universe & strategy & vol (\\%) & CVaR (\\%) & MDD & turn. & $N_{\\mathrm{eff}}$ & net (\\%) & Sharpe", rowsC, "llrrrrrrr")

# Table dense baseline
rowsD = []
for n in (4, 6, 8, 10, 12, 15):
    d = db[(db.n == n) & (db.structure == "dense")].iloc[0]
    s = db[(db.n == n) & (db.structure == "sparse")].iloc[0]
    rowsD.append(f"{n} & {d.max_block:.0f} & {d.wall_s:.2f} & {d.peak_rss_mb / 1000:.2f} & {d.status.replace('_', ' ')} & {s.max_block:.0f} & {s.wall_s:.2f}")
for n in (30, 50):
    e_t = ex[(ex.quantity == "time") & (ex.n == n)].iloc[0]
    e_m = ex[(ex.quantity == "peak memory") & (ex.n == n)].iloc[0]
    sb = cx[(cx.n == n) & (cx.budget == "chain")].iloc[0]
    dd_ = cx[(cx.n == n) & (cx.structure == "dense")].iloc[0]
    rowsD.append(f"{n} & {dd_.max_block:.0f} & {sci(e_t.estimate, 1)}$^\\ast$ & {e_m.estimate / 1000:.0f}$^\\ast$ & extrapolated & {sb.max_block:.0f} & {sb.wall_s:.1f}")
table("Dense versus sparse relaxation, order $d=2$, one synthetic instance per $n$. Dense times and memory are measured up to $n=15$; "
      "rows marked $^\\ast$ are log-log extrapolations from five measured points, not measurements (95\\% prediction intervals in "
      "\\texttt{results/dense\\_extrapolation.csv}). Sparse figures use the chain budget.",
      "tab:D", "$n$ & dense block & dense (s) & dense (GB) & dense status & sparse block & sparse (s)", rowsD, "rrrrlrr")

def minus(t):
    return t.replace("-", "$-$")


rowsE = []
ABBR = {"Poly_full_SOS_w4=1": "SOS full", "Poly_sector_SOS_w4=1": "SOS sector", "Poly_sector_local_w4=1": "local sector",
        "Poly_full_local_w4=1": "local full", "MinVar_LW": "MinVar", "ERC_LW": "ERC", "MinCVaR95": "MinCVaR"}
for _, r in ct.iterrows():
    rowsE.append(minus(f"{r.universe} & {ABBR.get(r.A, r.A)} & {ABBR.get(r.B, r.B)} & "
                 f"{100 * r.d_vol:+.2f} [{100 * r.vol_lo:+.2f}, {100 * r.vol_hi:+.2f}] & "
                 f"{100 * r.d_cvar5:+.2f} [{100 * r.cvar_lo:+.2f}, {100 * r.cvar_hi:+.2f}] & "
                 f"{100 * r.d_netret:+.1f} [{100 * r.netret_lo:+.1f}, {100 * r.netret_hi:+.1f}] & {r.sharpe_p:.2f}"))
table("Experiment C, all %d pairwise comparisons, difference A minus B in percentage points (annualised volatility; daily 5\\%% CVaR; "
      "annualised net return at 5 bp) with 95\\%% circular-block bootstrap intervals (blocks of 21 days), and the p-value of the "
      "Ledoit--Wolf Sharpe-ratio test. No multiple-comparison correction; the intervals are descriptive." % len(ct),
      "tab:E", "U & A & B & vol & CVaR & net return & $p$", rowsE, "lllllll")

for k_, v_ in T.items():
    open(os.path.join(OUT, f"tab_{k_}.tex"), "w").write("% generated by scripts/make_paper_numbers.py; do not edit\n" + v_)
print(f"{len(M)} macros, {len(T)} tables")
