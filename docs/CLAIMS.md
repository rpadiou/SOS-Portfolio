# Claims register

Every quantitative or qualitative claim made in the README, the paper or a
docstring has one line here: where it is made, which script proves it, where the
result lives, and its status (`verified`, `refuted`, `corrected`, `removed`,
`pending`). Status is updated phase by phase.

| ID | Claim (v1.0) | Source | Proof script | Result file | Status |
|----|--------------|--------|--------------|-------------|--------|
| E1 | The 2-asset benchmark is non-convex ("Hessian changes sign across K") | paper Sec. 2.2, Fig. 1 caption, README motivation | `scripts/reproduce_review_findings.py e1` | `results/review_findings_v1.json` | refuted (min Hessian eigenvalue on K = 1.6814 > 0; 1 local minimum) — pending rewrite |
| E2 | Default synthetic instances illustrate the value of global optimisation | README, paper Sec. 7 | `scripts/reproduce_review_findings.py e2` | same | refuted (sparse = dense = local, 1 local minimum on the 5 default instances) — pending |
| E3 | Sparse bound equals the dense bound (Thm. 7: lambda_sp = lambda_d) | paper Thm. 7, README, test `test_sparse_single_clique_matches_dense` | `scripts/reproduce_review_findings.py e3`; `tests/regression/test_e3_sparse_budget.py` | same | refuted (v1.0 sparse lb = -22.07 vs dense -2.43 on instance (6,2,0,skew 2)); cause: budget relaxed to E[sum x] — pending fix |
| E4 | "398x reduction in SDP variables at n=50" | README, paper Tables 2-3, `main.py` | `scripts/reproduce_review_findings.py e4` | same | corrected pending (398.7x is a ratio of PSD-matrix entries; unique moments ratio is 252.8x; metrics inconsistent between rows) |
| E5 | Dense SDP takes ">60 s" (n=15) and ">60 h" (n=50) | README, paper Table 3 | `scripts/reproduce_review_findings.py e5` | same | unsupported (v1.0 dense builder is scalar-loop based; values are not SDP timings) — pending |
| E6 | n=50 default: CLARABEL certified in ~1.85 s; SCS (default) lb -0.006045, no flat extension | README, `main.py` | `scripts/reproduce_review_findings.py e6` | same | reproduced; default solver to be switched to CLARABEL — pending |
| E7 | "Mathematical correctness guaranteed by the 51-test suite" | paper Sec. 7.4 | — | — | removed (several tests assert only `isfinite`; none uses overlapping cliques) — pending |
| E8a | Henrion-Lasserre extraction for r>1 | `extractor.py` docstring | — | — | refuted (random combination + non-orthogonal eig; atom merge by position) — pending |
| E8b | Robust extension is "distributionally robust (Delage-Ye)" | README, paper Sec. 6 | — | — | refuted (Frobenius-ball on kappa; SOC applied to all degree-4 moments) — pending |
| E8c | Local solver returns feasible UB | `local_solver.py` | — | — | refuted (scaling projection, `success: True` unconditionally) — pending |
| R1 | "a guarantee inaccessible to gradient-based methods" | README | — | — | removed (promotional) — pending |
| R2 | "real-time rebalancing feasible" | paper Sec. 7.4 | — | — | removed — pending |
| R3 | Interpretation: 60.6/34.4 allocation driven by asymmetric kurtosis | paper Sec. 2.2 | — | — | pending (ablation or removal) |

Note on E5: in this environment the v1.0 dense n=10 instance solved in 10.4 s with
status `optimal` (review: 11.6 s, `optimal_inaccurate`). All other numbers of E1-E6
reproduced exactly (see `results/review_findings_v1.json`).

## v2 experiments (results-driven entries)

| ID | Claim tested | Protocol | Proof script | Result file | Status |
|----|--------------|----------|--------------|-------------|--------|
| X-B1 | Optimising the Frobenius-ball robust objective gives a smaller *true* regret than nominal when kappa is estimated with error | `experiments/PROTOCOL_B.md` | `experiments/exp_b_robust.py`, `experiments/summarize_exp_b.py` | `results/exp_b.jsonl`, `results/exp_b_summary.md` | refuted on the tested grid: on n10-mixed the ball is worse than nominal for rho <= 0.2 at every gamma (mean regret x1.2 to x300 at rho = 0.05) and only matches it at rho = 0.5 for gamma <= 0.1 (x0.95-0.98); the automatic rule (c = 1, 2) never beats nominal. On n6-mixed it is worse at rho = 0.2 and 0.5 |
| X-B2 | The ball does more than shrinking kappa_hat towards a structured target | same | same | same | refuted: shrinkage beats nominal when noise is large (rho >= 0.2 on the mixed markets; mean regret x0.2-0.3 at rho = 0.5) while the ball does not |
| X-B3 | Stress markets discriminate between methods | same | same | same | not informative: nominal solution is a corner (one asset at 100%, eff. N = 1), regret ~ 0 for all methods; ratios to nominal are degenerate |
| X-B4 | Local multi-start result is globally optimal | same | same | same | partially verified: SDP order-2 check on 280 solves certifies 85% of nominal and 83% of robust cases; uncertified robust cases may reflect the relaxation gap ||E z|| <= E||z||, not local failure |

Experiment B is a negative result for the robust extension (see also E8b): it is
reported as found, per the reading rule of the protocol.

| ID | Claim tested | Protocol | Proof script | Result file | Status |
|----|--------------|----------|--------------|-------------|--------|
| X-C1 | On real data the globally certified minimiser differs from the multi-start local one | `experiments/PROTOCOL_C.md` | `experiments/exp_c_real_data.py`, `experiments/summarize_exp_c.py` | `results/exp_c_summary.md` | U10 (dense): refuted, SOS and local coincide on all 132 rebalances. U37 (sector cliques): local is suboptimal (> 1e-4 relative) on 14%, 6% and 4% of dates for w4 = 0, 1, 3 (max relative excess 3.0, 3.4, 0.28). Every SOS solve is certified (a posteriori gap about -1e-6 to -3e-7, i.e. solver tolerance) |
| X-C2 | The SOS minimiser has out-of-sample value over simple baselines | same | same | same | not supported: on U10 (w4 = 1) volatility is 1.0 pt below 1/N (95% CI [-1.96, -0.14]) but 0.7 pt above Ledoit-Wolf min-variance (CI [0.25, 1.20]); net return differences have intervals containing 0; Sharpe p-values 0.32 to 0.98. On U37 the sector model has volatility 1.8 pt below 1/N and an interval containing 0 against min-variance. No multiple-comparison correction |
| X-C3 | Certified global beats local out of sample | same | same | same | not supported: U37 sector, w4 = 1, SOS has slightly higher volatility than local (+0.19 pt, CI [0.06, 0.41]); no difference in CVaR or net return |
| X-C4 | Price of sparsity (sector model versus full co-moment model) | same | same | same | measured: the full-model objective at the sector-SOS solution is 0.32, 0.42, 0.55 (w4 = 0, 1, 3) versus 0.09, 0.22, 0.34 at the full-model local solution, and worse on 98 to 100% of dates; out-of-sample volatility and CVaR are close |
| X-C5 | `frac_lb_le_obj` below 1 (0.21 on U10, w4 = 0) | same | same | `results/exp_c_model.csv` | unexplained, not investigated: median certificate gap is -2e-6, so the bound exceeds the objective by about the solver tolerance; to be checked before the number is quoted |

Caveats of experiment C: survivorship bias, 37 fixed US large caps, no shorting, 5 and 10 bp costs only, 13 comparisons listed without correction. Extractor cap added before the evaluation run (protocol amendment 3).
