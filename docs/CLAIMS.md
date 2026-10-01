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
| X-B1 | Optimising the Frobenius-ball objective gives a smaller *true* regret than nominal when kappa is estimated with error | `experiments/PROTOCOL_B.md` | `experiments/exp_b_robust.py`, `experiments/summarize_exp_b.py` | `results/exp_b.jsonl`, `results/exp_b_summary.md` | not supported on the tested grid, with one marginal exception. On n10-mixed the ball is worse than nominal for rho <= 0.2 at every gamma (mean regret x1.2 to x300 at rho = 0.05). At rho = 0.5 and gamma = 0.02, 0.05, 0.1 it improves mean regret by 2%, 4% and 5% (paired bootstrap intervals exclude 0); at gamma >= 0.2 it is worse. The automatic rule (c = 1, 2) never beats nominal. On n6-mixed the ball is worse at rho = 0.2 and 0.5. Corrected: an earlier version said the ball only matched nominal at rho = 0.5 |
| X-B2 | The ball does more than shrinking kappa_hat towards a structured target | same | same | same | refuted: shrinkage beats nominal when noise is large (rho >= 0.2 on the mixed markets; mean regret x0.2-0.3 at rho = 0.5) while the ball does not |
| X-B3 | Stress markets discriminate between methods | same | same | same | mostly not informative: the nominal solution is a corner and equals x* in 95 to 100% of draws in every stress cell. One cell discriminates: n10-stress at rho = 0.5, where nominal regret is 2.2e-2 (5% of draws miss the corner), the ball with gamma <= 0.5 changes nothing, gamma = 1 and the rule with c = 2 raise it to 0.64-0.72, and shrinkage lowers it to 0 for s >= 0.5. Corrected: an earlier version said regret was about 0 for all methods in all stress cells |
| X-B4 | Local multi-start result is globally optimal | same | same | `results/exp_b.csv` (columns `sdp_lb`, `sdp_certified`) | verified for the nominal problem, partial for the ball problems: order-2 SDP check on 1120 solves (10 draws per market, rho and method; 280 per market). Nominal: certified global in 100% of cases. Ball problems: 99% (gamma 0.02), 98% (0.05), 96% (0.1), 89% (0.2), 51% (0.5), 25% (1.0). Uncertified ball cases may reflect the relaxation gap ||E z|| <= E||z|| rather than a local failure (not tested). Corrected: an earlier version of this line quoted 85% and 83%, which were the figures of the last market printed in the summary, not the total |

Experiment B is a negative result for the robust extension (see also E8b): it is
reported as found, per the reading rule of the protocol.

| ID | Claim tested | Protocol | Proof script | Result file | Status |
|----|--------------|----------|--------------|-------------|--------|
| X-C1 | On real data the globally certified minimiser differs from the multi-start local one | `experiments/PROTOCOL_C.md` | `experiments/exp_c_real_data.py`, `experiments/summarize_exp_c.py` | `results/exp_c_summary.md` | U10 (dense): refuted, SOS and local coincide on all 132 rebalances. U37 (sector cliques): local is suboptimal (> 1e-4 relative) on 14%, 6% and 4% of dates for w4 = 0, 1, 3 (max relative excess 3.0, 3.4, 0.28). Every SOS solve is certified (a posteriori gap about -1e-6 to -3e-7, i.e. solver tolerance) |
| X-C2 | The SOS minimiser has out-of-sample value over simple baselines | same | same | same | not supported: on U10 (w4 = 1) volatility is 1.0 pt below 1/N (95% CI [-1.96, -0.14]) but 0.7 pt above Ledoit-Wolf min-variance (CI [0.25, 1.20]); net return differences have intervals containing 0; Sharpe p-values 0.32 to 0.98. On U37 the sector model has volatility 1.8 pt below 1/N and an interval containing 0 against min-variance. No multiple-comparison correction |
| X-C3 | Certified global beats local out of sample | same | same | same | not supported: U37 sector, w4 = 1, SOS has slightly higher volatility than local (+0.19 pt, CI [0.06, 0.41]); no difference in CVaR or net return |
| X-C4 | Price of sparsity (sector model versus full co-moment model) | same | same | same | measured: the full-model objective at the sector-SOS solution is 0.32, 0.42, 0.55 (w4 = 0, 1, 3) versus 0.09, 0.22, 0.34 at the full-model local solution, and worse on 98 to 100% of dates; out-of-sample volatility and CVaR are close |
| X-C5 | `frac_lb_le_obj` below 1 (0.21 on U10, w4 = 0) means the bound is invalid | same | `scripts/check_bound_tolerance.py`, `tests/test_bound_soundness.py` | `results/bound_tolerance.csv`, `results/exp_c_model.csv` | resolved, not a bug in the relaxation: the old metric tested `lb <= obj + 1e-6` in absolute terms, but at the default CLARABEL tolerance (1e-7) the bound exceeds the objective at the SOS point by a median 2e-6 (w4 = 0) and up to 1.3e-5 on U10 and 1.2e-4 on U37 (2020 windows), always with the same sign. Re-solving 39 windows at tolerance 1e-9 divides the excess by 30 to 100 where the solver converges (w4 = 0, U37 2020) and leaves it unchanged where it stalls (`optimal_inaccurate`, all w4 = 1 and 3 cases: 1e-7 to 1.5e-6). Largest excess at 1e-9: 3.5e-6 relative. Same polynomial on both sides (checked to 9 digits), stored SOS points feasible (max violation 4e-16). Metric replaced by the relative excess (median, max, fraction <= 1e-5). A bias below about 1e-6 relative cannot be excluded; certificates hold to solver accuracy, not exactly |

Caveats of experiment C: survivorship bias, 37 fixed US large caps, no shorting, 5 and 10 bp costs only, 13 comparisons listed without correction. Extractor cap added before the evaluation run (protocol amendment 3).
