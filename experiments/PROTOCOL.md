# Experiment A — local versus global (written before the runs)

Question: on degree-4 portfolio risk polynomials, how often does a multi-start local
solver miss the global minimum, how large is the miss, and what does the order-2
relaxation certify, as a function of regime and dimension?

## Instances

`SyntheticMarket.generate(n, n_clusters, seed, ...)`, block (cluster) interaction graph,
budget `0.95 <= sum x <= 1`, long-only. Synthetic and uncalibrated in every regime.

| regime | skew_scale | kurt_base | impact_base | notes |
|---|---|---|---|---|
| `calibrated` | 0.3 | 0.5 | 0.15 | generator defaults |
| `mixed` | 1.0 | 0.5 | 0.15 | moderately skewed |
| `stress` | {1, 2, 4} | {0.3, 0.5} | 0.05 | 6 sub-cells, strong skewness, low kurtosis |

`n` and clusters: n=2 (1), 4 (2), 6 (2), 8 (2), 10 (2), 15 (3), 30 (6), 50 (10).
Seeds: 0-49 per cell (`calibrated`, `mixed`); `stress` uses seeds 0-19 per sub-cell (6 x 20 = 120
instances per n) to bound run time. Reductions are declared in `results/meta.json`.

## Methods

* Sparse relaxation (order d=2, CLARABEL, star budget; cliques from the chordal extension
  of the cluster graph, which here are the clusters).
* Dense relaxation (same constraints, single clique) for n <= 10 only: measured cost at n=12 is
  ~40 s / 1 GB and n=15 is reported separately in `results/dense_baseline.csv`.
* Local search: 100 SLSQP runs with exact linear constraints, starts = Dirichlet samples with
  random total weight in [0.95, 1] (i.i.d., so the first K runs are an unbiased K-start search).
  A run counts as an upper bound only if feasible to 1e-8.

## Definitions (fixed before running)

* `f_cert`: reference optimum = min(UB_100, f(x_hat)) where x_hat comes from extraction + polish
  when it passes the certificate; otherwise UB_100.
* Certificate (primary): feasible x_hat with `f(x_hat) - lb <= max(1e-6, 1e-4 |lb|)`.
  Flat extension (rank tol 1e-5) is reported as secondary.
* tau = max(1e-6, 1e-4 |f_cert|).  "Local search with K starts fails" iff `UB_K - f_cert > tau`.
* Single-start failure probability: fraction of the 100 runs with `f_run - f_cert > tau`
  (infeasible runs count as failures).
* Relative gap: `(UB_100 - lb) / max(|lb|, 1e-3)`; absolute gap also reported (1e-3 is the order
  of magnitude of |f*| in the calibrated regime at n=50).
* Sparse vs dense: `lb_dense - lb_sparse` (>= 0 up to solver accuracy, see tests).
* Times: wall seconds of the relaxation (build + solve) and of the 100 local runs, measured
  inside worker processes (6 concurrent workers, CLARABEL single-threaded): indicative only.

## Expected outcome (stated before the runs, to be confronted with results)

Calibrated regime: local = global. Stress regime: multiple local minima; single-start failures
frequent; K=20-100 starts close most but not necessarily all of the gap as n grows. The relaxation
at d=2 is expected to be tight but not always; anything else is reported as found.
