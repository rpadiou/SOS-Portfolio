# Limitations

What the project does not show, what it knows it cannot show, and what it did not test.
Numbers are not repeated here; each point names its source (`results/`, the paper, or an entry of `docs/CLAIMS.md`).

## What the project does not show

- That global optimisation is useless in finance. It tests one polynomial family (synthetic) and one polynomial estimated on 37 stocks.
- That the Frobenius-ball formulation fails in general. Experiment B draws the estimation error; it does not use real estimation error (X-B1 to X-B4).
- That the certified minimiser is a good portfolio. The objective has no return term, so net return measures something nothing optimises (X-C2).
- Anything about markets other than 37 US large caps, 2015-2025, long-only, daily data.

## Known limits

- **Comparator.** The local solver is SLSQP from random starts on the budget set. A practitioner would start from equal weights or from the minimum-variance portfolio, or would use a global solver; the project did not compare with one (BARON, SCIP, Gurobi). The failure rates of Experiment A are rates of this comparator.
- **Model.** The synthetic polynomial is not the fourth co-moment of a portfolio. The concentration terms are a stylised penalty, not derived from an execution model (X-D2). The sector model sets cross-sector co-moments to zero (X-C4).
- **Bound.** The reported bound is the primal objective of the moment problem, so "certified" means certified to solver accuracy (X-D3, X-C5). The dual objective was not extracted.
- **Dense costs.** Dense timings and memory beyond `n = 15` are extrapolations (`results/dense_extrapolation.csv`).
- **Reference bias.** In Experiment A an instance that is not certified takes the best of 100 local runs as its reference, so 100 starts cannot fail there by construction (X-D4).
- **Statistics.** The comparisons of Experiment C are not corrected for multiple testing; one period, one universe, survivorship bias.
- **Estimation and model error are not separated.** The explanation that moment estimation error hides the benefit of the certified minimiser is a hypothesis, not a measurement.
- **Naming.** The default regime of Experiment A is called `calibrated` in the code and tables; nothing is calibrated to data. Turnover is the sum of absolute weight changes, buys and sells both counted, and costs are charged on that sum.

## Not tested

- Why local minima multiply with the number of assets; why shrinkage beats the ball in Experiment B; why the global minimiser does not transfer out of sample.
- The bound accuracy for four assets or more, for sparse relaxations with several cliques and auxiliary budget variables, and for the polynomials of Experiment C. A bias common to every solver would go undetected (X-S3).
- Order 3 for the two 8-asset instances where the sparse bound is below the dense one (X-D6).
- The effect on Experiment C of a new price download.

## External fragilities

- `scripts/download_data.py` depends on Yahoo Finance through `yfinance`. The service can change or stop working, and the script would then fail or return different data.
- The original snapshot is not versioned (terms of use). `data/manifest.json` holds its tickers, dates and SHA-256; a new download is expected to give a different hash.
<!-- BEGIN:download -->
- A new download can differ from the file used: in a re-download of 154,401 price cells, 26 differed by more than 1e-6 in relative terms (median 1.1e-7, largest 1.3e-6). The effect on the results of experiment C was not measured.
<!-- END:download -->
- Dependencies are pinned in `requirements.lock` (Python 3.12). Newer versions of CVXPY, CLARABEL or SciPy may change solver statuses and therefore some certification verdicts.
- The continuous integration runs the fast tests only. The slow soundness test is run by hand with `pytest -m slow`.
