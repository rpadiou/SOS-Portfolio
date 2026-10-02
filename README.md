# SOS-Portfolio

Global optimisation of a degree-4 portfolio risk polynomial with the Lasserre moment-SOS hierarchy,
dense and with correlative sparsity, and an empirical check of what the resulting certificate is worth.
The write-up is in [paper/sos_portfolio.pdf](paper/sos_portfolio.pdf).

## What it is

- Order-2 relaxation of `min f(x)` on `{x >= 0, b_lo <= sum x <= b_hi}`, dense and sparse (cliques of the interaction graph).
- An a posteriori certificate: a feasible point whose value is within a tolerance of the lower bound. The bound is the primal objective of the moment problem, so the certificate holds to solver accuracy.
- A multi-start local solver, for comparison.
- Three experiments: A (local search against the relaxation, synthetic), B (estimation error on kurtosis coefficients),
  C (37 US stocks, walk-forward, 2015-2025).

## What it is not

- Not a trading strategy and not a calibrated risk model. The synthetic markets are uncalibrated.
- Not evidence that the hierarchy beats other portfolio methods: experiment C finds no out-of-sample gain over risk-based baselines.

## Results

Generated from `results/` by `scripts/make_readme_tables.py`. Details, intervals and caveats are in the paper.

<!-- BEGIN:headline -->
| experiment | measured | result |
|---|---|---|
| A, 1760 synthetic instances | probability that one local start misses the global optimum, default regime | 0.30 at n=15, 0.83 at n=50 |
| A | order-2 relaxation certifies the optimum, to solver accuracy | all but 29 of 1760 (default regime, n=50: 74%) |
| A | sparse SDP against 100 local runs, n=50, default regime (median of 50 instances, six concurrent workers) | 80 s against 33 s |
| size, n=50 | dense over sparse PSD entries (v1 claimed 398) | 128 to 215 |
| B, kurtosis estimation error | ball formulation against nominal, 10 assets | worse for rho <= 0.2, 2 to 5% better at rho = 0.5 (gamma <= 0.1) |
| B | shrinkage at rho = 0.5, regret over nominal | 0.21 |
| C, 37 US stocks, 2015-2025 | local solver suboptimal on sector model (w4 = 0, 1, 3) | 14%, 6%, 4% of dates |
| C | net-return intervals containing 0 | 13 of 13 comparisons |
<!-- END:headline -->

## Known limits

- Dense costs beyond n = 15 are extrapolated, not measured.
- The SDP bound is only as accurate as the solver; it can exceed the objective at the optimum by about 1e-6 relative (paper, appendix B).
- Experiment B uses drawn estimation errors on synthetic markets. Experiment C uses a fixed universe of current large caps (survivorship bias).
- In experiment A, an instance that is not certified takes the best of 100 local runs as its reference, which understates the failure rate of 100 starts; the paper gives the rates by regime and n, with and without those instances.
- The 13 comparisons of experiment C are not corrected for multiple testing.
- Version 1 (tag `v1.0-paper`) had errors; they are listed in `docs/CLAIMS.md` and in the paper.

## Reproduce

```bash
pip install -e ".[dev,data]"
pytest                      # fast tests
pytest -m slow              # bound soundness test (about a minute)
python main.py --mode sparse --n-assets 15 --n-clusters 3
python experiments/exp_a_local_vs_global.py && python experiments/summarize_exp_a.py
python experiments/exp_b_robust.py && python experiments/summarize_exp_b.py   # writes the 14 MB results/exp_b.jsonl (not versioned)
python scripts/download_data.py && python experiments/exp_c_real_data.py && python experiments/summarize_exp_c.py
python scripts/make_paper_numbers.py && python paper/generate_figures.py && (cd paper && latexmk -pdf sos_portfolio.tex)
```

Experiment C needs `data/raw/prices.parquet`, which is not versioned; `data/manifest.json` records the file used (tickers, dates, SHA-256).
A new download can give a different hash if Yahoo has revised the adjusted prices. Some prices then differ from the file used here, so the results of experiment C can change; the size of the change was not measured.
Experiment C takes a few hours on six workers; the per-strategy times are in `results/exp_c_alloc.jsonl`.

## Layout

```
src/sos_portfolio/   polynomial ring, relaxations, indexing, chordal graphs, extraction, local solver, empirical model
experiments/         protocols (written before the runs), experiment scripts, summarisers
scripts/             complexity and dense-baseline measurements, data download, paper numbers
results/             outputs used by the paper
docs/                claims register, design decisions
paper/               LaTeX source, generated numbers and tables, PDF
tests/               unit, regression and slow soundness tests
```

## References

Lasserre, SIAM J. Optim. 11(3), 2001. Waki, Kim, Kojima, Muramatsu, SIAM J. Optim. 17(1), 2006.
Lasserre, SIAM J. Optim. 17(3), 2006. Henrion, Lasserre, in Positive Polynomials in Control, 2005. Full list in the paper.

Parts of the code and the text were written with an AI assistant. I reviewed the results, ran an audit of the claims, and corrected the errors it found (see docs/CLAIMS.md and Appendix A of the paper).
