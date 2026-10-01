# Experiment C — real data (protocol frozen before any strategy is run)

Question (not "beat Markowitz"): does the *globally certified* minimiser of an estimated degree-4
risk polynomial have out-of-sample value relative to its local solutions and to simple baselines, and what
does imposing sector sparsity cost?

## Data (frozen snapshot)

* `scripts/download_data.py`: Yahoo Finance adjusted close (splits + dividends), 37 large-cap US stocks in 9
  GICS sectors (`data/universe.csv`), 2009-06 -> 2025-12, stored in `data/raw/prices.parquet`. No missing values.
* Caveats: survivorship bias (universe = current large caps), Yahoo adjustments, no delisted names, daily data
  only, US equities only, no shorting, no financing/borrow costs, prices are close-to-close.
* Daily log returns. Sample: first return 2010-01-04 ... last 2025-12.

## Walk-forward

* Estimation window: 756 trading days ending the day *before* the rebalance date (no look-ahead).
* Rebalance: first trading day of each month; weights then drift with returns until the next rebalance.
* Burn-in 2010-01..2014-12: pipeline checks only. Evaluation 2015-01..2025-12 (132 rebalances).
* Constraints: long-only, `sum x = 1`, `0 <= x_i <= 0.25` for every strategy that optimises (min-variance, min-CVaR, the polynomial
  models; ERC, HRP and 1/N are not capped and their maximum weight is reported).
  **Amendment 1 (made before any strategy was run):** the cap replaces the `x_i <= 1` bound. Without it the SDP is badly
  conditioned (moments up to 1 for weights that are ~1/N at the optimum; interior-point solvers returned bounds above feasible
  objective values). The SDP is solved in `u = x / 0.25 in [0,1]`.
* Costs: `c * sum_i |w_new,i - w_drift,i|` charged on the rebalance day, c in {5, 10} bps per unit of one-way traded notional.

## Polynomial model

`F(x) = l2 m2(x) - l3 m3(x) + l4 m4(x)`, `m_k(x) = mean_t (x' r~_t)^k`, r~ = window returns minus window mean
(no return term: the comparison is about risk).
Returns are rescaled per window so that the average asset volatility is 1; `l_k = w_k / mean_i |m_k(e_i)|`
(each term contributes `w_k` on a single-asset portfolio, on average).
**Amendment 2 (before any run):** the first plan normalised at equal weights, which divides by a diversified third moment close to
zero and produced coefficients spanning five orders of magnitude and failed/invalid SDP solves on the pipeline check.
**Frozen a priori: w = (1, 1, 1).** Ablation over `w4 in {0, 1, 3}` (w2=w3=1) is reported in full. Nothing is tuned on
the evaluation period; nothing is tuned on the burn-in either (it is used only to check that the pipeline runs).

* **V1 (exact, dense):** full co-moment polynomial, n=10 assets fixed a priori as the first two tickers of the first five
  sectors of `universe.csv` (AAPL, MSFT, JNJ, PFE, JPM, BAC, PG, KO, HD, MCD). Global SOS (dense, d=2) vs multi-start local.
* **V2 (sectoral, sparse):** n=37. `m_k` replaced by the sum over sectors of the moments of the sector sub-portfolio
  (intra-sector co-moments only; cross-sector co-moments and covariances set to zero). Sparse SOS (d=2, cliques = sectors)
  vs multi-start local. Sparsity is a modelling choice, not a property of the data. **Price of sparsity:** evaluate the
  *full* 37-asset model (`m_k` over all assets) at the V2 solution and compare with the multi-start local solution of the
  full model; report both the model objective and out-of-sample metrics.

## Comparators (same data, same costs, run on both universes)

1/N; long-only minimum variance with Ledoit-Wolf covariance (`sklearn.covariance.LedoitWolf`); equal risk contribution
(Ledoit-Wolf covariance); HRP (Lopez de Prado 2016, own implementation, single linkage); min-CVaR 95% (Rockafellar-Uryasev
LP on the window scenarios); the polynomial solved locally (multi-start, 20 starts); the polynomial solved by SOS.

## Metrics (out of sample, daily)

annualised volatility, 5% CVaR (daily), max drawdown, realised skewness, mean turnover per rebalance, effective number of
assets `1/sum w^2` (mean over rebalances), net return at 5 and 10 bps, Sharpe ratio (secondary, with caveat), compute time
per rebalance, and for the SOS runs the fraction of rebalances certified (a posteriori certificate) and the gap.

## Inference

Circular block bootstrap (block 21 days, 2000 resamples) for differences between strategies in volatility, CVaR and
net return; Ledoit-Wolf (2008) studentised circular-block bootstrap test for Sharpe-ratio differences. All comparisons
are listed in the output; no selection of favourable ones. Multiple comparisons are not corrected; p-values are
descriptive and this is stated in the summary.

## Expected outcome (before running)

The polynomial objective is estimated from 756 days of data; third and fourth sample co-moments are very noisy
(DeMiguel-Garlappi-Uppal 2009: estimated optimal rules rarely beat 1/N out of sample). The most likely result is
that SOS-global = local (the model is close to convex in these windows), and that neither is distinguishable from the
simple baselines on volatility/CVaR. That would be reported as is.
