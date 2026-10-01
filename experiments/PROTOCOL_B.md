# Experiment B — effect of the Frobenius-ball robustness (written before the runs)

Question: when the kurtosis coefficients kappa are estimated with error, does optimising
the robust objective `f(x; kappa_hat) + delta ||W z(x)||_2` give a smaller *true* regret than
the nominal optimisation, and does it do more than shrinking kappa_hat towards a structured target?

## Setup

* Markets: `SyntheticMarket.generate` with (n=6, 2 clusters) and (n=10, 2 clusters), regimes
  `mixed` (skew 1.0, kurt 0.5, impact 0.15) and `stress-moderate` (skew 2.0, kurt 0.3, impact 0.05); seeds 0, 1.
* kappa is the vector theta of modelled coefficients (diagonal kappa_ii and within-cluster kappa_ij);
  Frobenius weights W: 1 on diagonals, sqrt(2) on off-diagonals (symmetric matrix norm).
* Estimation error: theta_hat = theta_true + eps, `||eps||_W = rho ||theta_true||_W` exactly, isotropic
  direction, rho in {0.05, 0.1, 0.2, 0.5}, 200 draws per (market, rho).
* Each draw also carries T=20 synthetic observations theta_t = theta_hat + d_t (d_t centred, per-coordinate
  std sqrt(T) * rho ||theta||_W / sqrt(p)), used only by the calibration rule.
* Methods per draw: nominal (delta=0); robust with delta = gamma ||theta_true||_W, gamma in
  {0.02, 0.05, 0.1, 0.2, 0.5, 1}; robust with the automatic rule delta = c * sigma_boot (c in {1, 2}, fixed
  in advance; sigma_boot = W-norm of the bootstrap std (500 resamples) of the mean of the T observations);
  shrinkage: nominal optimisation of (1-s) theta_hat + s * target, target = (mean diagonal, mean off-diagonal),
  s in {0.25, 0.5, 0.75, 1}.
* Optimiser: multi-start SLSQP (20 starts) on the exact robust objective (analytic gradient); on 10 draws per
  (market, rho, delta) the order-2 robust SDP bound is also computed to report how often the local result
  is certified global (gap <= 1e-4 relative).
* x*_true: global minimiser of f(x; theta_true), best of 100 starts checked against the SDP bound.

## Metrics

true regret `f(x_hat; theta_true) - f(x*_true; theta_true)`; distance `0.5 ||x_hat - x*_true||_1`;
instability = mean over draws of `0.5||x_hat_j - mean allocation||_1`; effective N `1/sum(x/sum x)^2`;
worst-case value `f(x_hat; theta_hat) + r ||W z(x_hat)||`, r = rho ||theta_true||_W (an upper bound on the true
objective since ||eps|| = r), and its excess over the true value (conservatism).

## Reading rule

Improvement is judged by median and mean regret relative to nominal, with a paired bootstrap 95% interval over
draws. Null or negative results are reported as found.
