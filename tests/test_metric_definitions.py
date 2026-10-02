"""The indicators of the paper, on the 3-asset, 5-day toy example of Appendix B, against values worked out by hand.

Weights are 1/3 on every day (rebalanced daily), costs are 10 bp per unit of traded notional, as in the appendix.
Hand values were computed in exact fractions and rounded to 17 digits; the tolerance is 1e-12.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sos_portfolio import empirical as em

ROOT = Path(__file__).resolve().parent.parent
TOL = 1e-12

# simple returns, one row per day, columns = assets A, B, C
R = np.array([[0.04, 0.02, 0.00],
              [-0.02, -0.04, 0.03],
              [0.06, 0.03, 0.00],
              [-0.06, -0.03, -0.03],
              [0.03, 0.00, 0.03]])

# gross portfolio return: mean of the three returns of the day, e.g. day 2: (-0.02 - 0.04 + 0.03) / 3 = -0.01
GROSS = [0.02, -0.01, 0.03, -0.04, 0.02]

# turnover on day k+1 = sum_i |1/3 - (1/3)(1 + r_i) / (1 + g)| = (1/3) sum_i |g - r_i| / (1 + g), day k data; none on day 1
#   after day 1: (1/3)(0.04) / 1.02, after day 2: (1/3)(0.08) / 0.99, after day 3: (1/3)(0.06) / 1.03, after day 4: (1/3)(0.04) / 0.96
TURNOVER = [0.0, 0.013071895424836602, 0.026936026936026935, 0.019417475728155338, 0.013888888888888888]
MEAN_TURNOVER = 0.01832857174447694                   # mean over the four rebalances after the first

# net return = gross - 0.001 * turnover
NET = [0.02, -0.010013071895424837, 0.029973063973063974, -0.04001941747572815, 0.01998611111111111]


@pytest.fixture(scope="module")
def simulate():
    spec = importlib.util.spec_from_file_location("summarize_exp_c", ROOT / "experiments" / "summarize_exp_c.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.simulate


def test_gross_net_and_turnover(simulate):
    W = np.full((5, 3), 1.0 / 3.0)
    gross, net, turn = simulate(W, [0, 1, 2, 3, 4], R, 0.001)
    np.testing.assert_allclose(gross, GROSS, atol=TOL, rtol=0)
    np.testing.assert_allclose(turn, TURNOVER, atol=TOL, rtol=0)
    np.testing.assert_allclose(net, NET, atol=TOL, rtol=0)
    assert float(np.mean(turn[1:])) == pytest.approx(MEAN_TURNOVER, abs=TOL)


def test_performance_indicators():
    p = em.performance(np.array(NET), np.array(GROSS))
    # volatility: gross returns, mean 0.004, sum of squared deviations 0.00332, variance (n - 1 = 4) 0.00083, times sqrt(252)
    assert p["ann_vol"] == pytest.approx(0.45734013600382812, abs=TOL)
    # CVaR 5%: five days, the 5% quantile (-0.034) lies between the two worst days, so the tail is the worst day, -0.04; sign flipped
    assert p["cvar5"] == pytest.approx(0.04, abs=TOL)
    # drawdown: cumulative sum of net returns peaks on day 3 and is 0.04001941747572815 lower on day 4
    assert p["max_drawdown"] == pytest.approx(0.04001941747572815, abs=TOL)
    # skewness: mean of z^3, z = (r - 0.004) / 0.028809720581775867 (std with n - 1); sum of cubed deviations -6.216e-05
    assert p["skew"] == pytest.approx(-0.51990484289829813, abs=TOL)
    # annualised net return: mean net 0.003985337142604418 times 252
    assert p["ann_return_net"] == pytest.approx(1.0043049599363134, abs=TOL)
    # Sharpe: mean net / std net (n - 1) times sqrt(252), risk-free rate zero
    assert p["sharpe_net"] == pytest.approx(2.1958933653805902, abs=TOL)
