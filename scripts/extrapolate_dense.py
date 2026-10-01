"""Log-log extrapolation of the measured dense cost to n=30 and n=50 (clearly labelled, never presented as measured).

Fit: log(y) = a + b log(B), B = largest PSD block C(n+2,2), y in {wall time, peak RSS}, on the dense runs of
results/dense_baseline.csv with n >= 6 (smaller instances are dominated by fixed overhead). Output:
results/dense_extrapolation.csv with point estimate and 95% prediction interval from ordinary least squares.
"""
import csv
from math import comb

import numpy as np
import pandas as pd
from scipy import stats

d = pd.read_csv("results/dense_baseline.csv")
d = d[(d.structure == "dense") & (d.n >= 6)]
B = np.array([comb(n + 2, 2) for n in d.n], dtype=float)
rows = []
for col, label, unit in [("wall_s", "time", "s"), ("peak_rss_mb", "peak memory", "MB")]:
    x, y = np.log(B), np.log(d[col].values)
    res = stats.linregress(x, y)
    n_pts = len(x)
    s = np.sqrt(np.sum((y - (res.intercept + res.slope * x)) ** 2) / (n_pts - 2))
    for n in (30, 50):
        x0 = np.log(comb(n + 2, 2))
        se = s * np.sqrt(1 + 1 / n_pts + (x0 - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
        t = stats.t.ppf(0.975, n_pts - 2)
        mu = res.intercept + res.slope * x0
        rows.append({"quantity": label, "unit": unit, "n": n, "exponent_on_block_size": round(res.slope, 2),
                     "estimate": float(np.exp(mu)), "pi95_low": float(np.exp(mu - t * se)),
                     "pi95_high": float(np.exp(mu + t * se)), "fit_points": n_pts,
                     "status": "EXTRAPOLATED (not measured)"})
pd.DataFrame(rows).to_csv("results/dense_extrapolation.csv", index=False)
print(pd.DataFrame(rows).to_string())
