"""Skewness, kurtosis, volatility and correlation of the 37 stocks of the local snapshot.

Daily log returns over the whole snapshot (the same returns as experiment C). Skewness is mean(z^3) and kurtosis
mean(z^4) with z = (r - mean) / std(ddof=1), as in empirical.performance. Writes results/universe_moments.csv and results/universe_summary.json.

    python scripts/describe_universe.py
"""
import json

import numpy as np
import pandas as pd

if __name__ == "__main__":
    px = pd.read_parquet("data/raw/prices.parquet")
    r = np.log(px).diff().dropna()
    z = (r - r.mean()) / r.std(ddof=1)
    out = pd.DataFrame({"ticker": r.columns, "ann_vol": (r.std(ddof=1) * np.sqrt(252)).values,
                        "skew": (z ** 3).mean().values, "kurtosis": (z ** 4).mean().values})
    out.to_csv("results/universe_moments.csv", index=False)
    C = np.corrcoef(r.values, rowvar=False)
    summary = {"n_returns": len(r), "first": str(r.index[0].date()), "last": str(r.index[-1].date()),
               "median_pairwise_corr": float(np.median(C[np.triu_indices(len(C), 1)]))}
    json.dump(summary, open("results/universe_summary.json", "w"), indent=1)
    print(summary, out.describe().round(2).to_string(), sep="\n")
