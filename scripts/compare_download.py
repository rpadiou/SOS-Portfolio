"""Download the prices again and compare them with the snapshot of data/raw/prices.parquet.

Appends one row to results/download_check.csv: number of cells compared, number of cells whose relative
difference exceeds 1e-6, and the median and the largest relative difference. The snapshot is not modified.

    python scripts/compare_download.py
"""
import csv
import datetime
import os

import numpy as np
import pandas as pd

SNAPSHOT = "data/raw/prices.parquet"
OUT = "results/download_check.csv"

if __name__ == "__main__":
    import yfinance as yf
    old = pd.read_parquet(SNAPSHOT)
    u = pd.read_csv("data/universe.csv")
    new = yf.download(list(u.ticker), start="2009-06-01", end="2025-12-31", auto_adjust=True, progress=False)["Close"][list(u.ticker)]
    common = old.index.intersection(new.index)
    a, b = old.loc[common].to_numpy(), new.loc[common].to_numpy()
    rel = np.abs(a - b) / np.abs(a)
    row = {"date": datetime.date.today().isoformat(), "rows_old": len(old), "rows_new": len(new), "rows_common": len(common),
           "cells": int(rel.size), "cells_above_1e-6": int(np.sum(rel > 1e-6)), "median_rel": float(np.nanmedian(rel)),
           "max_rel": float(np.nanmax(rel))}
    new_file = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new_file:
            w.writeheader()
        w.writerow(row)
    print(row)
