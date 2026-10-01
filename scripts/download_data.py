"""Download adjusted daily prices (yfinance) for data/universe.csv into data/raw/prices.parquet.

The universe is fixed ex post (current large caps): survivorship bias. Prices are split- and
dividend-adjusted by Yahoo (auto_adjust=True). Missing values are not filled here.
The raw file is not versioned (Yahoo terms of use); data/manifest.json records what experiment C used,
including the SHA-256 of the file. Adjusted prices can change when Yahoo revises them, so a new download
may not match the hash.

    python scripts/download_data.py                 # download, then write the manifest
    python scripts/download_data.py --manifest-only # describe the existing file
"""
import hashlib
import json
import sys

import pandas as pd

PATH = "data/raw/prices.parquet"


def write_manifest():
    px = pd.read_parquet(PATH)
    info = {"file": PATH, "sha256": hashlib.sha256(open(PATH, "rb").read()).hexdigest(), "rows": int(px.shape[0]),
            "tickers": list(px.columns), "first_date": str(px.index.min().date()), "last_date": str(px.index.max().date()),
            "missing_values": int(px.isna().sum().sum()), "source": "Yahoo Finance via yfinance, auto_adjust=True"}
    json.dump(info, open("data/manifest.json", "w"), indent=1)


if __name__ == "__main__":
    if "--manifest-only" not in sys.argv:
        import yfinance as yf
        u = pd.read_csv("data/universe.csv")
        px = yf.download(list(u.ticker), start="2009-06-01", end="2025-12-31", auto_adjust=True, progress=False)["Close"]
        px = px[list(u.ticker)]
        print(px.shape, "missing ratio per ticker:\n", px.isna().mean().round(4).to_string())
        px.to_parquet(PATH)
    write_manifest()
