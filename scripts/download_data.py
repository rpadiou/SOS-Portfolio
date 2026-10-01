"""Download adjusted daily prices (yfinance) for data/universe.csv and freeze them in data/raw/prices.parquet.

The universe is fixed ex post (current large caps): survivorship bias. Prices are split- and
dividend-adjusted by Yahoo (auto_adjust=True). Missing values are not filled here.
"""
import pandas as pd
import yfinance as yf

u = pd.read_csv("data/universe.csv")
px = yf.download(list(u.ticker), start="2009-06-01", end="2025-12-31", auto_adjust=True, progress=False)["Close"]
px = px[list(u.ticker)]
print(px.shape, "missing ratio per ticker:\n", px.isna().mean().round(4).to_string())
px.to_parquet("data/raw/prices.parquet")
