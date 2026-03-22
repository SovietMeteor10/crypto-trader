"""
Fetch macro context features for crypto regime modelling.
All freely available, no paid subscriptions required.
"""

import pandas as pd
import yfinance as yf
import requests
from pathlib import Path


def _extract_close(df, name):
    """Extract Close column from yfinance DataFrame (handles MultiIndex)."""
    if isinstance(df.columns, pd.MultiIndex):
        return df['Close'].iloc[:, 0].rename(name)
    return df['Close'].rename(name)


def fetch_vix(start: str, end: str) -> pd.Series:
    """Fetch CBOE VIX from Yahoo Finance."""
    vix = yf.download('^VIX', start=start, end=end, progress=False)
    return _extract_close(vix, 'vix')


def fetch_dxy(start: str, end: str) -> pd.Series:
    """Fetch DXY (Dollar Index) from Yahoo Finance."""
    dxy = yf.download('DX-Y.NYB', start=start, end=end, progress=False)
    return _extract_close(dxy, 'dxy')


def fetch_treasury_yield(start: str, end: str) -> pd.Series:
    """Fetch US 10-year Treasury yield from FRED (CSV endpoint)."""
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id=DGS10"
    )
    try:
        df = pd.read_csv(url, index_col=0, parse_dates=True)
        df.index = pd.DatetimeIndex(df.index)
        series = pd.to_numeric(df['DGS10'], errors='coerce').dropna()
        return series.loc[start:end].rename('us10y')
    except Exception as e:
        print(f"FRED fetch failed: {e}, using yfinance fallback")
        tnx = yf.download('^TNX', start=start, end=end, progress=False)
        return _extract_close(tnx, 'us10y')


def fetch_sp500(start: str, end: str) -> pd.Series:
    """Fetch S&P 500 for BTC-equity correlation computation."""
    sp = yf.download('^GSPC', start=start, end=end, progress=False)
    return _extract_close(sp, 'sp500')


def build_macro_dataset(start: str = '2023-01-01', end: str = '2024-12-31') -> pd.DataFrame:
    """Fetch all macro features and combine into daily DataFrame."""
    print("Fetching macro data...")
    vix = fetch_vix(start, end)
    dxy = fetch_dxy(start, end)
    us10y = fetch_treasury_yield(start, end)
    sp500 = fetch_sp500(start, end)

    macro = pd.DataFrame({
        'vix': vix,
        'dxy': dxy,
        'us10y': us10y,
        'sp500': sp500,
    }).ffill().dropna()

    # Derived features
    macro['vix_ma20'] = macro['vix'].rolling(20).mean()
    macro['vix_regime'] = (macro['vix'] > macro['vix_ma20']).astype(int)
    macro['dxy_ret_5d'] = macro['dxy'].pct_change(5)
    macro['sp500_ret_20d'] = macro['sp500'].pct_change(20)
    macro['yield_change'] = macro['us10y'].diff()

    output_path = Path("/home/ubuntu/projects/crypto-trader/data_cache/macro_2023_2024.parquet")
    macro.to_parquet(output_path)
    print(f"Saved macro data: {len(macro)} days to {output_path}")
    return macro


if __name__ == "__main__":
    macro = build_macro_dataset()
    print(macro.tail())
    print(macro.describe())
