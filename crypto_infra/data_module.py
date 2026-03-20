"""DataModule — fetches, caches, validates OHLCV and funding rates."""

import os
import time
import hashlib
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import ccxt
import pandas as pd
import numpy as np


class DataError(Exception):
    """Raised when data cannot be fetched, is incomplete, or fails validation."""


class DataModule:
    def __init__(self, cache_dir: str = "./data_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._exchange = ccxt.binance({"options": {"defaultType": "future"}})

    def _cache_key(self, prefix: str, symbol: str, timeframe: str, start: str, end: str) -> str:
        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        return os.path.join(
            self.cache_dir,
            f"{prefix}_{safe_symbol}_{timeframe}_{start}_{end}.parquet",
        )

    def _cache_key_funding(self, symbol: str, start: str, end: str) -> str:
        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        return os.path.join(
            self.cache_dir,
            f"funding_{safe_symbol}_{start}_{end}.parquet",
        )

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        cache_path = self._cache_key("ohlcv", symbol, timeframe, start, end)
        if os.path.exists(cache_path):
            df = pd.read_parquet(cache_path)
            df.index = pd.to_datetime(df.index, utc=True)
            return df

        print(f"Fetching {symbol} {timeframe} {start} to {end}...")

        since_ms = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

        all_candles = []
        current = since_ms
        limit = 1000

        while current < end_ms:
            for attempt in range(6):
                try:
                    candles = self._exchange.fetch_ohlcv(
                        symbol, timeframe, since=current, limit=limit
                    )
                    break
                except (ccxt.RateLimitExceeded, ccxt.NetworkError) as e:
                    if attempt == 5:
                        raise DataError(f"Failed after 5 retries: {e}")
                    wait = 2 ** attempt
                    print(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)

            if not candles:
                break

            all_candles.extend(candles)
            last_ts = candles[-1][0]
            if last_ts <= current:
                break
            current = last_ts + 1

        if not all_candles:
            raise DataError(f"No data returned for {symbol} {timeframe} {start}-{end}")

        df = pd.DataFrame(
            all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()

        # Filter to requested range
        df = df[(df.index >= pd.Timestamp(start, tz="UTC")) &
                (df.index < pd.Timestamp(end, tz="UTC"))]

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(np.float64)

        self.validate(df)
        df.to_parquet(cache_path)
        return df

    def get_funding_rates(
        self,
        symbol: str,
        start: str,
        end: str,
    ) -> pd.Series:
        cache_path = self._cache_key_funding(symbol, start, end)
        if os.path.exists(cache_path):
            df = pd.read_parquet(cache_path)
            s = df.iloc[:, 0]
            s.index = pd.to_datetime(s.index, utc=True)
            return s

        print(f"Fetching funding rates {symbol} {start} to {end}...")

        since_ms = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

        all_rates = []
        current = since_ms

        while current < end_ms:
            for attempt in range(6):
                try:
                    rates = self._exchange.fetch_funding_rate_history(
                        symbol, since=current, limit=1000
                    )
                    break
                except (ccxt.RateLimitExceeded, ccxt.NetworkError) as e:
                    if attempt == 5:
                        raise DataError(f"Failed fetching funding rates after 5 retries: {e}")
                    time.sleep(2 ** attempt)

            if not rates:
                break

            all_rates.extend(rates)
            last_ts = rates[-1]["timestamp"]
            if last_ts <= current:
                break
            current = last_ts + 1

        if not all_rates:
            raise DataError(f"No funding rate data for {symbol} {start}-{end}")

        records = [
            {"timestamp": r["timestamp"], "funding_rate": r["fundingRate"]}
            for r in all_rates
        ]
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()

        series = df["funding_rate"].astype(np.float64)
        df.to_parquet(cache_path)
        return series

    def get_multi(
        self,
        symbols: list[str],
        timeframe: str,
        start: str,
        end: str,
    ) -> dict[str, pd.DataFrame]:
        results = {}

        def fetch(sym):
            return sym, self.get_ohlcv(sym, timeframe, start, end)

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(fetch, s): s for s in symbols}
            for future in as_completed(futures):
                sym, df = future.result()
                results[sym] = df

        return results

    def validate(self, df: pd.DataFrame) -> bool:
        if df.empty:
            raise DataError("DataFrame is empty")

        # Check required columns
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise DataError(f"Missing columns: {missing}")

        # No NaN
        nan_cols = df.columns[df.isna().any()].tolist()
        if nan_cols:
            raise DataError(f"NaN values found in columns: {nan_cols}")

        # No duplicate timestamps
        if df.index.duplicated().any():
            raise DataError("Duplicate timestamps found")

        # Monotonically increasing
        if not df.index.is_monotonic_increasing:
            raise DataError("Timestamps are not monotonically increasing")

        # OHLC relationships
        if (df["high"] < df["low"]).any():
            raise DataError("high < low detected")
        if (df["high"] < df["open"]).any():
            raise DataError("high < open detected")
        if (df["high"] < df["close"]).any():
            raise DataError("high < close detected")
        if (df["low"] > df["open"]).any():
            raise DataError("low > open detected")
        if (df["low"] > df["close"]).any():
            raise DataError("low > close detected")

        return True
