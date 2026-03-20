"""Tests for DataModule."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from crypto_infra.data_module import DataModule, DataError


@pytest.fixture
def data_module(tmp_path):
    return DataModule(cache_dir=str(tmp_path / "cache"))


def test_ohlcv_fetch_and_cache(data_module):
    """Fetch 30 days of BTC 1h data, check shape/columns/NaN/OHLC, confirm cache."""
    df = data_module.get_ohlcv("BTC/USDT:USDT", "1h", "2024-06-01", "2024-07-01")

    # Check columns
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in df.columns, f"Missing column: {col}"

    # Check no NaN
    assert not df.isna().any().any(), "NaN values found"

    # Check reasonable row count (30 days * 24 hours = ~720)
    assert len(df) > 600, f"Too few rows: {len(df)}"

    # Check OHLC relationships
    assert (df["high"] >= df["low"]).all(), "high < low found"
    assert (df["high"] >= df["open"]).all(), "high < open found"
    assert (df["high"] >= df["close"]).all(), "high < close found"

    # Check index is DatetimeIndex
    assert isinstance(df.index, pd.DatetimeIndex)

    # Check cache file exists
    cache_files = os.listdir(data_module.cache_dir)
    assert len(cache_files) >= 1, "No cache file created"

    # Fetch again - should load from cache (no network call)
    df2 = data_module.get_ohlcv("BTC/USDT:USDT", "1h", "2024-06-01", "2024-07-01")
    assert len(df2) == len(df)


def test_funding_rate_fetch(data_module):
    """Fetch 30 days of BTC funding rates."""
    rates = data_module.get_funding_rates("BTC/USDT:USDT", "2024-06-01", "2024-07-01")

    assert isinstance(rates.index, pd.DatetimeIndex)
    assert len(rates) > 0, "No funding rates returned"

    # Values should be reasonable (between -1% and 1% per 8h)
    assert (rates.abs() < 0.01).all(), f"Unreasonable funding rate: {rates.abs().max()}"


def test_data_validation_catches_nan(data_module):
    """DataFrame with NaN values should raise DataError."""
    df = pd.DataFrame({
        "open": [1.0, np.nan, 3.0],
        "high": [2.0, 3.0, 4.0],
        "low": [0.5, 1.0, 2.0],
        "close": [1.5, 2.5, 3.5],
        "volume": [100, 200, 300],
    }, index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"))

    with pytest.raises(DataError, match="NaN"):
        data_module.validate(df)


def test_data_validation_catches_duplicates(data_module):
    """DataFrame with duplicate timestamps should raise DataError."""
    idx = pd.DatetimeIndex(
        ["2024-01-01 00:00", "2024-01-01 00:00", "2024-01-01 02:00"],
        tz="UTC",
    )
    df = pd.DataFrame({
        "open": [1.0, 2.0, 3.0],
        "high": [2.0, 3.0, 4.0],
        "low": [0.5, 1.0, 2.0],
        "close": [1.5, 2.5, 3.5],
        "volume": [100, 200, 300],
    }, index=idx)

    with pytest.raises(DataError, match="Duplicate"):
        data_module.validate(df)
