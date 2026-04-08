"""
data.py
-------
Data loading, caching, and train/test splitting utilities.

All functions are pure (no side effects beyond the CSV cache) and accept
explicit arguments so they can be used from notebooks or scripts without
relying on global state.
"""

import os
import pandas as pd
import yfinance as yf

from .config import CACHE_DIR


def load_price_data(
    ticker: str,
    start: str,
    end: str,
    cache_dir: str = CACHE_DIR,
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance, caching to CSV to avoid rate limits.

    On the first call the data is fetched from yfinance and persisted to
    ``<cache_dir>/<ticker>_<start_year>_<end_year>.csv``.  Subsequent calls
    load from the cache file instead.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (e.g. ``'AAPL'``).
    start : str
        Start date in ``'YYYY-MM-DD'`` format (inclusive).
    end : str
        End date in ``'YYYY-MM-DD'`` format (exclusive).
    cache_dir : str
        Directory in which to store the CSV cache.

    Returns
    -------
    pd.DataFrame
        DataFrame with a DatetimeIndex and columns
        ``['Open', 'High', 'Low', 'Close', 'Volume']``.

    Raises
    ------
    AssertionError
        If the downloaded data contains missing values.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(
        cache_dir, f"{ticker}_{start[:4]}_{end[:4]}.csv"
    )

    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    else:
        df = yf.download(ticker, start=start, end=end, progress=False)
        # yfinance ≥0.2 returns MultiIndex columns — flatten them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # Normalise column order regardless of yfinance version
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.to_csv(cache_file)

    assert df.isnull().sum().sum() == 0, (
        f"Missing values detected in {ticker} data — check the cache or date range."
    )
    return df


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    split_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target by calendar date (no shuffling).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with a DatetimeIndex.
    y : pd.Series
        Target series aligned with *X*.
    split_date : str
        ISO date string; rows before this date go to train, the rest to test.

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple of DataFrames / Series
    """
    mask = X.index < split_date
    return X[mask], X[~mask], y[mask], y[~mask]
