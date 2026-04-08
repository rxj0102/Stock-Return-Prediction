"""
features.py
-----------
Feature engineering for stock return prediction.

Every feature is constructed from information available *at close on day t*
only.  The target (next-day log return) is appended at the end via a
``shift(-1)`` — the sole intentional look-ahead.

Public API
~~~~~~~~~~
``build_features(df_raw)`` → ``pd.DataFrame``
    Full pipeline: accepts raw OHLCV DataFrame, returns a clean feature
    DataFrame with 80+ columns plus a ``Target`` column.

``prepare_matrices(df)`` → ``(X, y)``
    Drops OHLCV and target columns, returns numpy-ready X and y with NaNs
    removed.
"""

import numpy as np
import pandas as pd
import ta

from .config import RETURN_LAGS, ROLLING_WINDOWS, MOM_WINDOWS


# ── Internal helpers ──────────────────────────────────────────────────────────

def _add_price_derivations(df: pd.DataFrame) -> pd.DataFrame:
    """Log return and scale-free price-structure ratios."""
    df["Log_Return"]    = np.log(df["Close"] / df["Close"].shift(1))
    df["HL_Pct"]        = (df["High"] - df["Low"]) / df["Close"]
    df["OC_Pct"]        = (df["Close"] - df["Open"]) / df["Open"]
    df["Typical_Price"] = (df["High"] + df["Low"] + df["Close"]) / 3
    return df


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical sin/cos encoding for day-of-week and month.

    Avoids the distance artefact that arises with ordinal encoding
    (Monday=0, Friday=4 should be adjacent in feature space).
    """
    df["DOW_sin"]   = np.sin(2 * np.pi * df.index.dayofweek / 5)
    df["DOW_cos"]   = np.cos(2 * np.pi * df.index.dayofweek / 5)
    df["Month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    df["Quarter"]   = df.index.quarter
    return df


def _add_lag_features(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """Return, volume, and high-low lags on stationary series only."""
    for lag in lags:
        df[f"Ret_Lag_{lag}"] = df["Log_Return"].shift(lag)
        df[f"Vol_Lag_{lag}"] = df["Volume"].shift(lag)
        df[f"HL_Lag_{lag}"]  = df["HL_Pct"].shift(lag)
    return df


def _add_rolling_statistics(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Rolling means, volatilities, and volume z-scores."""
    for w in windows:
        df[f"MA_{w}"]      = df["Close"].rolling(w).mean()
        df[f"Ret_MA_{w}"]  = df["Log_Return"].rolling(w).mean()
        df[f"Ret_Vol_{w}"] = df["Log_Return"].rolling(w).std()

    vol_roll20_mean = df["Volume"].rolling(20).mean()
    vol_roll20_std  = df["Volume"].rolling(20).std()
    df["Vol_Rel"]    = df["Volume"] / vol_roll20_mean
    df["Vol_ZScore"] = (df["Volume"] - vol_roll20_mean) / vol_roll20_std
    return df


def _add_momentum(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Price momentum (scale-free) and return acceleration."""
    for w in windows:
        df[f"Mom_{w}"] = df["Close"] / df["Close"].shift(w) - 1
    df["Ret_Accel"] = df["Log_Return"].diff()   # second derivative of price
    return df


def _add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Standard technical indicators via the `ta` library."""
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_Width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    df["BB_Pct"]   = bb.bollinger_pband()

    # MACD
    macd = ta.trend.MACD(close)
    df["MACD"]      = macd.macd()
    df["MACD_Sig"]  = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    # Momentum oscillators
    df["RSI"]      = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["Stoch"]    = ta.momentum.StochasticOscillator(high, low, close).stoch()
    df["Williams"] = ta.momentum.WilliamsRIndicator(high, low, close).williams_r()

    # Trend strength
    adx = ta.trend.ADXIndicator(high, low, close, window=14)
    df["ADX"]     = adx.adx()
    df["ADX_Pos"] = adx.adx_pos()
    df["ADX_Neg"] = adx.adx_neg()

    # Other trend / cycle indicators
    df["CCI"]      = ta.trend.CCIIndicator(high, low, close).cci()
    aroon          = ta.trend.AroonIndicator(high, low, window=25)
    df["Aroon_Osc"] = aroon.aroon_up() - aroon.aroon_down()

    # Volume-based
    df["OBV"]    = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    df["MFI"]    = ta.volume.MFIIndicator(high, low, close, volume).money_flow_index()

    # Volatility
    df["ATR_Pct"] = (
        ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
        / close
    )
    return df


def _add_market_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Support/resistance, regime, volume-price correlation, candlestick proxies."""
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    df["Dist_Resist"]  = (high.rolling(20).max() - close) / close
    df["Dist_Support"] = (close - low.rolling(20).min()) / close
    df["VP_Corr"]      = volume.rolling(20).corr(close)
    df["Regime"]       = (df["MA_20"] > df["MA_50"]).astype(int)
    df["Higher_High"]  = (high > high.shift(1)).astype(int)
    df["Lower_Low"]    = (low  < low.shift(1)).astype(int)
    return df


# ── Public API ────────────────────────────────────────────────────────────────

def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature-engineering pipeline on raw OHLCV data.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw price data with columns ``['Open', 'High', 'Low', 'Close', 'Volume']``
        and a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Original OHLCV columns plus 80+ engineered features and a ``Target``
        column (next-day log return).  Rows with NaN in any column are
        **not** dropped here — call ``prepare_matrices`` for that.
    """
    df = df_raw.copy()
    df = _add_price_derivations(df)
    df = _add_calendar_features(df)
    df = _add_lag_features(df, RETURN_LAGS)
    df = _add_rolling_statistics(df, ROLLING_WINDOWS)
    df = _add_momentum(df, MOM_WINDOWS)
    df = _add_technical_indicators(df)
    df = _add_market_structure(df)

    # Target: next-day log return — the only intentional look-ahead
    df["Target"] = df["Log_Return"].shift(-1)

    return df


# Raw OHLCV columns — present in df_raw but not informative as model features
# (all relevant information is captured by the normalised engineered columns).
_OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


def prepare_matrices(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract the clean feature matrix X and target vector y.

    Drops raw OHLCV columns, ``Log_Return`` (t=0 leaks the target at t=1),
    and any rows with NaN values introduced by lag/rolling operations.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`build_features`.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (DatetimeIndex preserved).
    y : pd.Series
        Next-day log-return target aligned with *X*.
    """
    drop_cols = _OHLCV_COLS + ["Log_Return", "Target", "Typical_Price"]
    df_clean = df.dropna().copy()

    X = df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns])
    y = df_clean["Target"]
    return X, y
