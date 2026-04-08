"""
tests/test_features.py
----------------------
Unit tests for src/features.py.

Tests verify correctness of feature construction without requiring a network
connection or real market data (synthetic OHLCV is generated inline).
"""

import numpy as np
import pandas as pd
import pytest

from src.features import build_features, prepare_matrices


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Return a minimal synthetic OHLCV DataFrame with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    dates  = pd.date_range("2020-01-01", periods=n, freq="B")
    close  = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    noise  = rng.uniform(0.001, 0.01, n)
    high   = close * (1 + noise)
    low    = close * (1 - noise)
    open_  = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture
def raw_df():
    return _make_ohlcv()


# ── build_features ────────────────────────────────────────────────────────────

class TestBuildFeatures:

    def test_returns_dataframe(self, raw_df):
        result = build_features(raw_df)
        assert isinstance(result, pd.DataFrame)

    def test_index_preserved(self, raw_df):
        result = build_features(raw_df)
        assert result.index.equals(raw_df.index)

    def test_target_column_present(self, raw_df):
        result = build_features(raw_df)
        assert "Target" in result.columns

    def test_target_is_next_day_log_return(self, raw_df):
        result = build_features(raw_df)
        log_ret = np.log(raw_df["Close"] / raw_df["Close"].shift(1))
        expected = log_ret.shift(-1)
        # Compare where both are non-NaN
        mask = result["Target"].notna() & expected.notna()
        np.testing.assert_allclose(
            result["Target"][mask].values,
            expected[mask].values,
            rtol=1e-10,
        )

    def test_cyclical_calendar_bounded(self, raw_df):
        result = build_features(raw_df)
        for col in ("DOW_sin", "DOW_cos", "Month_sin", "Month_cos"):
            assert col in result.columns
            assert result[col].between(-1, 1).all(), f"{col} out of [-1, 1]"

    def test_no_look_ahead_in_lags(self, raw_df):
        """Lag features at row i must only use data from rows < i."""
        result = build_features(raw_df)
        # Ret_Lag_1 at index i should equal Log_Return at index i-1
        log_ret = result["Log_Return"]
        lag1    = result["Ret_Lag_1"]
        mask = lag1.notna() & log_ret.shift(1).notna()
        np.testing.assert_allclose(
            lag1[mask].values,
            log_ret.shift(1)[mask].values,
            rtol=1e-10,
        )

    def test_ohlcv_columns_still_present(self, raw_df):
        """build_features should not drop OHLCV — prepare_matrices does that."""
        result = build_features(raw_df)
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in result.columns

    def test_feature_count_reasonable(self, raw_df):
        result = build_features(raw_df)
        # Should have well more than 20 engineered columns
        assert result.shape[1] > 40


# ── prepare_matrices ──────────────────────────────────────────────────────────

class TestPrepareMatrices:

    def test_returns_x_and_y(self, raw_df):
        df = build_features(raw_df)
        X, y = prepare_matrices(df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_no_nans_in_output(self, raw_df):
        df = build_features(raw_df)
        X, y = prepare_matrices(df)
        assert not X.isnull().any().any(), "X contains NaN"
        assert not y.isnull().any(),        "y contains NaN"

    def test_ohlcv_dropped(self, raw_df):
        df = build_features(raw_df)
        X, _ = prepare_matrices(df)
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col not in X.columns, f"Raw column {col!r} was not dropped"

    def test_log_return_dropped(self, raw_df):
        """Current-day Log_Return would leak the target."""
        df = build_features(raw_df)
        X, _ = prepare_matrices(df)
        assert "Log_Return" not in X.columns

    def test_x_y_same_length(self, raw_df):
        df = build_features(raw_df)
        X, y = prepare_matrices(df)
        assert len(X) == len(y)

    def test_index_alignment(self, raw_df):
        df = build_features(raw_df)
        X, y = prepare_matrices(df)
        assert X.index.equals(y.index)
