"""
tests/test_strategy.py
-----------------------
Unit tests for src/strategy.py — risk metrics and backtest logic.
"""

import numpy as np
import pytest

from src.strategy import (
    BacktestResult,
    max_drawdown,
    run_backtest,
    sharpe_ratio,
    sortino_ratio,
)


# ── sharpe_ratio ──────────────────────────────────────────────────────────────

class TestSharpeRatio:

    def test_zero_std_returns_zero(self):
        assert sharpe_ratio(np.ones(100) * 0.01) == pytest.approx(0.0)

    def test_positive_mean_positive_sharpe(self):
        returns = np.ones(252) * 0.001  # constant positive daily return
        # std = 0, so result is 0 by convention
        assert sharpe_ratio(returns) == 0.0

    def test_iid_normal(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0.001, 0.01, 252)
        s = sharpe_ratio(r, ann=252)
        # For mean≈0.001, std≈0.01 → SR ≈ 0.001/0.01 * sqrt(252) ≈ 1.58
        assert s == pytest.approx(r.mean() / r.std() * np.sqrt(252), rel=1e-6)

    def test_annualisation_scaling(self):
        rng = np.random.default_rng(1)
        r = rng.normal(0.001, 0.01, 100)
        s252 = sharpe_ratio(r, ann=252)
        s126 = sharpe_ratio(r, ann=126)
        assert s252 == pytest.approx(s126 * np.sqrt(2), rel=1e-6)


# ── sortino_ratio ─────────────────────────────────────────────────────────────

class TestSortinoRatio:

    def test_no_negative_returns_returns_zero(self):
        returns = np.abs(np.random.default_rng(0).normal(0.001, 0.01, 100))
        assert sortino_ratio(returns) == pytest.approx(0.0)

    def test_greater_than_sharpe_for_positive_skew(self):
        """Sortino >= Sharpe when downside vol < total vol."""
        rng = np.random.default_rng(2)
        r = rng.normal(0.001, 0.01, 500)
        assert sortino_ratio(r) >= sharpe_ratio(r) - 1e-9  # allow fp tolerance


# ── max_drawdown ──────────────────────────────────────────────────────────────

class TestMaxDrawdown:

    def test_monotonically_increasing_no_drawdown(self):
        cum_r = np.linspace(0, 0.5, 100)
        assert max_drawdown(cum_r) == pytest.approx(0.0, abs=1e-10)

    def test_known_drawdown(self):
        # Wealth goes 1 → 1.5 → 1 → from peak 1.5 down to 1 = 33.3% DD
        cum_r = np.array([0.0, 0.5, 0.0])
        assert max_drawdown(cum_r) == pytest.approx(1 / 3, rel=1e-6)

    def test_output_in_unit_interval(self):
        rng = np.random.default_rng(3)
        cum_r = np.cumsum(rng.normal(0, 0.01, 200))
        dd = max_drawdown(cum_r)
        assert 0.0 <= dd <= 1.0


# ── run_backtest ──────────────────────────────────────────────────────────────

class TestRunBacktest:

    def _returns(self, n=252, seed=0):
        rng = np.random.default_rng(seed)
        return rng.normal(0, 0.01, n)

    def test_returns_backtest_result(self):
        y = self._returns()
        result = run_backtest(y, y + 0.001)
        assert isinstance(result, BacktestResult)

    def test_always_in_market(self):
        """If all predictions exceed threshold, every day is in market."""
        y    = self._returns()
        pred = np.ones(len(y)) * 0.05   # well above any reasonable threshold
        result = run_backtest(y, pred, threshold=0.001)
        assert result.days_in_market == len(y)

    def test_never_in_market(self):
        """If no prediction exceeds threshold, no trades, zero strategy return."""
        y    = self._returns()
        pred = -np.abs(y) - 1.0        # all deeply negative
        result = run_backtest(y, pred, threshold=0.001)
        assert result.days_in_market == 0
        assert result.total_return == pytest.approx(0.0)

    def test_transaction_cost_reduces_return(self):
        """Higher transaction cost should produce lower (or equal) total return."""
        y    = self._returns()
        pred = y + 0.002
        r_no_cost   = run_backtest(y, pred, transaction_cost=0.0)
        r_with_cost = run_backtest(y, pred, transaction_cost=0.01)
        assert r_with_cost.total_return <= r_no_cost.total_return

    def test_cum_strategy_length(self):
        n = 200
        y = self._returns(n)
        result = run_backtest(y, y)
        assert len(result.cum_strategy) == n
        assert len(result.cum_bnh) == n

    def test_summary_dict_populated(self):
        y = self._returns()
        result = run_backtest(y, y + 0.001)
        assert "Total Return (%)" in result.summary
        assert "Annualised Sharpe" in result.summary
        assert "Max Drawdown (%)" in result.summary

    def test_win_rate_bounded(self):
        y = self._returns()
        result = run_backtest(y, y + 0.001)
        assert 0.0 <= result.win_rate <= 1.0

    def test_n_trades_non_negative(self):
        y = self._returns()
        result = run_backtest(y, y)
        assert result.n_trades >= 0
