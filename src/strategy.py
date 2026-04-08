"""
strategy.py
-----------
Simple long-only trading strategy and associated risk/return metrics.

Strategy rules
~~~~~~~~~~~~~~
- Enter a long position on day *t+1* when the predicted log return for *t+1*
  exceeds ``threshold``.
- Apply a flat ``transaction_cost`` (round-trip) on every position change.
- No short selling; no position sizing (equal-weight, binary signal).

Public API
~~~~~~~~~~
``run_backtest(y_true, y_pred, threshold, transaction_cost)``
    Returns ``BacktestResult`` with raw daily P&L, cumulative returns, and
    summary metrics.

``sharpe_ratio``, ``sortino_ratio``, ``max_drawdown``
    Standalone risk metrics for use in other contexts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ANNUALISATION, SIGNAL_THRESHOLD, TRANSACTION_COST


# ── Risk metrics ──────────────────────────────────────────────────────────────

def sharpe_ratio(returns: np.ndarray, ann: int = ANNUALISATION) -> float:
    """Annualised Sharpe ratio (assumes zero risk-free rate).

    Parameters
    ----------
    returns : np.ndarray
        Daily log returns of the strategy.
    ann : int
        Number of trading days per year.

    Returns
    -------
    float — 0.0 if standard deviation is zero.
    """
    std = returns.std()
    return float((returns.mean() / std) * np.sqrt(ann)) if std > 1e-10 else 0.0


def sortino_ratio(returns: np.ndarray, ann: int = ANNUALISATION) -> float:
    """Annualised Sortino ratio (penalises downside volatility only).

    Parameters
    ----------
    returns : np.ndarray
        Daily log returns of the strategy.
    ann : int
        Number of trading days per year.

    Returns
    -------
    float — 0.0 if downside deviation is zero.
    """
    downside_std = returns[returns < 0].std()
    return float((returns.mean() / downside_std) * np.sqrt(ann)) if downside_std > 1e-10 else 0.0


def max_drawdown(cum_returns: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of a cumulative-return series.

    Parameters
    ----------
    cum_returns : np.ndarray
        Cumulative returns (e.g. output of ``np.cumsum`` on log returns,
        then ``np.exp``-ed, minus 1).

    Returns
    -------
    float in [0, 1] — 0 means no drawdown ever.
    """
    wealth = 1.0 + cum_returns
    peak   = np.maximum.accumulate(wealth)
    return float(((peak - wealth) / peak).max())


# ── Backtest ──────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """Container for all outputs of :func:`run_backtest`."""
    signals:           np.ndarray
    strategy_returns:  np.ndarray
    bnh_returns:       np.ndarray
    cum_strategy:      np.ndarray
    cum_bnh:           np.ndarray
    total_return:      float
    bnh_total_return:  float
    sharpe:            float
    sortino:           float
    max_dd:            float
    bnh_max_dd:        float
    bnh_sharpe:        float
    win_rate:          float
    days_in_market:    int
    n_trades:          int
    summary: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.summary = {
            "Total Return (%)":      round(self.total_return * 100, 2),
            "B&H Total Return (%)":  round(self.bnh_total_return * 100, 2),
            "Annualised Sharpe":     round(self.sharpe, 2),
            "B&H Sharpe":            round(self.bnh_sharpe, 2),
            "Sortino Ratio":         round(self.sortino, 2),
            "Max Drawdown (%)":      round(self.max_dd * 100, 2),
            "B&H Max Drawdown (%)":  round(self.bnh_max_dd * 100, 2),
            "Win Rate (active, %)":  round(self.win_rate * 100, 2),
            "Days in Market":        self.days_in_market,
            "Number of Trades":      self.n_trades,
        }


def run_backtest(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = SIGNAL_THRESHOLD,
    transaction_cost: float = TRANSACTION_COST,
) -> BacktestResult:
    """Simulate a long-only strategy and compute risk/return statistics.

    Parameters
    ----------
    y_true : np.ndarray
        Realised next-day log returns.
    y_pred : np.ndarray
        Model's predicted next-day log returns.
    threshold : float
        Minimum predicted return required to enter a position.
    transaction_cost : float
        Round-trip cost applied on every position change (both entry and exit).

    Returns
    -------
    BacktestResult
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    signals          = (y_pred > threshold).astype(int)
    strategy_returns = y_true * signals
    # Deduct transaction cost whenever the position changes
    strategy_returns -= np.abs(np.diff(signals, prepend=0)) * transaction_cost

    cum_strategy = np.exp(np.cumsum(strategy_returns)) - 1
    cum_bnh      = np.exp(np.cumsum(y_true)) - 1

    active   = strategy_returns[signals == 1]
    win_rate = float(np.mean(active > 0)) if len(active) > 0 else 0.0

    return BacktestResult(
        signals=signals,
        strategy_returns=strategy_returns,
        bnh_returns=y_true,
        cum_strategy=cum_strategy,
        cum_bnh=cum_bnh,
        total_return=float(cum_strategy[-1]),
        bnh_total_return=float(cum_bnh[-1]),
        sharpe=sharpe_ratio(strategy_returns),
        sortino=sortino_ratio(strategy_returns),
        max_dd=max_drawdown(cum_strategy),
        bnh_max_dd=max_drawdown(cum_bnh),
        bnh_sharpe=sharpe_ratio(y_true),
        win_rate=win_rate,
        days_in_market=int(signals.sum()),
        n_trades=int(np.abs(np.diff(signals, prepend=0)).sum()),
    )


def print_backtest_summary(result: BacktestResult) -> None:
    """Print a formatted comparison table of strategy vs buy-and-hold."""
    print(f"\n{'Metric':<28} {'Strategy':>12} {'Buy & Hold':>12}")
    print("-" * 54)
    rows = [
        ("Total Return (%)",       f"{result.total_return*100:.2f}%",      f"{result.bnh_total_return*100:.2f}%"),
        ("Annualised Sharpe",      f"{result.sharpe:.2f}",                 f"{result.bnh_sharpe:.2f}"),
        ("Sortino Ratio",          f"{result.sortino:.2f}",                "—"),
        ("Max Drawdown (%)",       f"{result.max_dd*100:.2f}%",            f"{result.bnh_max_dd*100:.2f}%"),
        ("Win Rate (active, %)",   f"{result.win_rate*100:.1f}%",          "—"),
        ("Days in Market",         str(result.days_in_market),             str(len(result.signals))),
        ("Number of Trades",       str(result.n_trades),                   "—"),
    ]
    for label, strat, bnh in rows:
        print(f"  {label:<26} {strat:>12} {bnh:>12}")


def plot_backtest(
    result: BacktestResult,
    dates: pd.DatetimeIndex,
) -> None:
    """Plot cumulative returns and drawdown for the strategy vs buy-and-hold.

    Parameters
    ----------
    result : BacktestResult
        Output of :func:`run_backtest`.
    dates : pd.DatetimeIndex
        Test-period dates aligned with ``result.cum_strategy``.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fmt = mdates.DateFormatter("%Y-%m")

    ax1.plot(dates, result.cum_bnh * 100, label="Buy & Hold", lw=2)
    ax1.plot(dates, result.cum_strategy * 100, label="Strategy", lw=2, linestyle="--")
    ax1.set_ylabel("Cumulative Return (%)")
    ax1.set_title("Strategy vs Buy & Hold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(fmt)
    ax1.tick_params(axis="x", rotation=30)

    wealth = 1 + result.cum_strategy
    peak   = np.maximum.accumulate(wealth)
    dd_pct = (wealth - peak) / peak * 100
    ax2.fill_between(dates, dd_pct, 0, alpha=0.6, color="#e74c3c")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_title("Strategy Drawdown")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(fmt)
    ax2.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.show()
