"""Can tomorrow's return be predicted from past returns?

Design, fixed before any model was run:

  series     daily excess returns of the US market, of the largest tenth of stocks and of the
             smallest tenth (Ken French Data Library, from July 1926)
  target     the next day's excess return
  features   17 numbers computed from the series' own past returns up to today (FEATURES)
  training   the most recent 10 years (2,520 days); every model is refitted once a year
  models     Zero        always predicts 0, the benchmark
             Mean        the average return of the training window
             AR(1)       regression on yesterday's return
             Ridge       linear in all features, fixed penalty
             Boosting    gradient boosted trees on all features, fixed settings
  trading    hold the series when the forecast is positive, cash otherwise. The position for
             day t+1 is set at the close of day t. Costs are charged per unit traded.

Nothing is tuned: no hyperparameter search, no feature selection, no model picked on test data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

MODELS = ["Zero", "Mean", "AR(1)", "Ridge", "Boosting"]
LAGS = (1, 2, 3, 4, 5)
MEANS = (5, 20, 60, 250)
VOLS = (5, 20, 60)
FEATURES = ([f"ret_lag{k}" for k in LAGS] + [f"mean_{w}" for w in MEANS] + [f"vol_{w}" for w in VOLS]
            + ["abs_ret", "drawdown_250", "gap_days", "up_streak", "down_streak"])
# Break dates fixed by regulation, not chosen from the data: the NYSE minimum tick went from
# eighths to sixteenths on 1997-06-24 and to pennies on 2001-01-29 (Nasdaq followed on
# 2001-04-09). Chordia, Roll and Subrahmanyam (2008) tie the fall in short-horizon
# predictability to these cuts.
TICK_REGIMES = [("Eighths", "1970-01-01", "1997-06-23"), ("Sixteenths", "1997-06-24", "2001-01-26"),
                ("Decimals", "2001-01-29", "2100-12-31")]
ERAS = [("1935-1969", "1935", "1969"), ("1970-1999", "1970", "1999"), ("2000-2012", "2000", "2012"),
        ("2013-2026", "2013", "2026")]


@dataclass(frozen=True)
class Config:
    train: int = 2520
    refit: int = 252
    ridge_alpha: float = 252.0
    gb_depth: int = 3
    gb_trees: int = 200
    gb_rate: float = 0.03
    gb_leaf: int = 100
    seed: int = 0
    cost: float = 0.0005
    nw_lags: int = 5
    block: int = 21
    n_boot: int = 1000


def streak(up: np.ndarray) -> np.ndarray:
    """Length of the current run of True values ending at each position."""
    out = np.zeros(len(up))
    run = 0
    for i, u in enumerate(up):
        run = run + 1 if u else 0
        out[i] = run
    return out


def build_features(r: pd.Series) -> pd.DataFrame:
    """Row t holds only what is known at the close of day t."""
    f = pd.DataFrame(index=r.index)
    for k in LAGS:
        f[f"ret_lag{k}"] = r.shift(k - 1)             # ret_lag1 is today's return
    for w in MEANS:
        f[f"mean_{w}"] = r.rolling(w).mean()
    for w in VOLS:
        f[f"vol_{w}"] = r.rolling(w).std()
    f["abs_ret"] = r.abs()
    level = np.log1p(r).cumsum()
    f["drawdown_250"] = level - level.rolling(250).max()
    f["gap_days"] = pd.Series(r.index, index=r.index).diff().dt.days.clip(upper=5)
    f["up_streak"] = streak((r > 0).to_numpy())
    f["down_streak"] = streak((r < 0).to_numpy())
    return f[FEATURES]


def fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, cfg: Config) -> dict:
    """Fit every model on the training rows and forecast the test rows."""
    out = {"Zero": np.zeros(len(Xte)), "Mean": np.full(len(Xte), ytr.mean())}
    x1, xm, ym = Xtr[:, 0], Xtr[:, 0].mean(), ytr.mean()
    slope = float(((x1 - xm) * (ytr - ym)).sum() / ((x1 - xm) ** 2).sum())
    out["AR(1)"] = ym + slope * (Xte[:, 0] - xm)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd[sd == 0] = 1.0
    Z, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    beta = np.linalg.solve(Z.T @ Z + cfg.ridge_alpha * np.eye(Z.shape[1]), Z.T @ (ytr - ym))
    out["Ridge"] = ym + Zte @ beta
    gb = HistGradientBoostingRegressor(max_depth=cfg.gb_depth, max_iter=cfg.gb_trees, learning_rate=cfg.gb_rate,
                                       min_samples_leaf=cfg.gb_leaf, l2_regularization=1.0,
                                       early_stopping=False, random_state=cfg.seed)
    out["Boosting"] = gb.fit(Xtr, ytr).predict(Xte)
    return out


def walk_forward(r: pd.Series, cfg: Config = Config()) -> pd.DataFrame:
    """Forecasts of the next day's return. Row t is the forecast made at the close of day t.

    Columns: one per model, plus target (the return of day t+1). A model refitted at position
    s is trained on feature rows whose targets are all realised by day s-1.
    """
    X = build_features(r)
    y = r.shift(-1)
    ok = X.notna().all(axis=1) & y.notna()
    X, y = X[ok].to_numpy(), y[ok]
    preds = {m: np.full(len(y), np.nan) for m in MODELS}
    for s in range(cfg.train, len(y), cfg.refit):
        e = min(s + cfg.refit, len(y))
        # training rows s-train .. s-2: the last training target is the return of the day at
        # row s-1, realised before the first test forecast is made at the close of the day at row s
        p = fit_predict(X[s - cfg.train:s - 1], y.to_numpy()[s - cfg.train:s - 1], X[s:e], cfg)
        for m in MODELS:
            preds[m][s:e] = p[m]
    out = pd.DataFrame(preds, index=y.index)
    out["target"] = y
    return out.iloc[cfg.train:]


def newey_west_t(d: np.ndarray, lags: int) -> float:
    n = len(d)
    e = d - d.mean()
    v = float(e @ e) / n
    for k in range(1, lags + 1):
        v += 2.0 * (1.0 - k / (lags + 1.0)) * float(e[k:] @ e[:-k]) / n
    return float(d.mean() / np.sqrt(v / n)) if v > 0 else float("nan")


def forecast_stats(pred: np.ndarray, y: np.ndarray, cfg: Config) -> dict:
    """Accuracy against the zero forecast. Clark and West (2007) t for nested models."""
    sse0 = float((y ** 2).sum())
    cw = 2.0 * y * pred                      # y^2 - (y - p)^2 + p^2
    return {"r2_oos": 1.0 - float(((y - pred) ** 2).sum()) / sse0,
            "clark_west_t": newey_west_t(cw, cfg.nw_lags) if np.any(pred != 0) else float("nan"),
            "direction": float(((pred > 0) == (y > 0)).mean()),
            "corr": float(np.corrcoef(pred, y)[0, 1]) if pred.std() > 0 else float("nan")}


def sharpe(x: np.ndarray) -> float:
    sd = x.std(ddof=1)
    return float(np.sqrt(252) * x.mean() / sd) if sd > 0 else float("nan")


def strategy_returns(pred: np.ndarray, y: np.ndarray, cost: float) -> tuple[np.ndarray, np.ndarray]:
    """Long when the forecast is positive, cash otherwise. Returns (net excess return, turnover)."""
    pos = (pred > 0).astype(float)
    turn = np.abs(np.diff(pos, prepend=0.0))
    return pos * y - cost * turn, turn


def break_even_cost(pred: np.ndarray, y: np.ndarray) -> float:
    """Cost per unit traded at which the strategy's Sharpe ratio falls to buy and hold's."""
    target = sharpe(y)
    if sharpe(strategy_returns(pred, y, 0.0)[0]) <= target:
        return 0.0
    lo, hi = 0.0, 0.05
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sharpe(strategy_returns(pred, y, mid)[0]) > target else (lo, mid)
    return lo


def sharpe_diff_interval(a: np.ndarray, b: np.ndarray, cfg: Config) -> dict:
    """Paired circular block bootstrap of Sharpe(a) - Sharpe(b)."""
    rng = np.random.default_rng(cfg.seed)
    n = len(a)
    n_blocks = int(np.ceil(n / cfg.block))
    offsets = np.arange(cfg.block)
    draws = np.empty(cfg.n_boot)
    for i in range(cfg.n_boot):
        idx = ((rng.integers(0, n, n_blocks)[:, None] + offsets) % n).ravel()[:n]
        draws[i] = sharpe(a[idx]) - sharpe(b[idx])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"diff": sharpe(a) - sharpe(b), "lo": float(lo), "hi": float(hi)}


def trading_stats(pred: np.ndarray, y: np.ndarray, cfg: Config, boot: bool = True) -> dict:
    gross, turn = strategy_returns(pred, y, 0.0)
    net, _ = strategy_returns(pred, y, cfg.cost)
    out = {"sharpe_gross": sharpe(gross), "sharpe_net": sharpe(net), "time_invested": float((pred > 0).mean()),
           "trades_per_year": float(turn.mean() * 252), "break_even_cost_bps": break_even_cost(pred, y) * 1e4}
    if boot:
        out["net_minus_buy_hold"] = sharpe_diff_interval(net, y, cfg)
    return out


def loss_diff_t(pred: np.ndarray, base: np.ndarray, y: np.ndarray, cfg: Config) -> float:
    """Diebold-Mariano t of squared error: negative means pred is more accurate than base."""
    return newey_west_t((y - pred) ** 2 - (y - base) ** 2, cfg.nw_lags)


def rolling_r2(pred: pd.Series, y: pd.Series, window: int = 2520) -> pd.Series:
    """Out-of-sample R2 against the zero forecast over a trailing window."""
    num = ((y - pred) ** 2).rolling(window).sum()
    return (1.0 - num / (y ** 2).rolling(window).sum()).dropna()
