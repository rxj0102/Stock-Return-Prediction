"""Checks on the study: no look-ahead, features, models, scoring and the strategy."""
import numpy as np
import pandas as pd
import pytest

from src import study
from src.data import SERIES, build, load, parse_block
from src.study import Config

CFG = Config(train=400, refit=100, gb_trees=20, n_boot=200)


def ar1_series(n=1200, phi=0.3, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 0.01, n)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + e[t]
    return pd.Series(r, index=pd.bdate_range("1990-01-01", periods=n))


@pytest.fixture(scope="module")
def r():
    return ar1_series()


@pytest.fixture(scope="module")
def fc(r):
    return study.walk_forward(r, CFG)


# ---------- data ----------

def test_snapshot_is_clean():
    d = load()
    assert list(d.columns) == SERIES and d.index.is_monotonic_increasing and d.index.is_unique
    assert not d.isna().any().any() and d.abs().max().max() < 0.5
    assert d.index[0].year == 1926


def test_parser_reads_first_table_and_missing_codes():
    text = "junk\n\n,A,B\n19260701,  1.00, -99.99\n19260702, -2.00,  0.50\n\n,A,B\n19260701, 9, 9\n"
    df = parse_block(text)
    assert df.shape == (2, 2) and df["A"].iloc[1] == pytest.approx(-0.02) and np.isnan(df["B"].iloc[0])


def test_build_subtracts_the_bill_rate():
    f = ",Mkt-RF,SMB,HML,RF\n19260701, 0.10, 0, 0, 0.01\n"
    s = "," + ",".join(["<= 0", "Lo 30", "Med 40", "Hi 30", "Lo 20", "Qnt 2", "Qnt 3", "Qnt 4", "Hi 20",
                        "Lo 10", "Dec 2", "Dec 3", "Dec 4", "Dec 5", "Dec 6", "Dec 7", "Dec 8", "Dec 9",
                        "Hi 10"]) + "\n19260701," + ",".join(["0"] * 9 + ["0.51"] + ["0"] * 8 + ["0.21"]) + "\n"
    out = build(f, s)
    assert out.loc["1926-07-01", "small"] == pytest.approx(0.0050)
    assert out.loc["1926-07-01", "large"] == pytest.approx(0.0020)
    assert out.loc["1926-07-01", "market"] == pytest.approx(0.0010)


# ---------- features ----------

def test_features_use_only_the_past(r):
    """Changing returns after day t must leave the feature row of day t unchanged."""
    t = 700
    changed = r.copy()
    changed.iloc[t + 1:] = 0.05
    a, b = study.build_features(r), study.build_features(changed)
    pd.testing.assert_frame_equal(a.iloc[: t + 1], b.iloc[: t + 1])


def test_feature_definitions(r):
    f = study.build_features(r)
    assert list(f.columns) == study.FEATURES and len(study.FEATURES) == 17
    t = 600
    assert f["ret_lag1"].iloc[t] == r.iloc[t] and f["ret_lag3"].iloc[t] == r.iloc[t - 2]
    assert f["mean_20"].iloc[t] == pytest.approx(r.iloc[t - 19:t + 1].mean())
    assert f["vol_5"].iloc[t] == pytest.approx(r.iloc[t - 4:t + 1].std())
    assert (f["drawdown_250"].dropna() <= 1e-12).all()
    assert f["gap_days"].iloc[1:].between(1, 5).all()


def test_streak_counts_runs():
    assert study.streak(np.array([True, True, False, True])).tolist() == [1, 2, 0, 1]


# ---------- the walk-forward loop ----------

def test_target_is_the_next_day(r, fc):
    day = fc.index[10]
    assert fc.loc[day, "target"] == r.shift(-1).loc[day]


def test_forecasts_ignore_the_future(r, fc):
    """Scramble every return after a cut. Forecasts made up to the cut must not move."""
    cut = fc.index[250]
    changed = r.copy()
    after = changed.index > cut
    changed[after] = np.random.default_rng(7).normal(0, 0.03, after.sum())
    again = study.walk_forward(changed, CFG)
    pd.testing.assert_frame_equal(fc.loc[:cut, study.MODELS], again.loc[:cut, study.MODELS])
    assert not np.allclose(fc.loc[cut:, "Boosting"].iloc[150:], again.loc[cut:, "Boosting"].iloc[150:])


def test_training_rows_end_before_the_first_test_target(r):
    """A model that memorises its training targets must not know any test target."""
    seen = {}

    def spy(Xtr, ytr, Xte, cfg):
        seen.setdefault("calls", []).append((ytr.copy(), len(Xte)))
        return {m: np.zeros(len(Xte)) for m in study.MODELS}

    orig, study.fit_predict = study.fit_predict, spy
    try:
        out = study.walk_forward(r, CFG)
    finally:
        study.fit_predict = orig
    y = r.shift(-1)[study.build_features(r).notna().all(axis=1) & r.shift(-1).notna()]
    pos = CFG.train
    for ytr, n_test in seen["calls"]:
        assert len(ytr) == CFG.train - 1
        assert ytr[-1] == y.iloc[pos - 2]              # last target used is realised on day pos-1
        pos += n_test
    assert len(out) == len(y) - CFG.train


def test_models_find_a_real_ar1(fc):
    y = fc["target"].to_numpy()
    for m in ("AR(1)", "Ridge", "Boosting"):
        assert study.forecast_stats(fc[m].to_numpy(), y, CFG)["r2_oos"] > 0.03
    assert (fc["Zero"] == 0).all()


def test_models_find_nothing_in_noise():
    noise = pd.Series(np.random.default_rng(3).normal(0, 0.01, 1500), index=pd.bdate_range("1990-01-01", periods=1500))
    f = study.walk_forward(noise, CFG)
    y = f["target"].to_numpy()
    for m in ("AR(1)", "Ridge", "Boosting"):
        assert study.forecast_stats(f[m].to_numpy(), y, CFG)["r2_oos"] < 0.01


def test_run_is_deterministic(r, fc):
    pd.testing.assert_frame_equal(fc, study.walk_forward(r, CFG))


# ---------- scoring ----------

def test_r2_of_zero_forecast_is_zero_and_of_truth_is_one():
    y = np.random.default_rng(0).normal(0, 0.01, 300)
    assert study.forecast_stats(np.zeros(300), y, CFG)["r2_oos"] == 0.0
    assert study.forecast_stats(y, y, CFG)["r2_oos"] == pytest.approx(1.0)


def test_clark_west_sign():
    y = np.random.default_rng(0).normal(0, 0.01, 2000)
    assert study.forecast_stats(0.5 * y, y, CFG)["clark_west_t"] > 5
    assert study.forecast_stats(-0.5 * y, y, CFG)["clark_west_t"] < -5


def test_loss_diff_sign():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 0.01, 2000)
    assert study.loss_diff_t(0.9 * y, np.zeros(2000), y, CFG) < -5
    assert study.loss_diff_t(np.zeros(2000), 0.9 * y, y, CFG) > 5


def test_newey_west_matches_plain_t_without_lags():
    d = np.random.default_rng(4).normal(0.001, 0.01, 500)
    assert study.newey_west_t(d, 0) == pytest.approx(d.mean() / (d.std() / np.sqrt(500)))


def test_rolling_r2_matches_direct(fc):
    r2 = study.rolling_r2(fc["AR(1)"], fc["target"], window=200)
    y, p = fc["target"].to_numpy()[-200:], fc["AR(1)"].to_numpy()[-200:]
    assert r2.iloc[-1] == pytest.approx(1 - ((y - p) ** 2).sum() / (y ** 2).sum())


# ---------- strategy ----------

def test_strategy_is_long_or_cash_and_pays_for_trades():
    pred = np.array([1.0, 1.0, -1.0, -1.0, 1.0])
    y = np.array([0.01, 0.02, 0.03, -0.04, 0.05])
    net, turn = study.strategy_returns(pred, y, 0.001)
    assert turn.tolist() == [1, 0, 1, 0, 1]
    assert net == pytest.approx([0.01 - 0.001, 0.02, -0.001, 0.0, 0.05 - 0.001])


def test_always_long_equals_buy_and_hold_after_one_entry_cost():
    y = np.random.default_rng(2).normal(0.0005, 0.01, 500)
    net, turn = study.strategy_returns(np.ones(500), y, 0.0005)
    assert turn.sum() == 1 and net[1:] == pytest.approx(y[1:])


def test_break_even_cost():
    rng = np.random.default_rng(5)
    y = rng.normal(0.0003, 0.01, 4000)
    perfect = np.sign(y)
    c = study.break_even_cost(perfect, y)
    assert c > 0
    assert study.sharpe(study.strategy_returns(perfect, y, c * 0.9)[0]) > study.sharpe(y)
    assert study.sharpe(study.strategy_returns(perfect, y, c * 1.1)[0]) < study.sharpe(y)
    assert study.break_even_cost(-perfect, y) == 0.0


def test_bootstrap_interval_covers_zero_for_identical_series_and_is_seeded():
    y = np.random.default_rng(6).normal(0.0003, 0.01, 1000)
    same = study.sharpe_diff_interval(y, y, CFG)
    assert same["diff"] == 0 and same["lo"] == 0 and same["hi"] == 0
    z = np.random.default_rng(8).normal(0.0003, 0.01, 1000)
    assert study.sharpe_diff_interval(y, z, CFG) == study.sharpe_diff_interval(y, z, CFG)
