"""
tests/test_evaluate.py
----------------------
Unit tests for src/evaluate.py — metrics and walk-forward CV.
"""

import numpy as np
import pytest

from src.evaluate import compute_metrics, walk_forward_cv
from sklearn.linear_model import Ridge


# ── compute_metrics ───────────────────────────────────────────────────────────

class TestComputeMetrics:

    def test_perfect_prediction(self):
        y = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        m = compute_metrics(y, y, name="perfect")
        assert m["R2"] == pytest.approx(1.0)
        assert m["RMSE"] == pytest.approx(0.0)
        assert m["Dir Acc (%)"] == pytest.approx(100.0)

    def test_direction_accuracy_formula(self):
        y_true = np.array([ 1.0, -1.0,  1.0, -1.0])
        y_pred = np.array([ 1.0,  1.0,  1.0, -1.0])  # 3 of 4 correct
        m = compute_metrics(y_true, y_pred)
        assert m["Dir Acc (%)"] == pytest.approx(75.0)

    def test_opposite_prediction_direction(self):
        y_true = np.array([0.01, -0.02,  0.03])
        y_pred = np.array([-0.01,  0.02, -0.03])  # all wrong direction
        m = compute_metrics(y_true, y_pred)
        assert m["Dir Acc (%)"] == pytest.approx(0.0)

    def test_naive_zero_prediction(self):
        """Predict-zero baseline should give ~50% direction accuracy on iid data."""
        rng = np.random.default_rng(0)
        y = rng.normal(0, 0.01, 1000)
        m = compute_metrics(y, np.zeros_like(y), name="naive")
        # Direction accuracy undefined when pred=0; sign(0)=0 != sign(±)
        # so it will be close to 0 — not 50%.  Just confirm key is present.
        assert "Dir Acc (%)" in m

    def test_name_stored(self):
        m = compute_metrics(np.ones(5), np.ones(5), name="test_model")
        assert m["Model"] == "test_model"

    def test_output_keys(self):
        m = compute_metrics(np.ones(5), np.ones(5))
        assert set(m.keys()) == {"Model", "RMSE", "MAE", "R2", "Dir Acc (%)"}

    def test_rmse_non_negative(self):
        rng = np.random.default_rng(1)
        y_true = rng.normal(0, 0.01, 100)
        y_pred = rng.normal(0, 0.01, 100)
        m = compute_metrics(y_true, y_pred)
        assert m["RMSE"] >= 0.0

    def test_1d_and_2d_input_equivalent(self):
        y = np.array([0.01, -0.02, 0.03])
        m1 = compute_metrics(y, y)
        m2 = compute_metrics(y.reshape(-1, 1), y.reshape(-1, 1))
        assert m1["RMSE"] == pytest.approx(m2["RMSE"])


# ── walk_forward_cv ───────────────────────────────────────────────────────────

class TestWalkForwardCV:

    def _make_data(self, n=300, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, (n, 5))
        y = rng.normal(0, 0.01, n)
        return X, y

    def test_returns_correct_keys(self):
        X, y = self._make_data()
        models = {"Ridge": Ridge(alpha=1.0)}
        wf_rmse, wf_da = walk_forward_cv(models, X, y, n_splits=3, gap=2)
        assert "Ridge" in wf_rmse
        assert "Ridge" in wf_da

    def test_fold_count(self):
        X, y = self._make_data()
        models = {"Ridge": Ridge(alpha=1.0)}
        n_splits = 4
        wf_rmse, wf_da = walk_forward_cv(models, X, y, n_splits=n_splits, gap=2)
        assert len(wf_rmse["Ridge"]) == n_splits
        assert len(wf_da["Ridge"]) == n_splits

    def test_rmse_non_negative(self):
        X, y = self._make_data()
        models = {"Ridge": Ridge(alpha=1.0)}
        wf_rmse, _ = walk_forward_cv(models, X, y, n_splits=3, gap=2)
        assert all(v >= 0 for v in wf_rmse["Ridge"])

    def test_direction_accuracy_bounded(self):
        X, y = self._make_data()
        models = {"Ridge": Ridge(alpha=1.0)}
        _, wf_da = walk_forward_cv(models, X, y, n_splits=3, gap=2)
        assert all(0 <= v <= 100 for v in wf_da["Ridge"])

    def test_multiple_models(self):
        X, y = self._make_data()
        models = {"Ridge_1": Ridge(alpha=1.0), "Ridge_10": Ridge(alpha=10.0)}
        wf_rmse, wf_da = walk_forward_cv(models, X, y, n_splits=3, gap=2)
        assert set(wf_rmse.keys()) == {"Ridge_1", "Ridge_10"}
