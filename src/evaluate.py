"""
evaluate.py
-----------
Metrics computation, model comparison, permutation-based feature selection,
and walk-forward cross-validation.

Key functions
~~~~~~~~~~~~~
``compute_metrics(y_true, y_pred, name)``
    Returns a dict with RMSE, MAE, R², and direction accuracy.

``select_features(X_train, y_train, X_test, y_test)``
    Permutation importance on the test set; returns a list of features whose
    net effect is positive (dropping harmful or irrelevant features).

``walk_forward_cv(models, X, y)``
    Expanding-window CV; returns per-fold RMSE and direction accuracy for
    each model.

``plot_model_comparison(results_df)``
    Side-by-side bar charts of direction accuracy and RMSE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .config import (
    MIN_SELECTED_FEATURES,
    N_CV_SPLITS,
    CV_GAP,
    PERM_IMPORTANCE_REPEATS,
    RANDOM_STATE,
    RF_PARAMS,
)


# ── Core metrics ──────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str = "",
) -> dict[str, object]:
    """Compute regression and direction-accuracy metrics.

    Parameters
    ----------
    y_true, y_pred : array-like
        Ground-truth and predicted log returns.
    name : str
        Label stored in the ``'Model'`` key.

    Returns
    -------
    dict with keys: ``Model``, ``RMSE``, ``MAE``, ``R2``, ``Dir Acc (%)``.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        "Model":       name,
        "RMSE":        np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE":         mean_absolute_error(y_true, y_pred),
        "R2":          r2_score(y_true, y_pred),
        "Dir Acc (%)": np.mean(np.sign(y_true) == np.sign(y_pred)) * 100,
    }


def print_metrics(m: dict) -> None:
    """Pretty-print a metrics dict with a directional flag."""
    flag = "✅" if m["Dir Acc (%)"] > 52 else ("⚠️ " if m["Dir Acc (%)"] > 50 else "❌")
    print(
        f"  {m['Model']:32}  DirAcc: {m['Dir Acc (%)']:.2f}%  {flag}  "
        f"RMSE: {m['RMSE']:.6f}  R²: {m['R2']:.4f}"
    )


# ── Feature selection ─────────────────────────────────────────────────────────

def select_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    plot: bool = True,
) -> tuple[list[str], pd.DataFrame]:
    """Select features via permutation importance on the held-out test set.

    Permutation importance is unbiased toward high-cardinality features,
    unlike impurity (Gini) importance from the tree splits.  Features with
    a negative mean importance are dropped (they hurt predictions on average).

    Parameters
    ----------
    X_train, y_train : array-like
        Training data for the initial Random Forest fit.
    X_test, y_test : array-like
        Test data on which permutation importance is evaluated.
    feature_names : list of str
        Column names aligned with the feature arrays.
    plot : bool
        Display a horizontal bar chart of the top-25 features.

    Returns
    -------
    selected_features : list of str
        Feature names with positive permutation importance (at least
        ``MIN_SELECTED_FEATURES``).
    importance_df : pd.DataFrame
        Full importance table sorted by mean importance (descending).
    """
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_train, y_train)

    perm = permutation_importance(
        rf, X_test, y_test,
        n_repeats=PERM_IMPORTANCE_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_df = (
        pd.DataFrame({
            "Feature":    feature_names,
            "Importance": perm.importances_mean,
            "Std":        perm.importances_std,
        })
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    if plot:
        _plot_feature_importance(importance_df)

    # Keep features whose net contribution is positive
    selected = importance_df[importance_df["Importance"] > 0]["Feature"].tolist()
    if len(selected) < MIN_SELECTED_FEATURES:
        selected = importance_df.head(MIN_SELECTED_FEATURES)["Feature"].tolist()

    n_dropped = len(feature_names) - len(selected)
    print(f"Selected {len(selected)} features  |  dropped {n_dropped} (negative importance)")
    return selected, importance_df


def _plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 25) -> None:
    top = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(
        top["Feature"][::-1], top["Importance"][::-1],
        xerr=top["Std"][::-1], capsize=3, color="steelblue", alpha=0.8,
    )
    ax.axvline(0, color="red", lw=0.8, linestyle="--")
    ax.set_xlabel("Mean decrease in R² (permutation importance on test set)")
    ax.set_title(f"Top {top_n} Features — Permutation Importance")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ── Walk-forward cross-validation ─────────────────────────────────────────────

def walk_forward_cv(
    models: dict[str, object],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = N_CV_SPLITS,
    gap: int = CV_GAP,
) -> tuple[dict[str, list], dict[str, list]]:
    """Expanding-window walk-forward cross-validation.

    Linear models (identified by the key ``'Ridge'``) are fitted on scaled
    features; all others receive raw features.

    Parameters
    ----------
    models : dict
        Mapping of model name → unfitted estimator.
    X : np.ndarray
        Full feature array (train + test concatenated).
    y : np.ndarray
        Full target array aligned with X.
    n_splits : int
        Number of expanding folds.
    gap : int
        Gap in rows between each train and validation window.

    Returns
    -------
    wf_rmse, wf_da : dict[str, list]
        Per-fold RMSE and direction-accuracy for each model.
    """
    tscv   = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    wf_rmse: dict[str, list] = {k: [] for k in models}
    wf_da:   dict[str, list] = {k: [] for k in models}

    for tr_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        sc = StandardScaler()
        X_tr_sc  = sc.fit_transform(X_tr)
        X_val_sc = sc.transform(X_val)

        for name, mdl in models.items():
            m = clone(mdl)
            use_scaled = name == "Ridge"
            Xf, Xv = (X_tr_sc, X_val_sc) if use_scaled else (X_tr, X_val)
            m.fit(Xf, y_tr)
            preds = m.predict(Xv)
            wf_rmse[name].append(np.sqrt(mean_squared_error(y_val, preds)))
            wf_da[name].append(np.mean(np.sign(y_val) == np.sign(preds)) * 100)

    return wf_rmse, wf_da


def print_walk_forward_results(
    wf_rmse: dict[str, list],
    wf_da: dict[str, list],
) -> None:
    """Print mean ± std walk-forward results for each model."""
    print("=" * 60)
    print("WALK-FORWARD RESULTS  (mean ± std across folds)")
    print("=" * 60)
    for k in wf_rmse:
        print(
            f"  {k:12}  "
            f"RMSE: {np.mean(wf_rmse[k]):.6f} ± {np.std(wf_rmse[k]):.6f}  "
            f"Dir Acc: {np.mean(wf_da[k]):.2f}% ± {np.std(wf_da[k]):.2f}%"
        )


# ── Model comparison ──────────────────────────────────────────────────────────

def plot_model_comparison(results_df: pd.DataFrame) -> None:
    """Side-by-side bar charts: direction accuracy and RMSE.

    Bars are colour-coded green (>52%), amber (50–52%), or red (<50%).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    sorted_da = results_df.sort_values("Dir Acc (%)")
    colors = [
        "#2ecc71" if v > 52 else ("#f39c12" if v > 50 else "#e74c3c")
        for v in sorted_da["Dir Acc (%)"]
    ]
    ax1.barh(sorted_da["Model"], sorted_da["Dir Acc (%)"], color=colors, alpha=0.85)
    ax1.axvline(50, color="black", linestyle="--", lw=1.5, label="Random (50%)")
    ax1.axvline(52, color="gray", linestyle=":", lw=1, label="Exploitable (~52%)")
    ax1.set_xlabel("Direction Accuracy (%)")
    ax1.set_title("Direction Accuracy by Model")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    sorted_rmse = results_df.sort_values("RMSE")
    ax2.barh(sorted_rmse["Model"], sorted_rmse["RMSE"], color="steelblue", alpha=0.8)
    ax2.set_xlabel("RMSE (log-return units)")
    ax2.set_title("RMSE by Model")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
