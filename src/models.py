"""
models.py
---------
Model definitions, hyperparameter search spaces, and ensemble construction.

Design choices
~~~~~~~~~~~~~~
- Linear models expect *scaled* input (StandardScaler applied externally).
- Tree-based models are scale-invariant and receive raw features.
- ``build_tuned_models`` runs RandomizedSearchCV with TimeSeriesSplit.
- ``build_ensembles`` returns voting, stacking, and OOF-weighted ensembles.
  StackingRegressor is intentionally avoided because it is incompatible with
  TimeSeriesSplit (early rows never appear in any validation fold); OOF
  predictions are generated manually instead.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import randint, uniform, loguniform
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from .config import (
    GBM_PARAMS,
    LGBM_PARAMS,
    MLP_PARAMS,
    RF_PARAMS,
    RANDOM_STATE,
    TUNING_N_ITER,
    XGB_PARAMS,
)


# ── Model factories ───────────────────────────────────────────────────────────

def get_linear_models() -> dict[str, object]:
    """Return a dict of named linear regressors (require scaled input)."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge (α=10)":      Ridge(alpha=10.0, random_state=RANDOM_STATE),
        "Lasso":             Lasso(alpha=1e-4, max_iter=10_000, random_state=RANDOM_STATE),
        "ElasticNet":        ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=10_000,
                                        random_state=RANDOM_STATE),
        "Huber":             HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=300),
    }


def get_tree_models() -> dict[str, object]:
    """Return a dict of named tree-based regressors (scale-invariant)."""
    return {
        "Random Forest":      RandomForestRegressor(**RF_PARAMS),
        "Gradient Boosting":  GradientBoostingRegressor(**GBM_PARAMS),
        "XGBoost":            XGBRegressor(**XGB_PARAMS),
        "LightGBM":           LGBMRegressor(**LGBM_PARAMS),
    }


def get_mlp() -> MLPRegressor:
    """Return the MLP regressor (requires scaled input)."""
    return MLPRegressor(**MLP_PARAMS)


# ── Hyperparameter search ─────────────────────────────────────────────────────

_SEARCH_SPACES: dict[str, tuple] = {
    "RF (Tuned)": (
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        {
            "n_estimators":    randint(100, 400),
            "max_depth":       [5, 8, 10, 15, None],
            "min_samples_leaf": randint(2, 15),
            "max_features":    ["sqrt", "log2", 0.5, 0.7],
        },
    ),
    "XGBoost (Tuned)": (
        XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
        {
            "n_estimators":    randint(100, 400),
            "max_depth":       randint(3, 9),
            "learning_rate":   loguniform(0.01, 0.2),
            "subsample":       uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "min_child_weight": randint(1, 10),
            "reg_alpha":       loguniform(1e-4, 1),
            "reg_lambda":      loguniform(0.1, 10),
        },
    ),
    "LightGBM (Tuned)": (
        LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
        {
            "n_estimators":     randint(100, 500),
            "learning_rate":    loguniform(0.01, 0.2),
            "num_leaves":       randint(15, 63),
            "min_child_samples": randint(10, 50),
            "subsample":        uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "reg_alpha":        loguniform(1e-4, 1),
            "reg_lambda":       loguniform(1e-4, 1),
        },
    ),
}


def build_tuned_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tscv: TimeSeriesSplit,
    verbose: bool = True,
) -> dict[str, object]:
    """Tune RF, XGBoost, and LightGBM via RandomizedSearchCV.

    Parameters
    ----------
    X_train : np.ndarray
        Raw (unscaled) training features.
    y_train : np.ndarray
        Training targets.
    tscv : TimeSeriesSplit
        Shared cross-validation splitter.
    verbose : bool
        Print CV RMSE and best params if True.

    Returns
    -------
    dict mapping model name → fitted best estimator
    """
    tuned: dict[str, object] = {}
    for name, (estimator, param_dist) in _SEARCH_SPACES.items():
        rs = RandomizedSearchCV(
            estimator, param_dist,
            n_iter=TUNING_N_ITER,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=0,
        )
        rs.fit(X_train, y_train)
        tuned[name] = rs.best_estimator_
        if verbose:
            print(f"  {name}: CV RMSE={-rs.best_score_:.6f}  params={rs.best_params_}")
    return tuned


# ── Ensemble construction ─────────────────────────────────────────────────────

def build_voting_ensemble(
    tuned_models: dict[str, object],
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> VotingRegressor:
    """Equal-weight voting ensemble of Ridge + tuned RF/XGBoost/LightGBM.

    Parameters
    ----------
    tuned_models : dict
        Output of :func:`build_tuned_models`.
    X_train, y_train : np.ndarray
        Training data (unscaled — VotingRegressor fits internally).

    Returns
    -------
    Fitted VotingRegressor.
    """
    rf_p   = tuned_models["RF (Tuned)"].get_params()
    xgb_p  = {**tuned_models["XGBoost (Tuned)"].get_params(), "verbosity": 0}
    lgbm_p = tuned_models["LightGBM (Tuned)"].get_params()

    voting = VotingRegressor([
        ("ridge", Ridge(alpha=10.0, random_state=RANDOM_STATE)),
        ("rf",    RandomForestRegressor(**rf_p)),
        ("xgb",   XGBRegressor(**xgb_p)),
        ("lgbm",  LGBMRegressor(**lgbm_p)),
    ])
    voting.fit(X_train, y_train)
    return voting


def build_stacking_ensemble(
    tuned_models: dict[str, object],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_train_scaled: np.ndarray,
    X_test: np.ndarray,
    X_test_scaled: np.ndarray,
) -> np.ndarray:
    """Manual time-series stacking that is compatible with TimeSeriesSplit.

    StackingRegressor uses cross_val_predict which does not work correctly with
    TimeSeriesSplit because the first fold has no prior validation rows.
    This function generates OOF predictions fold-by-fold, fits a Ridge
    meta-learner on them, and returns test-set predictions.

    Parameters
    ----------
    tuned_models : dict
        Output of :func:`build_tuned_models`.
    X_train, y_train : np.ndarray
        Raw training features and targets.
    X_train_scaled : np.ndarray
        Scaled version of X_train (for Ridge base model).
    X_test : np.ndarray
        Raw test features.
    X_test_scaled : np.ndarray
        Scaled test features (for Ridge base model).

    Returns
    -------
    np.ndarray
        Meta-learner predictions on the test set.
    """
    rf_p  = {**tuned_models["RF (Tuned)"].get_params(), "n_jobs": -1}
    base_configs = {
        "ridge": Ridge(alpha=10.0, random_state=RANDOM_STATE),
        "rf":    RandomForestRegressor(**rf_p),
        "xgb":   XGBRegressor(**{**tuned_models["XGBoost (Tuned)"].get_params(), "verbosity": 0}),
    }

    n_train = len(X_train)
    oof = {k: np.full(n_train, np.nan) for k in base_configs}

    for tr_idx, val_idx in TimeSeriesSplit(n_splits=5).split(X_train):
        sc_fold = StandardScaler()
        Xf_tr_sc  = sc_fold.fit_transform(X_train[tr_idx])
        Xf_val_sc = sc_fold.transform(X_train[val_idx])

        for k, mdl in base_configs.items():
            m = clone(mdl)
            if k == "ridge":
                m.fit(Xf_tr_sc, y_train[tr_idx])
                oof[k][val_idx] = m.predict(Xf_val_sc)
            else:
                m.fit(X_train[tr_idx], y_train[tr_idx])
                oof[k][val_idx] = m.predict(X_train[val_idx])

    # Mask rows without OOF predictions (first fold has no prior val rows)
    valid = ~np.isnan(next(iter(oof.values())))
    meta  = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    meta.fit(np.column_stack([oof[k][valid] for k in base_configs]), y_train[valid])

    # Re-fit base models on full training set
    test_preds = []
    for k, mdl in base_configs.items():
        m = clone(mdl)
        if k == "ridge":
            m.fit(X_train_scaled, y_train)
            test_preds.append(m.predict(X_test_scaled))
        else:
            m.fit(X_train, y_train)
            test_preds.append(m.predict(X_test))

    return meta.predict(np.column_stack(test_preds))


def build_weighted_ensemble(
    tuned_models: dict[str, object],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    tscv: TimeSeriesSplit,
    verbose: bool = True,
) -> np.ndarray:
    """Inverse-RMSE weighted average of Ridge, XGBoost, and LightGBM.

    Weights are derived from cross-validation RMSE so no test-set information
    leaks into the ensemble construction.

    Parameters
    ----------
    tuned_models : dict
        Output of :func:`build_tuned_models`.
    X_train, y_train : np.ndarray
        Training data.
    X_test : np.ndarray
        Test features.
    tscv : TimeSeriesSplit
        Shared CV splitter.
    verbose : bool
        Print computed ensemble weights if True.

    Returns
    -------
    np.ndarray
        Weighted-average predictions on the test set.
    """
    members = {
        "Ridge":    Ridge(alpha=10.0, random_state=RANDOM_STATE),
        "XGBoost":  tuned_models["XGBoost (Tuned)"],
        "LightGBM": tuned_models["LightGBM (Tuned)"],
    }

    # Compute CV RMSE per model (on unscaled features — tree models only)
    cv_rmse = {}
    for name, mdl in members.items():
        scores = cross_val_score(
            mdl, X_train, y_train,
            cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1,
        )
        cv_rmse[name] = -scores.mean()

    raw_w = {k: 1.0 / v for k, v in cv_rmse.items()}
    total = sum(raw_w.values())
    weights = {k: v / total for k, v in raw_w.items()}

    if verbose:
        print("  Ensemble weights:", ", ".join(f"{k}:{v:.3f}" for k, v in weights.items()))

    # Fit on full training set and predict
    preds = {}
    for name, mdl in members.items():
        mdl.fit(X_train, y_train)
        preds[name] = mdl.predict(X_test)

    return sum(weights[k] * preds[k] for k in weights)
