"""
config.py
---------
Central configuration for all pipeline parameters.
Change values here; nothing else in the codebase should define them.
"""

# ── Data ──────────────────────────────────────────────────────────────────────
TICKER     = "AAPL"
START_DATE = "2018-01-01"
END_DATE   = "2024-01-01"
SPLIT_DATE = "2022-01-01"   # train < SPLIT_DATE ≤ test
CACHE_DIR  = "data"         # CSV cache directory (relative to repo root)

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Cross-validation ─────────────────────────────────────────────────────────
N_CV_SPLITS = 5
CV_GAP      = 5   # trading-day gap between train and validation folds

# ── Feature engineering ───────────────────────────────────────────────────────
RETURN_LAGS     = [1, 2, 3, 5, 10]
ROLLING_WINDOWS = [5, 10, 20, 50]
MOM_WINDOWS     = [5, 10, 20]

# ── Feature selection ─────────────────────────────────────────────────────────
PERM_IMPORTANCE_REPEATS = 10
MIN_SELECTED_FEATURES   = 10   # fallback: keep top-N if fewer pass the filter

# ── Model hyperparameters (defaults — tuning overrides these) ─────────────────
RF_PARAMS = dict(
    n_estimators=200, max_depth=8, min_samples_leaf=5,
    random_state=RANDOM_STATE, n_jobs=-1,
)
GBM_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    min_samples_leaf=5, subsample=0.8,
    validation_fraction=0.1, n_iter_no_change=20,
    random_state=RANDOM_STATE,
)
XGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, max_depth=5,
    min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
)
LGBM_PARAMS = dict(
    n_estimators=500, learning_rate=0.03, num_leaves=31,
    max_depth=6, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1,
    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
)
MLP_PARAMS = dict(
    hidden_layer_sizes=(128, 64, 32), activation="relu",
    solver="adam", alpha=1e-3, batch_size=64,
    learning_rate="adaptive", learning_rate_init=1e-3,
    max_iter=500, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=15, random_state=RANDOM_STATE, verbose=False,
)

# ── Hyperparameter search ─────────────────────────────────────────────────────
TUNING_N_ITER = 30

# ── Trading strategy ─────────────────────────────────────────────────────────
SIGNAL_THRESHOLD  = 0.001   # min predicted log-return to enter a position
TRANSACTION_COST  = 0.001   # round-trip cost applied on every position change
ANNUALISATION     = 252     # trading days per year
