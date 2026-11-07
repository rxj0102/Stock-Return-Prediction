# Stock Return Prediction

[![CI](https://github.com/rxj0102/Stock-Return-Prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/rxj0102/Stock-Return-Prediction/actions/workflows/ci.yml)

ML pipeline for predicting next-day equity returns — benchmarked on AAPL (2018–2023) with direction accuracy as the primary evaluation criterion.

---

## Problem & Methodology

**Research question:** Can ML models predict the *direction* of next-day log returns better than a random walk?

**Why log returns?** Raw price gives R²≈0.98 via autocorrelation — that is not predictive signal. Log returns are stationary, additive, and approximately normal. Direction accuracy > 52% is the exploitability threshold.

**Pipeline:**

```
Raw OHLCV → Feature Engineering (80+ features) → Permutation Feature Selection
    → Linear / Tree / Neural models → Hyperparameter Tuning → Ensemble Methods
    → Walk-Forward CV → Trading Strategy Backtest
```

**No-leakage design:**
- All features use only information available at close on day *t*
- `StandardScaler` fitted on train split only
- `TimeSeriesSplit` with a 5-day gap throughout (no shuffling)
- Permutation importance evaluated on the held-out test set

**Feature groups:**

| Group | Description |
|-------|-------------|
| Returns | Log-return lags (1,2,3,5,10d), rolling mean/vol (5,10,20,50d) |
| Momentum | 5/10/20-day price momentum, return acceleration |
| Technicals | RSI, MACD, Bollinger Bands, ATR, ADX, OBV, MFI, CCI, Aroon, Stochastic, Williams %R |
| Volume | Relative volume, 20d z-score, volume-price correlation |
| Structure | 20d support/resistance distance, MA-crossover regime, candlestick proxies |
| Calendar | Day-of-week + month with sin/cos cyclical encoding |

**Models evaluated (17 total):**

- **Linear:** Linear Regression, Ridge, Lasso, ElasticNet, Huber
- **Tree-based:** Random Forest, Gradient Boosting, XGBoost, LightGBM (+ tuned variants)
- **Neural:** MLP (3 hidden layers, early stopping)
- **Ensembles:** Equal-weight Voting, Time-Series Stacking (OOF Ridge meta-learner), Inverse-RMSE Weighted Average

---

## Key Results

| Model | Direction Accuracy | Notes |
|-------|:-----------------:|-------|
| Weighted Ensemble (OOF) | ~53–54% | Best overall |
| LightGBM (Tuned) | ~52–53% | Best single model |
| Ridge | ~51–52% | Best linear model |
| Naive (predict 0) | ~50% | Random-walk baseline |

> Results vary slightly by run due to random search stochasticity.
> Direction accuracy above 52% is the practical bar for a tradeable signal at 0.1% round-trip cost.

**Backtest (2022–2023, long-only, 0.1% round-trip cost):**
- Strategy Sharpe > Buy & Hold Sharpe on most runs
- Max drawdown substantially lower than buy-and-hold (selective market exposure)

---

## Tech Stack

| Category | Library |
|----------|---------|
| Data acquisition | `yfinance` |
| Technical indicators | `ta` |
| ML models | `scikit-learn`, `xgboost`, `lightgbm` |
| Data wrangling | `pandas`, `numpy`, `scipy` |
| Visualisation | `matplotlib`, `seaborn` |
| Testing | `pytest` |

Python 3.10+

---

## Repository Structure

```
Stock-Return-Prediction/
├── src/
│   ├── config.py       # All constants and hyperparameters
│   ├── data.py         # Data loading, CSV caching, train/test split
│   ├── features.py     # 80+ feature engineering functions
│   ├── models.py       # Model definitions, tuning, ensemble construction
│   ├── evaluate.py     # Metrics, feature selection, walk-forward CV, plots
│   └── strategy.py     # Backtesting, Sharpe/Sortino/drawdown metrics
├── notebooks/
│   └── 01_stock_return_prediction.ipynb  # Thin orchestration layer
├── data/               # CSV cache (auto-created on first run, gitignored)
├── tests/
│   ├── test_features.py
│   ├── test_evaluate.py
│   └── test_strategy.py
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run tests

```bash
pytest tests/ -v
```

### 3. Run the notebook

```bash
jupyter notebook notebooks/01_stock_return_prediction.ipynb
```

Data is fetched from Yahoo Finance on the first run and cached to `data/`. Subsequent runs load from cache instantly.

### 4. Use modules directly

```python
from src.data     import load_price_data
from src.features import build_features, prepare_matrices
from src.evaluate import compute_metrics

df_raw = load_price_data("AAPL", "2018-01-01", "2024-01-01")
df     = build_features(df_raw)
X, y   = prepare_matrices(df)
```

---

## Research-Level Improvement Directions

### 1. Regime-Conditioned Modelling
Train separate models for bull/bear/sideways regimes (identified via hidden Markov models or volatility clustering). Current models learn a single average relationship; regime-specific models can adapt to the non-stationarity that causes live performance degradation.

### 2. Alternative Data & Cross-Asset Features
Incorporate lagged returns from related instruments (SPY, QQQ, sector ETFs, VIX term structure, bond yields). Single-stock models ignore systematic factor movements that often dominate next-day returns. Cross-asset signals are orthogonal to purely price-derived features and consistently improve direction accuracy in academic literature.

### 3. Probabilistic Forecasting & Kelly Sizing
Replace point predictions with calibrated probability distributions (e.g., conformal prediction, quantile regression, or Bayesian ensembles). Use the predicted probability and variance to implement fractional Kelly position sizing rather than a binary on/off signal — this directly optimises risk-adjusted return instead of direction accuracy and is how production quant strategies are typically structured.
