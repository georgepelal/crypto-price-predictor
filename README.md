# Crypto ML Price Predictor

An institutional-grade Quantitative Machine Learning pipeline for cryptocurrency price movement forecasting and strategy backtesting, powered by **Technical Analysis**, **Perpetual Derivatives (Funding Rates & Open Interest)**, **Crypto Fear & Greed Sentiment**, and **Meta-Labeling with Dynamic Bet Sizing (Marcos López de Prado)**.

---

## Architecture Highlights

1. **Multi-Factor Data Pipeline**:
   * **OHLCV Market Data**: Automated paginated downloader via CCXT from Binance.
   * **Perpetual Futures Derivatives**: 8-hour Funding Rates (`fr_zscore`, `fr_cum_24h`) and Open Interest (`oi_change_1h`, `oi_price_alignment`, `oi_to_vol_ratio`).
   * **Alternative Data (Sentiment)**: Real-time & historical Crypto Fear & Greed Index from Alternative.me (`fng_zscore_30d`, `fng_contrarian_signal`).

2. **Zero Lookahead Bias**:
   * Every single technical, derivative, and sentiment indicator is computed strictly at time $t$ or earlier.
   * Dynamic labeling uses fee hurdles and volatility barriers ($\text{Fee} + \alpha \times \text{ATR}$).

3. **Walk-Forward Validation**:
   * Replaces flawed standard K-Fold CV with expanding-window **Walk-Forward Cross-Validation** (5 folds) to eliminate lookahead leakage.

4. **Institutional Meta-Labeling (Marcos López de Prado)**:
   * **Primary Model**: Predicts market direction ($+1$ Long, $-1$ Short, $0$ Flat).
   * **Secondary Meta-Model**: Predicts the *probability of trade profitability net of fees*.
   * **Fractional Kelly Bet Sizing**: Trades with $P(\text{Win}) \le 50\%$ are vetoed (size = 0). High-conviction trades are scaled smoothly up to 100% allocation, minimizing drawdowns and filtering false breakouts.

5. **Realistic Backtesting Engine**:
   * Simulates execution on the subsequent bar ($t+1$).
   * Deducts realistic exchange taker fees (5 bps) and slippage (2 bps) on every position transition.
   * Computes Sharpe Ratio, Sortino Ratio, Maximum Drawdown (MDD), Win Rate, and Profit Factor.

---

## Quickstart

### 1. Setup Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Commands

#### A. Standard End-to-End Pipeline
```bash
# Train & Backtest BTC/USDT (1h) with Multi-Timeframe Context & Directional Edge
python3 run_pipeline.py --symbol "BTC/USDT" --timeframe "1h" --bars 700 --use-mtf --signal-mode edge --edge-threshold 0.15

# Train with Optuna Bayesian Optimization (25 trials)
python3 run_pipeline.py --symbol "BTC/USDT" --optimize --trials 25
```

#### B. Advanced Institutional Meta-Labeling Pipeline
```bash
# Runs Primary Directional Model + Meta-Model Gatekeeper + Kelly Bet Sizing
python3 run_meta_pipeline.py --symbol "BTC/USDT" --timeframe "1h" --bars 700
python3 run_meta_pipeline.py --symbol "ETH/USDT" --timeframe "1h" --bars 700
```

#### C. Multi-Asset Screener
```bash
# Comparative benchmark across BTC, ETH, and SOL
python3 screen_assets.py --symbols "BTC/USDT" "ETH/USDT" "SOL/USDT" --timeframes "1h"
```

#### D. Live Real-Time Predictor & Paper Trading Bot
```bash
# Immediate signal on the latest candle with Directional Edge
python3 live_predictor.py --symbol "BTC/USDT" --signal-mode edge --edge-threshold 0.15

# Run continuously as an hourly bot
python3 live_predictor.py --symbol "BTC/USDT" --watch
```

---

## Project Structure

```text
pricePredictor/
├── configs/
│   └── default_config.yaml      # Hyperparameters, fee rates, feature settings
├── data/
│   ├── raw/                     # Cached OHLCV, Derivatives, and Fear & Greed CSVs
│   └── processed/               # Cleaned tabular datasets
├── models/                      # Saved trained models (.joblib)
├── reports/                     # Out-of-sample equity curves and screener summaries (.png, .csv)
├── src/
│   ├── data_loader.py           # CCXT downloader (OHLCV + Derivatives + Fear & Greed + MTF)
│   ├── features.py              # 70 non-lookahead indicators + MTF + FracDiff + Microstructure
│   ├── fracdiff.py              # Fractional Differentiation (FFD) stationarity with memory
│   ├── microstructure.py        # Amihud Illiquidity, Corwin-Schultz, Roll, and Order Flow Imbalance
│   ├── conformal.py             # Conformal Prediction selective classification gate
│   ├── regime_hmm.py            # 3-state GMM Market Regime Filter & chop veto
│   ├── feature_selection.py     # De-correlation & permutation importance pruning
│   ├── paper_tracker.py         # SQLite automated virtual trade & performance logger
│   ├── labeling.py              # Volatility-buffered directional labels
│   ├── model.py                 # Walk-Forward CV trainer with LightGBM & HistGBM fallback
│   ├── meta_labeling.py         # Secondary meta-model & fractional Kelly bet sizing
│   ├── optimizer.py             # Optuna Bayesian hyperparameter optimizer
│   ├── backtest.py              # Vectorized backtester with Precision Gates & Kelly sizing
│   └── utils.py                 # Summary tables & matplotlib equity charts
├── tests/
│   └── test_pipeline.py         # Automated unit test suite (14/14 passing)
├── run_pipeline.py              # Main training & backtesting runner
├── run_meta_pipeline.py         # Meta-labeling comparative runner
├── screen_assets.py             # Multi-asset benchmark runner
├── live_predictor.py            # Real-time inference & SQLite paper trading bot
├── requirements.txt
└── README.md
```
