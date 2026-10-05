"""Vectorized backtesting engine with realistic fees, slippage, and quant metrics.

Simulates Long/Short or Long-only trading strategies on out-of-sample predictions.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


def run_backtest(
    df: pd.DataFrame,
    predictions: pd.Series,
    probabilities: Optional[pd.DataFrame] = None,
    custom_positions: Optional[pd.Series] = None,
    initial_capital: float = 10000.0,
    taker_fee: float = 0.0005,       # 0.05%
    slippage: float = 0.0002,        # 0.02%
    confidence_threshold: float = 0.45,
    signal_mode: str = "edge",       # "edge", "precision", or "absolute"
    edge_threshold: float = 0.06,    # e.g. 6% difference between P(Up) and P(Down)
    allow_short: bool = True,
    conformal_singletons: Optional[pd.Series] = None,
    regime_probs: Optional[pd.DataFrame] = None,
    use_mtf_gate: bool = False,
    bars_per_year: int = 8760,       # 24 * 365 for hourly
) -> Dict[str, Any]:
    """Execute realistic trading backtest on predicted signals or custom continuous bet sizes."""
    # Align on valid test mask
    mask = ~predictions.isna()
    sub_df = df.loc[mask].copy()

    n = len(sub_df)
    if n < 10:
        return {"error": "Insufficient out-of-sample data points for backtesting."}

    if custom_positions is not None:
        sub_df["signal"] = custom_positions.loc[mask].fillna(0.0)
    elif signal_mode in ["edge", "precision", "conformal"] and probabilities is not None and not probabilities.empty:
        signals = np.zeros(n, dtype=float)
        p_up = probabilities["prob_class_2"].loc[mask].fillna(0.0).values if "prob_class_2" in probabilities.columns else np.zeros(n)
        p_down = probabilities["prob_class_0"].loc[mask].fillna(0.0).values if "prob_class_0" in probabilities.columns else np.zeros(n)
        edge = p_up - p_down

        # Optional Conformal singletons
        singletons = None
        if conformal_singletons is not None:
            singletons = conformal_singletons.reindex(sub_df.index).fillna(-1).values

        # Optional Regime chop veto
        prob_chop = None
        if regime_probs is not None and "prob_high_vol_chop" in regime_probs.columns:
            prob_chop = regime_probs["prob_high_vol_chop"].reindex(sub_df.index).fillna(0.0).values

        # Optional MTF Trend Confluence
        tf4h_trend = sub_df["tf4h_trend_ratio"].values if ("tf4h_trend_ratio" in sub_df.columns and use_mtf_gate) else None

        for i in range(n):
            # 1. Chop Veto: High-volatility mean-reverting chop
            if prob_chop is not None and prob_chop[i] >= 0.40:
                signals[i] = 0.0
                continue

            # 2. Conformal Singleton Filter: Reject ambiguous multi-class predictions
            if singletons is not None and singletons[i] == -1:
                signals[i] = 0.0
                continue

            # 3. Long Condition: Positive Edge + Conformal UP (if available) + 4h Uptrend (if gated)
            is_long = (edge[i] >= edge_threshold)
            if singletons is not None:
                is_long = is_long and (singletons[i] == 2)
            if tf4h_trend is not None:
                is_long = is_long and (tf4h_trend[i] > 0)

            # 4. Short Condition: Negative Edge + Conformal DOWN + 4h Downtrend
            is_short = (edge[i] <= -edge_threshold) and allow_short
            if singletons is not None:
                is_short = is_short and (singletons[i] == 0)
            if tf4h_trend is not None:
                is_short = is_short and (tf4h_trend[i] < 0)

            if is_long:
                signals[i] = 1.0
            elif is_short:
                signals[i] = -1.0
            else:
                signals[i] = 0.0

        sub_df["signal"] = signals
        sub_df["edge"] = edge
    else:
        preds = predictions.loc[mask].astype(int)
        signals = np.zeros(n, dtype=float)

        for i in range(n):
            pred = preds.iloc[i]
            conf = 1.0
            if probabilities is not None and not probabilities.empty:
                prob_col = f"prob_class_{pred}"
                if prob_col in probabilities.columns:
                    conf = probabilities[prob_col].iloc[i]

            if conf >= confidence_threshold:
                if pred == 2:  # Up -> Long
                    signals[i] = 1.0
                elif pred == 0 and allow_short:  # Down -> Short
                    signals[i] = -1.0
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0

        sub_df["signal"] = signals

    # Position at bar t is held into bar t+1 (Shift by 1 to prevent lookahead)
    sub_df["position"] = sub_df["signal"].shift(1).fillna(0.0)

    # Asset simple return: Close_t / Close_{t-1} - 1
    asset_returns = sub_df["close"].pct_change().fillna(0.0)
    sub_df["asset_return"] = asset_returns

    # Trading costs incurred whenever position changes
    pos_change = (sub_df["position"] - sub_df["position"].shift(1).fillna(0.0)).abs()
    cost_per_trade = taker_fee + slippage
    trading_costs = pos_change * cost_per_trade
    sub_df["trading_costs"] = trading_costs

    # Strategy net returns: position * asset_return - costs
    strategy_returns = (sub_df["position"] * asset_returns) - trading_costs
    sub_df["strategy_return"] = strategy_returns

    # Equity curves
    sub_df["equity"] = initial_capital * (1.0 + strategy_returns).cumprod()
    sub_df["benchmark_equity"] = initial_capital * (1.0 + asset_returns).cumprod()

    # Performance Metrics
    total_strategy_return = (sub_df["equity"].iloc[-1] / initial_capital) - 1.0
    total_benchmark_return = (sub_df["benchmark_equity"].iloc[-1] / initial_capital) - 1.0

    years = max(n / bars_per_year, 1e-4)
    cagr = (1.0 + total_strategy_return) ** (1.0 / years) - 1.0
    benchmark_cagr = (1.0 + total_benchmark_return) ** (1.0 / years) - 1.0

    # Drawdown
    running_max = sub_df["equity"].cummax()
    drawdown = (sub_df["equity"] - running_max) / running_max
    max_drawdown = float(drawdown.min())

    bench_running_max = sub_df["benchmark_equity"].cummax()
    bench_drawdown = (sub_df["benchmark_equity"] - bench_running_max) / bench_running_max
    bench_max_drawdown = float(bench_drawdown.min())

    # Sharpe & Sortino (Annualized)
    mean_ret = strategy_returns.mean()
    std_ret = strategy_returns.std()
    sharpe = float((mean_ret / (std_ret + 1e-10)) * np.sqrt(bars_per_year)) if std_ret > 0 else 0.0

    downside_returns = strategy_returns[strategy_returns < 0]
    downside_std = downside_returns.std()
    sortino = float((mean_ret / (downside_std + 1e-10)) * np.sqrt(bars_per_year)) if len(downside_returns) > 0 and downside_std > 0 else 0.0

    # Win rate and trades count
    trades_mask = pos_change > 0
    trade_returns = strategy_returns[sub_df["position"] != 0]
    winning_bars = (trade_returns > 0).sum()
    total_active_bars = len(trade_returns)
    win_rate = float(winning_bars / total_active_bars) if total_active_bars > 0 else 0.0

    gross_profit = strategy_returns[strategy_returns > 0].sum()
    gross_loss = np.abs(strategy_returns[strategy_returns < 0].sum())
    profit_factor = float(gross_profit / (gross_loss + 1e-10)) if gross_loss > 0 else float("inf")

    total_trades = int(trades_mask.sum())

    metrics = {
        "initial_capital": initial_capital,
        "final_capital": float(sub_df["equity"].iloc[-1]),
        "total_return_pct": float(total_strategy_return * 100),
        "benchmark_return_pct": float(total_benchmark_return * 100),
        "cagr_pct": float(cagr * 100),
        "benchmark_cagr_pct": float(benchmark_cagr * 100),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "benchmark_max_drawdown_pct": round(bench_max_drawdown * 100, 2),
        "win_rate_pct": round(win_rate * 100, 2),
        "profit_factor": round(profit_factor, 2),
        "total_trades": total_trades,
        "active_bars": total_active_bars,
        "total_test_bars": n,
    }

    return {
        "metrics": metrics,
        "results_df": sub_df[["close", "signal", "position", "strategy_return", "equity", "benchmark_equity"]],
    }
