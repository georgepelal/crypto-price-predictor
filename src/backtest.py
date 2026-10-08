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
    use_ev_gate: bool = False,
    ev_hurdle: float = 0.002,
    meta_probabilities: Optional[pd.Series] = None,
    meta_threshold: float = 0.55,
    profit_mult: float = 1.5,
    stop_mult: float = 1.0,
    use_barrier_holding: bool = False,
    max_holding_bars: int = 8,
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

        # Optional Meta-Model Probabilities
        meta_p = meta_probabilities.reindex(sub_df.index).fillna(0.5).values if meta_probabilities is not None else None

        # ATR for dynamic EV calculation
        atr_vals = sub_df["norm_atr"].values if "norm_atr" in sub_df.columns else np.full(n, 0.01)
        round_trip_friction = (taker_fee + slippage) * 2.0

        active_pos = 0.0
        tp_target = 0.0
        sl_target = 0.0
        holding_bars = 0

        for i in range(n):
            c_i = float(sub_df["close"].iloc[i])
            h_i = float(sub_df["high"].iloc[i]) if "high" in sub_df.columns else c_i
            l_i = float(sub_df["low"].iloc[i]) if "low" in sub_df.columns else c_i

            # 0. Check active position barrier resolution
            if active_pos != 0.0 and use_barrier_holding:
                holding_bars += 1
                if active_pos > 0:  # Long
                    if l_i <= sl_target or h_i >= tp_target or holding_bars >= max_holding_bars:
                        active_pos = 0.0
                        holding_bars = 0
                elif active_pos < 0:  # Short
                    if h_i >= sl_target or l_i <= tp_target or holding_bars >= max_holding_bars:
                        active_pos = 0.0
                        holding_bars = 0

            # If flat (or barrier holding disabled), evaluate new signals
            if active_pos == 0.0 or not use_barrier_holding:
                # 1. Chop Veto: High-volatility mean-reverting chop
                if prob_chop is not None and prob_chop[i] >= 0.40:
                    signals[i] = 0.0
                    continue

                # 2. Conformal Singleton Filter: Reject ambiguous multi-class predictions
                if singletons is not None and singletons[i] == -1:
                    signals[i] = 0.0
                    continue

                # 3. Base Directional Edge
                is_long = (edge[i] >= edge_threshold)
                is_short = (edge[i] <= -edge_threshold) and allow_short

                if singletons is not None:
                    is_long = is_long and (singletons[i] == 2)
                    is_short = is_short and (singletons[i] == 0)

                if tf4h_trend is not None:
                    is_long = is_long and (tf4h_trend[i] > 0)
                    is_short = is_short and (tf4h_trend[i] < 0)

                # 4. Expected Value (EV) Filter: Positive Mathematical Expectancy
                if use_ev_gate:
                    curr_atr = max(atr_vals[i], 0.005) if not np.isnan(atr_vals[i]) else 0.01
                    ev_l = (p_up[i] * profit_mult * curr_atr) - (p_down[i] * stop_mult * curr_atr) - round_trip_friction
                    ev_s = (p_down[i] * profit_mult * curr_atr) - (p_up[i] * stop_mult * curr_atr) - round_trip_friction

                    if is_long and ev_l < ev_hurdle:
                        is_long = False
                    if is_short and ev_s < ev_hurdle:
                        is_short = False

                # 5. Meta-Labeling Precision Filter: Veto trade if probability of success < threshold
                if meta_p is not None and (is_long or is_short):
                    if meta_p[i] < meta_threshold:
                        is_long = False
                        is_short = False

                if use_barrier_holding:
                    curr_atr = max(atr_vals[i], 0.005) if not np.isnan(atr_vals[i]) else 0.01
                    if is_long:
                        active_pos = 1.0
                        tp_target = c_i * (1.0 + (profit_mult * curr_atr))
                        sl_target = c_i * (1.0 - (stop_mult * curr_atr))
                        holding_bars = 0
                    elif is_short:
                        active_pos = -1.0
                        tp_target = c_i * (1.0 - (profit_mult * curr_atr))
                        sl_target = c_i * (1.0 + (stop_mult * curr_atr))
                        holding_bars = 0
                    signals[i] = active_pos
                else:
                    if is_long:
                        signals[i] = 1.0
                    elif is_short:
                        signals[i] = -1.0
                    else:
                        signals[i] = 0.0
            else:
                signals[i] = active_pos

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
