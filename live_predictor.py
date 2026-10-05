"""Live Crypto Inference and Paper Trading Signal Generator.

Fetches the most recent live exchange candle, computes real-time technical and
derivatives features, loads the trained model, and generates immediate directional signals
with dynamic Stop-Loss / Take-Profit targets.
"""

import argparse
from datetime import datetime, timezone
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_loader import fetch_ohlcv_ccxt, fetch_derivatives_data, fetch_multitimeframe_ohlcv
from src.features import generate_crypto_features, add_multitimeframe_features


def predict_next_candle(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    model_path: str = None,
    use_derivatives: bool = True,
    confidence_threshold: float = 0.35,
    signal_mode: str = "edge",
    edge_threshold: float = 0.06,
    paper_trade: bool = True,
) -> dict:
    """Execute live inference on the most recent completed candle."""
    clean_sym = symbol.replace("/", "_").replace(":", "_")
    if model_path is None:
        model_path = f"models/{clean_sym}_{timeframe}_model.joblib"

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Trained model checkpoint '{model_path}' not found. "
            f"Please run 'python3 run_pipeline.py --symbol {symbol} --timeframe {timeframe}' first."
        )

    print(f"[*] Loading model from {model_path}...")
    checkpoint = joblib.load(model_path)
    model = checkpoint["model"]
    feature_cols = checkpoint["feature_cols"]

    # 1. Fetch recent candles (1h, 4h, 1d)
    print(f"[*] Fetching latest live market data for {symbol} ({timeframe})...")
    raw_df, df_4h, df_1d = fetch_multitimeframe_ohlcv(
        symbol=symbol,
        primary_timeframe=timeframe,
        primary_bars=120,
        force_download=True,
    )

    if use_derivatives:
        start_ms = int(raw_df.index[0].timestamp() * 1000)
        deriv_df = fetch_derivatives_data(symbol=symbol, timeframe=timeframe, since_ms=start_ms, force_download=True)
        if not deriv_df.empty:
            raw_df = raw_df.join(deriv_df, how="left")
            raw_df["funding_rate"] = raw_df["funding_rate"].ffill().bfill().fillna(0.0001)
            raw_df["oi_val"] = raw_df["oi_val"].ffill().bfill().fillna(8e9)

    # 2. Engineer features
    feat_df = generate_crypto_features(raw_df)
    feat_df = add_multitimeframe_features(feat_df, df_4h=df_4h, df_1d=df_1d)

    # Ensure all expected feature columns exist
    for col in feature_cols:
        if col not in feat_df.columns:
            feat_df[col] = 0.0

    # Extract latest completed candle (the last row)
    latest_bar = feat_df.iloc[-1]
    candle_time = feat_df.index[-1]
    curr_close = float(latest_bar["close"])
    curr_atr = float(latest_bar.get("norm_atr", 0.01))

    # Prepare feature vector
    X_live = pd.DataFrame([latest_bar[feature_cols].values], columns=feature_cols)

    # 3. Model Inference
    probs = model.predict_proba(X_live)[0]
    classes = model.classes_  # e.g. [0, 1, 2]
    prob_dict = {c: float(p) for c, p in zip(classes, probs)}

    p_down = prob_dict.get(0, 0.0)
    p_neutral = prob_dict.get(1, 0.0)
    p_up = prob_dict.get(2, 0.0)

    # 4. Market Regime & Conformal Gate Evaluation
    from src.regime_hmm import MarketRegimeFilter
    from src.conformal import ConformalClassifier

    rf = MarketRegimeFilter()
    regime_info = rf.evaluate_live_bar(feat_df)

    cc = ConformalClassifier(alpha=0.20)
    conformal_info = cc.evaluate_live_bar(prob_dict)

    # 5. Directional Edge & Signal Logic
    edge = p_up - p_down

    if signal_mode == "edge":
        if edge >= edge_threshold:
            signal = "LONG / BUY"
            action_color = "\033[92m"  # Green
            sl_price = curr_close * (1.0 - (1.8 * curr_atr))
            tp_price = curr_close * (1.0 + (1.2 * curr_atr))
        elif edge <= -edge_threshold:
            signal = "SHORT / SELL"
            action_color = "\033[91m"  # Red
            sl_price = curr_close * (1.0 + (1.8 * curr_atr))
            tp_price = curr_close * (1.0 - (1.2 * curr_atr))
        else:
            signal = "NEUTRAL / HOLD (WAIT)"
            action_color = "\033[93m"  # Yellow
            sl_price = None
            tp_price = None
    else:
        if p_up >= confidence_threshold and p_up > p_down:
            signal = "LONG / BUY"
            action_color = "\033[92m"  # Green
            sl_price = curr_close * (1.0 - (1.8 * curr_atr))
            tp_price = curr_close * (1.0 + (1.2 * curr_atr))
        elif p_down >= confidence_threshold and p_down > p_up:
            signal = "SHORT / SELL"
            action_color = "\033[91m"  # Red
            sl_price = curr_close * (1.0 + (1.8 * curr_atr))
            tp_price = curr_close * (1.0 - (1.2 * curr_atr))
        else:
            signal = "NEUTRAL / HOLD (WAIT)"
            action_color = "\033[93m"  # Yellow
            sl_price = None
            tp_price = None

    # Apply precision vetoes if in chop or ambiguous prediction set
    precision_veto = None
    if regime_info.get("is_chop_veto"):
        precision_veto = "CHOP VETO: High-volatility mean-reverting chop detected"
    elif conformal_info.get("abstain") and signal not in ["NEUTRAL / HOLD (WAIT)"]:
        precision_veto = f"CONFORMAL ABSTENTION: Ambiguous prediction set {conformal_info['prediction_set']}"
    elif float(latest_bar.get("adx_14", 25.0)) < 20.0 and signal not in ["NEUTRAL / HOLD (WAIT)"]:
        precision_veto = f"ADX CHOP VETO: Market lacks trend (ADX {float(latest_bar.get('adx_14', 0.0)):.1f} < 20)"

    if precision_veto and signal != "NEUTRAL / HOLD (WAIT)":
        signal = f"HOLD / VETO ({precision_veto})"
        action_color = "\033[93m"  # Yellow
        sl_price = None
        tp_price = None

    reset_color = "\033[0m"

    print("\n" + "=" * 65)
    print(f"       LIVE CRYPTO SIGNAL: {symbol} ({timeframe})")
    print("=" * 65)
    print(f" Bar Timestamp (UTC)  : {candle_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f" Current Price        : ${curr_close:,.2f}")
    print(f" Dynamic 1h Volatility: {curr_atr * 100:.2f}% (ATR)")
    print(f" Market Regime        : {regime_info['dominant_regime'].upper()} (Chop Risk: {regime_info['prob_high_vol_chop']*100:.1f}%)")
    print(f" Conformal Certainty  : Set={conformal_info['prediction_set']} (Singleton: {'YES' if conformal_info['is_singleton'] else 'NO - ABSTAIN'})")
    if "adx_14" in latest_bar:
        adx_val = float(latest_bar['adx_14'])
        trend_lbl = "STRONG TREND" if adx_val >= 25.0 else ("MODERATE" if adx_val >= 20.0 else "CHOP / RANGE")
        print(f" ADX Trend Strength   : {adx_val:.1f} ({trend_lbl}) | +DI: {float(latest_bar.get('adx_plus_di', 0)):.1f}, -DI: {float(latest_bar.get('adx_minus_di', 0)):.1f}")
    if "micro_cvd_zscore" in latest_bar:
        print(f" CVD Flow & Divergence: Z-Score={float(latest_bar['micro_cvd_zscore']):+.2f} | Div={float(latest_bar.get('micro_cvd_divergence', 0)):+.2f}")
    if "micro_liq_flush_bullish" in latest_bar:
        is_bull = bool(float(latest_bar.get("micro_liq_flush_bullish", 0)) > 0)
        is_bear = bool(float(latest_bar.get("micro_liq_flush_bearish", 0)) > 0)
        flush_str = "BULLISH FLUSH (Snapback Reversal)" if is_bull else ("BEARISH FLUSH (Snapback Reversal)" if is_bear else "None (Calm)")
        print(f" Liquidation Cascade  : {flush_str}")
    if "session_london_ny_overlap" in latest_bar:
        is_overlap = bool(float(latest_bar["session_london_ny_overlap"]) > 0)
        is_core = bool(float(latest_bar.get("session_core_hours", 0)) > 0)
        is_wknd = bool(float(latest_bar.get("is_weekend", 0)) > 0)
        session_str = "LONDON/NY OVERLAP (Max Institutional Liquidity)" if is_overlap else ("CORE EU/US TRADING" if is_core else ("WEEKEND LULL (Low Depth)" if is_wknd else "ASIAN SESSION / OFF-HOURS"))
        print(f" Institutional Session: {session_str}")
    if "funding_rate" in latest_bar:
        print(f" Funding Rate (8h)    : {latest_bar['funding_rate'] * 100:+.4f}% (Z-Score: {latest_bar.get('fr_zscore', 0):+.2f})")
    if "oi_val" in latest_bar:
        print(f" Open Interest        : ${latest_bar['oi_val']:,.0f}")
    print("-" * 65)
    print(" Model Probability Breakdown:")
    print(f"   Bearish (Down) : {p_down * 100:.1f}%")
    print(f"   Neutral (Flat) : {p_neutral * 100:.1f}%")
    print(f"   Bullish (Up)   : {p_up * 100:.1f}%")
    print(f"   Directional Edge (Up - Down): {edge * 100:+.2f}% (Mode: {signal_mode}, Threshold: ±{edge_threshold * 100:.1f}%)")
    print("-" * 65)
    print(f" RECOMMENDED ACTION   : {action_color}{signal}{reset_color}")
    if sl_price and tp_price:
        print(f" Suggested Stop-Loss  : ${sl_price:,.2f} ({((sl_price/curr_close)-1)*100:+.2f}%) [Asymmetric 1.8x ATR]")
        print(f" Suggested Take-Profit: ${tp_price:,.2f} ({((tp_price/curr_close)-1)*100:+.2f}%) [Asymmetric 1.2x ATR]")

    # Automated SQLite Paper Trading execution
    paper_res = {}
    if paper_trade:
        from src.paper_tracker import execute_paper_candle, get_paper_trading_summary
        paper_res = execute_paper_candle(
            symbol=symbol,
            candle_time=candle_time,
            open_price=float(latest_bar.get("open", curr_close)),
            high_price=float(latest_bar.get("high", curr_close)),
            low_price=float(latest_bar.get("low", curr_close)),
            close_price=curr_close,
            signal=signal,
            sl_price=sl_price,
            tp_price=tp_price,
            prob_dict=prob_dict,
            edge=edge,
        )
        if paper_res.get("actions"):
            print("-" * 65)
            print(" PAPER TRADING EXECUTIONS (SQLite):")
            for act in paper_res["actions"]:
                print(f"   [+] {act}")
            summary = get_paper_trading_summary()
            print(f"   Virtual Account: Realized PnL: ${summary['realized_pnl_usd']:+,.2f} | Win Rate: {summary['win_rate_pct']:.1f}% ({summary['winning_trades']}/{summary['closed_trades']} trades)")

    print("=" * 65 + "\n")

    return {
        "timestamp": candle_time,
        "price": curr_close,
        "signal": signal,
        "p_down": p_down,
        "p_neutral": p_neutral,
        "p_up": p_up,
        "edge": edge,
        "sl": sl_price,
        "tp": tp_price,
        "paper_actions": paper_res.get("actions", []),
    }


def main():
    parser = argparse.ArgumentParser(description="Live Crypto Predictor & Paper Trading Bot")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading pair (e.g. BTC/USDT)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe (e.g. 1h, 15m)")
    parser.add_argument("--signal-mode", type=str, default="edge", choices=["edge", "absolute"], help="Entry logic: edge or absolute")
    parser.add_argument("--edge-threshold", type=float, default=0.06, help="Directional edge threshold P(Up) - P(Down)")
    parser.add_argument("--confidence", type=float, default=0.35, help="Confidence threshold to trigger entry in absolute mode")
    parser.add_argument("--no-paper-trade", dest="paper_trade", action="store_false", help="Disable virtual paper trade logging")
    parser.add_argument("--watch", action="store_true", help="Run indefinitely, re-checking at each new bar")
    args = parser.parse_args()

    if not args.watch:
        predict_next_candle(
            symbol=args.symbol,
            timeframe=args.timeframe,
            confidence_threshold=args.confidence,
            signal_mode=args.signal_mode,
            edge_threshold=args.edge_threshold,
            paper_trade=args.paper_trade,
        )
        return

    print(f"[*] Starting Paper Trading Watcher for {args.symbol} ({args.timeframe})... Press Ctrl+C to stop.")
    while True:
        try:
            predict_next_candle(
                symbol=args.symbol,
                timeframe=args.timeframe,
                confidence_threshold=args.confidence,
                signal_mode=args.signal_mode,
                edge_threshold=args.edge_threshold,
                paper_trade=args.paper_trade,
            )
            # Sleep 60 seconds before next check
            time.sleep(60)
        except KeyboardInterrupt:
            print("\n[*] Stopping paper trading bot.")
            sys.exit(0)
        except Exception as e:
            print(f"[!] Error in live watcher: {e}")
            time.sleep(30)


if __name__ == "__main__":
    main()
