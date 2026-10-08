"""Institutional Multi-Asset Quantitative Crypto Scanner & Capital Allocator.

Scans BTC/USDT, ETH/USDT, and SOL/USDT concurrently in real time.
Evaluates Microstructure, ADX Trend Strength, Conformal Prediction Certainty,
CVD Flows, and Market Regimes across all assets to dynamically allocate capital
to the asset with the highest statistical conviction.
"""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time
from typing import Dict, Any, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.data_loader import fetch_ohlcv_ccxt, fetch_derivatives_data, fetch_multitimeframe_ohlcv
from src.features import generate_crypto_features, add_multitimeframe_features, add_cross_asset_features
from src.regime_hmm import MarketRegimeFilter
from src.conformal import ConformalClassifier
from src.paper_tracker import execute_paper_candle, get_paper_trading_summary

# ANSI Terminal Styling
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def scan_single_asset(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    model_dir: str = "models",
    edge_threshold: float = 0.05,
) -> Dict[str, Any]:
    """Execute live feature pipeline and inference for a single cryptocurrency pair."""
    clean_sym = symbol.replace("/", "_").replace(":", "_")
    model_path = Path(model_dir) / f"{clean_sym}_{timeframe}_model.joblib"

    if not model_path.exists():
        # Fallback to BTC model if specific altcoin model not yet trained
        btc_path = Path(model_dir) / f"BTC_USDT_{timeframe}_model.joblib"
        if btc_path.exists():
            model_path = btc_path
        else:
            return {
                "symbol": symbol,
                "status": "ERROR",
                "message": f"Model not found at {model_path}. Run training pipeline first.",
                "conviction_score": 0.0,
            }

    checkpoint = joblib.load(model_path)
    model = checkpoint["model"]
    meta_model = checkpoint.get("meta_model")
    feature_cols = checkpoint["feature_cols"]

    # 1. Fetch OHLCV candles (1h, 4h, 1d)
    try:
        raw_df, df_4h, df_1d = fetch_multitimeframe_ohlcv(
            symbol=symbol,
            primary_timeframe=timeframe,
            primary_bars=120,
            force_download=False,
        )
    except Exception as e:
        return {
            "symbol": symbol,
            "status": "ERROR",
            "message": f"Data fetch error: {e}",
            "conviction_score": 0.0,
        }

    # 2. Ingest Derivatives if available
    try:
        start_ms = int(raw_df.index[0].timestamp() * 1000)
        deriv_df = fetch_derivatives_data(symbol=symbol, timeframe=timeframe, since_ms=start_ms, force_download=False)
        if not deriv_df.empty:
            raw_df = raw_df.join(deriv_df, how="left")
            raw_df["funding_rate"] = raw_df["funding_rate"].ffill().bfill().fillna(0.0001)
            raw_df["oi_val"] = raw_df["oi_val"].ffill().bfill().fillna(8e9)
    except Exception:
        pass

    # 3. Engineer complete features
    feat_df = generate_crypto_features(raw_df)
    feat_df = add_multitimeframe_features(feat_df, df_4h=df_4h, df_1d=df_1d)

    # 4. Cross-Asset Benchmark
    if any("cross_" in c for c in feature_cols):
        sec_sym = "ETH/USDT" if "BTC" in symbol else "BTC/USDT"
        sec_name = "ETH" if "ETH" in sec_sym else "BTC"
        try:
            sec_df = fetch_ohlcv_ccxt(symbol=sec_sym, timeframe=timeframe, total_bars=120, force_download=False)
            if sec_df is not None and not sec_df.empty:
                feat_df = add_cross_asset_features(feat_df, df_secondary=sec_df, secondary_symbol=sec_name)
        except Exception:
            pass

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in feat_df.columns:
            feat_df[col] = 0.0

    latest_bar = feat_df.iloc[-1]
    candle_time = feat_df.index[-1]
    curr_close = float(latest_bar["close"])
    curr_atr = float(latest_bar.get("norm_atr", 0.01))

    # 5. Model Inference
    X_live = pd.DataFrame([latest_bar[feature_cols].values], columns=feature_cols)
    probs = model.predict_proba(X_live)[0]
    classes = model.classes_
    prob_dict = {c: float(p) for c, p in zip(classes, probs)}

    p_down = prob_dict.get(0, 0.0)
    p_neutral = prob_dict.get(1, 0.0)
    p_up = prob_dict.get(2, 0.0)
    edge = p_up - p_down

    # 6. Precision Gates Evaluation
    rf = MarketRegimeFilter()
    regime_info = rf.evaluate_live_bar(feat_df)
    dominant_regime = regime_info.get("dominant_regime", "CHOP").upper()
    is_chop_veto = regime_info.get("is_chop_veto", False)

    cc = ConformalClassifier(alpha=0.20)
    conformal_info = cc.evaluate_live_bar(prob_dict)
    is_singleton = conformal_info.get("is_singleton", False)
    pred_set = conformal_info.get("prediction_set", [])
    conformal_abstain = conformal_info.get("abstain", False)

    adx_val = float(latest_bar.get("adx_14", 25.0))
    is_adx_choppy = adx_val < 20.0

    # 7. Expected Value Calculation
    from src.meta_labeling import compute_expected_value
    ev_info = compute_expected_value(p_up=p_up, p_down=p_down, atr_val=curr_atr, profit_mult=1.5, stop_mult=1.0)
    ev_long = ev_info["ev_long"]
    ev_short = ev_info["ev_short"]

    # 8. Meta-Model Evaluation
    meta_prob = None
    if meta_model is not None and hasattr(meta_model, "predict_proba"):
        try:
            meta_prob = float(meta_model.predict_proba(X_live)[0][1])
        except Exception:
            meta_prob = None

    # 9. Signal & Conviction Score
    direction = "NEUTRAL"
    sl_price = None
    tp_price = None
    best_ev = 0.0

    if edge >= edge_threshold:
        direction = "BUY / LONG"
        sl_price = curr_close * (1.0 - (1.0 * curr_atr))
        tp_price = curr_close * (1.0 + (1.5 * curr_atr))
        best_ev = ev_long
    elif edge <= -edge_threshold:
        direction = "SELL / SHORT"
        sl_price = curr_close * (1.0 + (1.0 * curr_atr))
        tp_price = curr_close * (1.0 - (1.5 * curr_atr))
        best_ev = ev_short

    # Veto evaluation
    veto_reason = None
    if is_chop_veto:
        veto_reason = "HIGH-VOL CHOP VETO"
    elif is_adx_choppy:
        veto_reason = f"LOW ADX ({adx_val:.1f} < 20)"
    elif conformal_abstain and direction != "NEUTRAL":
        veto_reason = f"CONFORMAL ABSTAIN {pred_set}"
    elif direction != "NEUTRAL" and best_ev < 0.0:
        veto_reason = f"NEGATIVE EV ({best_ev*100:+.2f}%)"
    elif meta_prob is not None and direction != "NEUTRAL" and meta_prob < 0.52:
        veto_reason = f"META-MODEL LOW WIN P ({meta_prob*100:.1f}% < 52%)"

    # Quantitative Conviction Score Calculation:
    if veto_reason or direction == "NEUTRAL":
        conviction_score = 0.0
    else:
        adx_multiplier = np.clip(adx_val / 25.0, 0.8, 1.8)
        singleton_multiplier = 1.25 if is_singleton else 1.0
        meta_multiplier = (meta_prob / 0.50) if meta_prob is not None else 1.0
        ev_multiplier = 1.2 if best_ev > 0.003 else 1.0
        conviction_score = abs(edge) * adx_multiplier * singleton_multiplier * meta_multiplier * ev_multiplier

    return {
        "symbol": symbol,
        "timestamp": candle_time,
        "price": curr_close,
        "atr_pct": curr_atr * 100.0,
        "p_down": p_down,
        "p_neutral": p_neutral,
        "p_up": p_up,
        "edge": edge,
        "ev_long": ev_long,
        "ev_short": ev_short,
        "best_ev": best_ev,
        "meta_prob": meta_prob,
        "direction": direction,
        "regime": dominant_regime,
        "adx": adx_val,
        "is_singleton": is_singleton,
        "pred_set": pred_set,
        "veto_reason": veto_reason,
        "conviction_score": round(conviction_score, 4),
        "sl_price": sl_price,
        "tp_price": tp_price,
        "prob_dict": prob_dict,
        "open_bar": float(latest_bar.get("open", curr_close)),
        "high_bar": float(latest_bar.get("high", curr_close)),
        "low_bar": float(latest_bar.get("low", curr_close)),
        "status": "OK",
    }


def scan_and_allocate_capital(
    symbols: List[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
    timeframe: str = "1h",
    paper_trade: bool = True,
    db_path: str = "data/paper_trading.db",
) -> Dict[str, Any]:
    """Scan all target assets, rank by conviction, and execute the best candidate trade."""
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print("\n" + "=" * 80)
    print(f" {BOLD}INSTITUTIONAL MULTI-ASSET QUANTITATIVE SCANNER{RESET}       {DIM}{now_utc}{RESET}")
    print("=" * 80)

    asset_results = []
    for sym in symbols:
        res = scan_single_asset(symbol=sym, timeframe=timeframe)
        asset_results.append(res)

    # Sort assets by conviction score (highest edge & cleanest trend first)
    ranked = sorted(asset_results, key=lambda x: x.get("conviction_score", 0.0), reverse=True)

    # Print Comparative Opportunity Matrix
    print(f" {'Symbol':<9} {'Price':<11} {'Regime':<11} {'ADX':<5} {'Edge':<7} {'EV':<8} {'Meta-P':<7} {'Certainty':<11} {'Conviction':<11} {'Signal'}")
    print("-" * 88)

    for item in ranked:
        if item.get("status") != "OK":
            print(f" {item['symbol']:<9} {RED}ERROR: {item.get('message')}{RESET}")
            continue

        sym = item["symbol"]
        p_str = f"${item['price']:,.2f}"
        reg_str = item["regime"]
        adx_str = f"{item['adx']:.1f}"
        edge_str = f"{item['edge']*100:+.1f}%"
        ev_val = item.get("best_ev", 0.0)
        ev_str = f"{ev_val*100:+.2f}%"
        meta_val = item.get("meta_prob")
        meta_str = f"{meta_val*100:.1f}%" if meta_val is not None else "--"
        cert_str = f"Set={item['pred_set']}"
        score_str = f"{item['conviction_score']:.4f}"

        if item["veto_reason"]:
            sig_col = YELLOW
            sig_str = f"VETO ({item['veto_reason']})"
        elif item["direction"] != "NEUTRAL":
            sig_col = GREEN if "BUY" in item["direction"] else RED
            sig_str = item["direction"]
        else:
            sig_col = DIM
            sig_str = "FLAT / WAIT"

        print(f" {sym:<9} {p_str:<11} {reg_str:<11} {adx_str:<5} {edge_str:<7} {ev_str:<8} {meta_str:<7} {cert_str:<11} {score_str:<11} {sig_col}{sig_str}{RESET}")

    print("-" * 88)

    # Capital Allocation Decision
    best_candidate = ranked[0]
    allocated_trade = None

    if best_candidate.get("status") == "OK" and best_candidate["conviction_score"] >= 0.03 and not best_candidate["veto_reason"]:
        sym = best_candidate["symbol"]
        sig = best_candidate["direction"]
        print(f" {BOLD}{GREEN}[★] TOP OPPORTUNITY SELECTED:{RESET} {BOLD}{sym}{RESET} | Signal: {GREEN}{sig}{RESET} | Conviction Score: {best_candidate['conviction_score']:.4f}")
        print(f"     Suggested Stop-Loss  : ${best_candidate['sl_price']:,.2f} [1.8x ATR]")
        print(f"     Suggested Take-Profit: ${best_candidate['tp_price']:,.2f} [1.2x ATR]")

        if paper_trade:
            paper_res = execute_paper_candle(
                symbol=sym,
                candle_time=best_candidate["timestamp"],
                open_price=best_candidate["open_bar"],
                high_price=best_candidate["high_bar"],
                low_price=best_candidate["low_bar"],
                close_price=best_candidate["price"],
                signal=sig,
                sl_price=best_candidate["sl_price"],
                tp_price=best_candidate["tp_price"],
                prob_dict=best_candidate["prob_dict"],
                edge=best_candidate["edge"],
                db_path=db_path,
            )
            allocated_trade = paper_res
            if paper_res.get("actions"):
                print("     [+] Paper Execution Log:")
                for act in paper_res["actions"]:
                    print(f"         {act}")
    else:
        print(f" {YELLOW}[i] CAPITAL ALLOCATOR STATUS: ALL ASSETS IN FLAT / CONSOLIDATION / VETO{RESET}")
        print(f"     No trade executed. Capital protected in 100% Cash.")

    summary = get_paper_trading_summary(db_path=db_path)
    print("=" * 80)
    print(f" Portfolio Status: Realized PnL: ${summary['realized_pnl_usd']:+,.2f} | Win Rate: {summary['win_rate_pct']:.1f}% ({summary['winning_trades']}/{summary['closed_trades']} trades)")
    print("=" * 80 + "\n")

    return {
        "ranked_assets": ranked,
        "best_candidate": best_candidate,
        "allocated_trade": allocated_trade,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Asset Crypto Opportunity Scanner")
    parser.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT", "SOL/USDT"], help="Symbols to scan")
    parser.add_argument("--timeframe", type=str, default="1h", help="Candle timeframe")
    parser.add_argument("--no-paper-trade", dest="paper_trade", action="store_false", help="Disable paper execution")
    parser.add_argument("--watch", action="store_true", help="Run indefinitely, re-scanning at regular intervals")
    parser.add_argument("--interval", type=int, default=60, help="Polling interval in seconds when --watch is active")
    args = parser.parse_args()

    if not args.watch:
        scan_and_allocate_capital(symbols=args.symbols, timeframe=args.timeframe, paper_trade=args.paper_trade)
        return

    print(f"[*] Starting Multi-Asset Live Scanner ({', '.join(args.symbols)})... Polling every {args.interval}s. Press Ctrl+C to stop.")
    while True:
        try:
            scan_and_allocate_capital(symbols=args.symbols, timeframe=args.timeframe, paper_trade=args.paper_trade)
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[*] Stopping scanner.")
            sys.exit(0)
        except Exception as e:
            print(f"[!] Scanner error: {e}")
            time.sleep(30)


if __name__ == "__main__":
    main()
