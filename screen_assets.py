"""Multi-Asset & Multi-Timeframe Screening Benchmark.

Runs automated Walk-Forward training and backtesting across multiple crypto pairs
(BTC, ETH, SOL) and timeframes (1h, 15m), generating a comparative alpha report.
"""

import argparse
import pandas as pd
from typing import List

from src.data_loader import fetch_ohlcv_ccxt, fetch_derivatives_data
from src.features import generate_crypto_features
from src.labeling import make_directional_labels
from src.model import train_walk_forward
from src.backtest import run_backtest


def benchmark_asset(
    symbol: str,
    timeframe: str = "1h",
    bars: int = 700,
    use_derivatives: bool = True,
    confidence: float = 0.35,
) -> dict:
    """Run full pipeline for a single asset and timeframe, returning summary metrics."""
    print(f"\n---> Benchmarking {symbol} ({timeframe}) | Bars: {bars} | Derivatives: {use_derivatives}...")

    # 1. Fetch OHLCV
    raw_df = fetch_ohlcv_ccxt(symbol=symbol, timeframe=timeframe, total_bars=bars)
    if raw_df is None or len(raw_df) < 150:
        return {"symbol": symbol, "timeframe": timeframe, "error": "Insufficient data"}

    # 2. Fetch Derivatives
    if use_derivatives:
        start_ts_ms = int(raw_df.index[0].timestamp() * 1000)
        deriv_df = fetch_derivatives_data(symbol=symbol, timeframe=timeframe, since_ms=start_ts_ms)
        if not deriv_df.empty:
            raw_df = raw_df.join(deriv_df, how="left")
            raw_df["funding_rate"] = raw_df["funding_rate"].ffill().bfill().fillna(0.0001)
            raw_df["oi_val"] = raw_df["oi_val"].ffill().bfill().fillna(8e9)

    # 3. Features & Labels
    feat_df = generate_crypto_features(raw_df)
    labeled_df = make_directional_labels(feat_df, forward_bars=1, atr_multiplier=0.35)

    feature_cols = [
        c for c in labeled_df.columns
        if c not in ["open", "high", "low", "close", "volume", "target", "fwd_return", "threshold"]
    ]
    valid_data = labeled_df[labeled_df["target"] != -1].dropna(subset=feature_cols + ["target"])

    X = valid_data[feature_cols]
    y = valid_data["target"].astype(int)

    effective_min_train = max(100, int(len(X) * 0.4))

    # 4. Train Model
    train_results = train_walk_forward(
        X,
        y,
        n_splits=5,
        min_train_size=effective_min_train,
        model_type="auto",
    )

    # 5. Backtest
    bt = run_backtest(
        df=valid_data,
        predictions=train_results["oof_preds"],
        probabilities=train_results["oof_probs"],
        confidence_threshold=confidence,
        bars_per_year=8760 if timeframe == "1h" else 35040,
    )

    m = bt.get("metrics", {})
    return {
        "Symbol": symbol,
        "Timeframe": timeframe,
        "Accuracy (%)": f"{train_results['overall_accuracy'] * 100:.1f}%",
        "Trades": m.get("total_trades", 0),
        "Win Rate (%)": f"{m.get('win_rate_pct', 0.0):.1f}%",
        "Strategy Ret (%)": f"{m.get('total_return_pct', 0.0):+.2f}%",
        "Buy & Hold (%)": f"{m.get('benchmark_return_pct', 0.0):+.2f}%",
        "Sharpe": f"{m.get('sharpe_ratio', 0.0):.2f}",
        "Max DD (%)": f"{m.get('max_drawdown_pct', 0.0):.2f}%",
        "Profit Factor": f"{m.get('profit_factor', 0.0):.2f}",
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Asset Crypto ML Screener")
    parser.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT", "SOL/USDT"], help="Symbols to screen")
    parser.add_argument("--timeframes", nargs="+", default=["1h"], help="Timeframes to screen (e.g. 1h 15m)")
    parser.add_argument("--bars", type=int, default=700, help="Number of candles per asset")
    parser.add_argument("--confidence", type=float, default=0.35, help="Confidence threshold")
    args = parser.parse_args()

    results = []
    for tf in args.timeframes:
        for sym in args.symbols:
            try:
                res = benchmark_asset(sym, timeframe=tf, bars=args.bars, confidence=args.confidence)
                results.append(res)
            except Exception as e:
                print(f"[!] Error benchmarking {sym} ({tf}): {e}")

    summary_df = pd.DataFrame(results)
    summary_csv = "reports/screener_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n" + "=" * 90)
    print("                      MULTI-ASSET CRYPTO ML SCREENER SUMMARY")
    print("=" * 90)
    print(summary_df.to_string(index=False))
    print("=" * 90)
    print(f"[+] Saved summary to {summary_csv}\n")


if __name__ == "__main__":
    main()
