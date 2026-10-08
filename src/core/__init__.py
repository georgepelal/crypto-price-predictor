"""Core ML pipeline (clean rewrite, 2026-10).

history   -> download + incrementally cache real market data (no synthetic fallback)
features  -> one feature builder shared by research, training and live trading
dataset   -> features + forward-return targets, pooled across symbols
walkforward -> rolling-origin out-of-sample evaluation (purged, no tuning on test data)
strategy  -> probabilities -> positions -> net returns after costs
metrics   -> classification + trading metrics, baselines, bootstrap confidence intervals
"""
