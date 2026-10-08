# AGENTS.md — shared notes for every AI agent on this repo

This project is worked on by more than one AI agent (Claude / Cowork and Google Antigravity),
plus Giorgos. This file is the shared source of truth. Read it before you change anything,
and add a line to the **Change log** at the bottom when you finish a piece of work.

## Ground rules

1. **Everything lives inside this folder.** Code, configs, data caches, models, reports, logs,
   scratch scripts — all under `pricePredictor/`. No global installs, no files in `~` or `/tmp`
   that the project depends on. If you need a throwaway script, put it in `scratch/` (gitignored).
2. **One Python environment: `.venv/` (macOS, Python 3.9.6).** Run things with `.venv/bin/python`.
   Keep code **Python 3.9 compatible**: no `X | Y` type unions, no `match`, use `typing.Optional/Union`.
   Add new dependencies to `requirements.txt` in the same change.
3. **Don't clobber another agent's in-flight work.** Check `git status` first. Uncommitted changes
   may belong to someone else — don't revert, reformat or commit them unless asked.
4. **Keep the tests green:** `.venv/bin/python -m unittest tests.test_pipeline` (29 tests, ~12 s).
   Tests must never touch `data/paper_trading.db` or real caches — use temp paths.
5. **Run commands from the repo root.** All paths in the code are relative (`data/raw`, `models/`, ...).
6. **Log what you did** in the Change log below (date, agent, one line).

## Layout

| Path | What it is |
|---|---|
| `run_pipeline.py` | Main train + walk-forward backtest runner (many `--use-*` flags). Saves `models/<SYM>_<TF>_model.joblib`. |
| `run_meta_pipeline.py` | Primary vs meta-labeling comparison runner. |
| `screen_assets.py` | Multi-asset benchmark → `reports/screener_summary.csv`. |
| `live_predictor.py` | Single-symbol live inference / paper bot. |
| `multi_asset_scanner.py` | Multi-asset live scanner + capital allocator (used by the daemon). |
| `report_dashboard.py` | Terminal dashboard over the paper-trading DB. |
| `run_live_2weeks.sh` / `stop_live_2weeks.sh` / `check_status.sh` | Background paper-trading daemon (macOS `caffeinate`), PID in `data/trader.pid`, log in `logs/live_trader.log`. |
| `src/` | Library code: data_loader, features, labeling, model, meta_labeling, backtest, conformal, regime_hmm, fracdiff, microstructure, feature_selection, optimizer, paper_tracker, utils. |
| `configs/default_config.yaml` | Defaults; CLI flags override. |
| `data/raw/` | Cached OHLCV / derivatives / Fear&Greed CSVs (gitignored). |
| `data/paper_trading.db` | Live paper-trading ledger (SQLite). **Real state — don't reset without asking.** |
| `models/`, `reports/`, `logs/` | Generated artifacts. |

## Known issues (found 2026-10-08, not fixed yet)

- **Live runs overwrite the training cache.** `multi_asset_scanner.py` / `live_predictor.py` call
  `fetch_multitimeframe_ohlcv(primary_bars=120)`, and `fetch_ohlcv_ccxt` writes whatever it fetched
  to the same `data/raw/<SYM>_<TF>.csv`. The 1h caches currently hold only ~120 bars, so the next
  training run re-downloads (or, offline, trains on 120 bars). Fix idea: merge new candles into the
  existing cache instead of replacing it, or use a separate live cache.
- **Silent synthetic-data fallback.** If the exchange fetch fails and no cache exists, `data_loader`
  returns synthetic candles. Fine for tests, dangerous for live/paper trading — should raise (or be
  opt-in via a flag) outside tests.
- **Cache freshness check** treats every timeframe other than `1h`/`15m` as 1 day (wrong for `4h`).
- **Model fallback:** the scanner silently uses the BTC model for any symbol without its own model.
- **Tiny out-of-sample samples.** With ~700 bars, backtests produce 2–8 trades
  (`reports/screener_summary.csv`). Win-rate / Sharpe numbers at that size are noise; don't tune to them.
- `.gitignore` doesn't cover `logs/`, `data/trader.pid`, or `scratch/`.

## Change log

- 2026-10-08 · Claude (Cowork) · Reviewed repo, ran tests (29/29 OK in a separate Linux env), created this file. No code changes.
