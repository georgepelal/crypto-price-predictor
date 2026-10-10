# AGENTS.md — shared notes for every AI agent on this repo

This project is worked on by more than one AI agent (Claude / Cowork and Google Antigravity),
plus Giorgos. This file is the shared source of truth. Read it before you change anything,
and add a line to the **Change log** at the bottom when you finish a piece of work.
Research results and decisions go in `docs/RESEARCH_LOG.md`.

## Ground rules

1. **Everything lives inside this folder.** Code, configs, data caches, models, reports, logs,
   scratch scripts — all under `pricePredictor/`. No global installs, no files in `~` or `/tmp`
   that the project depends on. Throwaway scripts go in `scratch/` (gitignored).
2. **One Python environment: `.venv/` (macOS, Python 3.9.6).** Run things with `.venv/bin/python`
   from the repo root. Keep code **Python 3.9 compatible** (no `X | Y` unions, no `match`).
   `requirements.txt` is pinned to the exact versions in `.venv`; change both together.
3. **Don't clobber another agent's in-flight work.** Check `git status` first. Commit your own work in
   small commits; don't revert or reformat someone else's uncommitted changes.
4. **Keep the tests green:** `.venv/bin/python -m unittest tests.test_core tests.test_pipeline`.
   Tests must never touch `data/paper_core.db`, `data/paper_trading.db` or `data/history/`.
5. **Honest evaluation only.** Never tune anything on the out-of-sample period and then report it as
   out-of-sample. Declare a config in `docs/RESEARCH_LOG.md` *before* running it. Report the whole grid.
6. **No synthetic data outside tests.** If a download fails, fail loudly.

## Two pipelines

### Core pipeline (current, `src/core/`) — use this
| Path | What it is |
|---|---|
| `src/core/history.py` | Binance spot klines (incl. taker-buy volume), perp funding, Fear & Greed. Append-only cache in `data/history/`. Never stores the open candle. Raises on failure. |
| `src/core/features.py` | `build_features()` — the ONE feature builder for research, training and live. Rows indexed by candle open time; features use only data known at candle close. |
| `src/core/dataset.py` | Pooled multi-symbol table + targets (`fwd_ret`, `target`, `ret_next`). |
| `src/core/models.py` | `lgbm` and `logreg`, fixed conservative hyper-parameters. |
| `src/core/walkforward.py` | Rolling-origin evaluation: retrain every 30 days, purge = horizon. |
| `src/core/strategy.py` | Positions, costs, funding; `step_position()` = cost-aware rule shared by backtest and live. |
| `src/core/metrics.py` | AUC, log-loss skill, monthly IC, Sharpe with block-bootstrap CI. |
| `src/core/deploy.py` | `DEFAULT_SPEC` (what is deployed), `train_final()`, `latest_signals()`. |
| `research.py` | Full walk-forward grid + baselines → `reports/research/<run>/summary.md`, `equity.png`. `--universe u3/u20`, `--primary model:h:policy`, `--reuse` + `--max-minutes` to run long grids in resumable chunks (memory: U20 needs ~2.5 GB). |
| `train.py` | Train the production model on all history → `models/core_lgbm_1h_h1.joblib` (+ `.json` metadata). |
| `paper_trader.py` | Hourly paper trader (no real orders) → `data/paper_<universe>.db` (override with env `PAPER_DB`). `--once`, `--status`. Parallel data refresh + one batched price request (~5 s per cycle for 18 coins). Retrains every 30 days, after acting. Skips coins with no fresh candle. |
| `src/core/universe.py` | Named universes (`u3`, `u20`), delisted list, `tradable()`. Shared by research and live. |
| `run_live_2weeks.sh` / `stop_live_2weeks.sh` / `check_status.sh` | Background daemon for `paper_trader.py` (macOS `caffeinate`), log in `logs/live_trader.log`. |
| `tests/test_core.py` | No-lookahead (truncation) test, purge test, P&L timing/cost tests, cache-merge test. |

### Legacy pipeline (Antigravity, before 2026-10-08) — kept, not deployed
`run_pipeline.py`, `run_meta_pipeline.py`, `screen_assets.py`, `live_predictor.py`,
`multi_asset_scanner.py`, `report_dashboard.py`, `src/*.py` (except `src/core/`), `data/raw/`,
`data/paper_trading.db`. Known problems if you revive it:
- Trained on ~700 bars → 2–8 out-of-sample trades; its reported win rates/Sharpes are noise.
- Optuna tunes thresholds on the same out-of-sample predictions it reports (selection bias).
- `data_loader` falls back to **synthetic data** silently; `.bfill()` after merges leaks future values
  into early rows; fracdiff picks `d` via ADF on the full sample.
- Live scanner computes features from 120 bars while training used long rolling windows (train/serve mismatch),
  and silently uses the BTC model for coins without a model.
- (Fixed 2026-10-08) short live fetches used to overwrite the training cache; `fetch_ohlcv_ccxt` now merges.

## Current status (2026-10-08)
- Deployed to paper (since 2026-10-10 ~11:00 UTC): **lgbm, 1h horizon, cost-aware long/flat, trained on U20
  (`src/core/universe.py`), trading the 18 still-listed U20 coins**. Ledger `data/paper_u20.db`, model
  `models/core_lgbm_1h_h1_u20.joblib`. The earlier 3-coin run (a few hours) is archived in `data/paper_core.db`.
- Out-of-sample 2023-01 → 2026-10: real predictive skill (AUC 0.557, IC t≈15) but the tradeable edge
  after costs is small and episodic. Details and caveats: `docs/RESEARCH_LOG.md`.
- 2026-10-10: experiment 2 (20 coins) passed its pre-declared test (Sharpe +1.85, CI [+0.97, +2.80]) but the
  edge has been ~flat since 2025-01. Proposed to Giorgos: switch the paper trader to the 20-coin model.
- Paper trader runs on Giorgos's Mac (3 coins 09:43–~11:00 UTC 2026-10-10, then U20). Live `signals.live_price` vs
  `bar_close_price` measures real latency slippage.

## Change log
- 2026-10-08 · Claude (Cowork) · Reviewed repo, ran tests, created this file.
- 2026-10-10 · Claude (Cowork) · Switched deployment to U20: `DEFAULT_SPEC` (train on 20, trade 18), new ledger `data/paper_u20.db`, faster cycle (parallel refresh, batched prices), latency logged (`signals.priced_at`), `--status` shows latency & live-vs-close slippage. Note: in the Cowork Linux VM, sqlite on the mounted folder fails without delete permission — test with `PAPER_DB=$HOME/x.db`.
- 2026-10-10 · Claude (Cowork) · Experiment 2 (20-coin universe, pre-declared): passed. Added `research.py --universe u20 --primary ... --max-minutes` (resumable, fold cache in `--out`), funding symbol overrides (1000SHIB), float32 datasets, equal weight across listed coins, `scripts/exp2_checks.py`, `scripts/exp2_latency.py`. Paper trader switched to U20 after Giorgos agreed (same day).
- 2026-10-10 · Claude (Cowork) · Fixed lost exec bit on run_live_2weeks.sh, added env pre-flight check (macOS needs `brew install libomp` for LightGBM), silenced urllib3 LibreSSL warning.
- 2026-10-08 · Claude (Cowork) · Baseline commit of Antigravity's uncommitted work (branch `honest-ml`).
  Added core pipeline (`src/core/`, `research.py`, `train.py`, `paper_trader.py`, `tests/test_core.py`,
  `docs/RESEARCH_LOG.md`), pinned requirements, daemon scripts now run `paper_trader.py`,
  legacy `fetch_ohlcv_ccxt` merges into cache instead of overwriting.
