"""Live paper trader for the core model (no real orders are ever sent).

Every hour, ~15 s after the candle closes:
  1. append the new candles / funding to data/history (same cache research uses),
  2. retrain if the model is older than the walk-forward retrain interval (30 days),
  3. predict P(up) for the just-closed bar, convert to expected return, apply the SAME
     cost-aware rule as the backtest (src/core/strategy.step_position),
  4. paper-fill at the live price +/- slippage, charge taker fees and perp funding,
  5. log everything to data/paper_core.db.

    .venv/bin/python paper_trader.py            # run forever
    .venv/bin/python paper_trader.py --once     # one decision cycle now (uses last closed bar)
    .venv/bin/python paper_trader.py --status   # print ledger summary
"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore", message=".*OpenSSL.*")
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.core import history
from src.core.deploy import DEFAULT_SPEC, latest_signals, load_model, model_age_days, model_path, train_final
from src.core.strategy import step_position

# One ledger per deployed universe. data/paper_core.db = the 3-coin run of 2026-10-10 (archived).
DB_PATH = Path(os.environ.get("PAPER_DB", f"data/paper_{DEFAULT_SPEC.get('universe', 'u3')}.db"))
START_CAPITAL = 10_000.0
TAKER_FEE = 0.0005
SLIPPAGE = 0.0002
TICKER_URL = "https://api.binance.com/api/v3/ticker/price"


def log(msg: str) -> None:
    print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z {msg}", flush=True)


# --------------------------------------------------------------------------- ledger

def db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS signals (
        decided_at TEXT, symbol TEXT, bar_open TEXT, prob_up REAL, exp_ret REAL, hurdle REAL,
        pos_before REAL, pos_after REAL, live_price REAL, bar_close_price REAL, model_trained_at TEXT,
        priced_at TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, side REAL, status TEXT,
        entry_time TEXT, entry_price REAL, qty REAL, notional REAL,
        exit_time TEXT, exit_price REAL, fees REAL DEFAULT 0, funding REAL DEFAULT 0,
        last_funding_ts TEXT, pnl REAL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS sleeves (symbol TEXT PRIMARY KEY, cash REAL)""")
    con.commit()
    return con


def ensure_sleeves(con, symbols):
    for s in symbols:
        con.execute("INSERT OR IGNORE INTO sleeves(symbol, cash) VALUES (?, ?)", (s, START_CAPITAL / len(symbols)))
    con.commit()


def open_trade(con, symbol):
    r = con.execute("SELECT id, side, entry_price, qty, notional, fees, funding, last_funding_ts, entry_time "
                    "FROM trades WHERE symbol=? AND status='OPEN'", (symbol,)).fetchone()
    if not r:
        return None
    keys = ["id", "side", "entry_price", "qty", "notional", "fees", "funding", "last_funding_ts", "entry_time"]
    return dict(zip(keys, r))


def live_prices(symbols) -> dict:
    """Last trade price for all symbols in ONE request (keeps decision latency low)."""
    import json as _json
    syms = [x.replace("/", "") for x in symbols]
    r = requests.get(TICKER_URL, params={"symbols": _json.dumps(syms, separators=(",", ":"))}, timeout=10)
    r.raise_for_status()
    by = {d["symbol"]: float(d["price"]) for d in r.json()}
    return {s: by[s.replace("/", "")] for s in symbols}


def accrue_funding(con, trade, funding_df, now):
    """Charge funding for settlements that happened while the position was open."""
    if trade is None or funding_df is None or funding_df.empty:
        return
    since = pd.Timestamp(trade["last_funding_ts"] or trade["entry_time"])
    fr = funding_df["funding_rate"]
    due = fr[(fr.index > since) & (fr.index <= now)]
    if due.empty:
        return
    # funding paid by longs when positive: cash flow = -side * notional * rate
    flow = float(sum(-trade["side"] * trade["notional"] * r for r in due.values))
    con.execute("UPDATE trades SET funding = funding + ?, last_funding_ts=? WHERE id=?",
                (flow, str(due.index[-1]), trade["id"]))


def close_trade(con, trade, price, now):
    side = trade["side"]
    fill = price * (1 - SLIPPAGE * side)
    fee = abs(trade["qty"] * fill) * TAKER_FEE
    gross = side * trade["qty"] * (fill - trade["entry_price"])
    fees = trade["fees"] + fee
    pnl = gross - fees + trade["funding"]
    con.execute("UPDATE trades SET status='CLOSED', exit_time=?, exit_price=?, fees=?, pnl=? WHERE id=?",
                (str(now), fill, fees, pnl, trade["id"]))
    con.execute("UPDATE sleeves SET cash = cash + ? WHERE symbol=?", (pnl, symbol_of(con, trade["id"])))
    return pnl


def symbol_of(con, trade_id):
    return con.execute("SELECT symbol FROM trades WHERE id=?", (trade_id,)).fetchone()[0]


def enter_trade(con, symbol, side, price, now):
    cash = con.execute("SELECT cash FROM sleeves WHERE symbol=?", (symbol,)).fetchone()[0]
    fill = price * (1 + SLIPPAGE * side)
    notional = cash  # 1x: whole sleeve
    qty = notional / fill
    fee = notional * TAKER_FEE
    con.execute("INSERT INTO trades(symbol, side, status, entry_time, entry_price, qty, notional, fees, funding, "
                "last_funding_ts) VALUES (?,?,?,?,?,?,?,?,0,?)",
                (symbol, side, "OPEN", str(now), fill, qty, notional, fee, str(now)))


# --------------------------------------------------------------------------- one cycle

def cycle(spec: dict) -> None:
    now = pd.Timestamp.now(tz="UTC")
    expected_bar = now.floor("1h") - pd.Timedelta(hours=1)
    symbols = list(spec["symbols"])
    data = None
    for attempt in range(6):
        data = history.update_all(symbols, spec["timeframe"], spec["data_start"], verbose=False, workers=8)
        stale = [s for s in symbols if data["ohlcv"][s].index[-1] < expected_bar]
        if not stale:
            break
        log(f"waiting for bar {expected_bar:%H:%M} on {', '.join(stale)} ...")
        time.sleep(10)
    if stale:
        log(f"[!] no fresh candle for {', '.join(stale)} - skipping them this hour (delisted/outage?)")
        symbols = [s for s in symbols if s not in stale]
        if "BTC/USDT" in stale or not symbols:
            log("[!] market data unavailable, skipping this hour")
            return

    need_train = not model_path(spec).exists()
    if need_train:
        log("no model found - training (one-off) ...")
        train_final(spec, offline=False, verbose=False)
    bundle = load_model(spec)

    sigs = latest_signals(bundle, data, symbols)
    prices = live_prices(symbols)
    priced_at = pd.Timestamp.now(tz="UTC")

    con = db()
    ensure_sleeves(con, spec["symbols"])
    allow_short = spec["policy"] == "cost_ls"
    changes = []
    for sig in sigs:
        sym = sig["symbol"]
        trade = open_trade(con, sym)
        accrue_funding(con, trade, data["funding"].get(sym), now)
        trade = open_trade(con, sym)
        cur = trade["side"] if trade else 0.0
        new = step_position(cur, sig["exp_ret"], spec["cost_per_side"], allow_short)
        price = prices[sym]
        if new != cur:
            if trade:
                pnl = close_trade(con, trade, price, priced_at)
                changes.append(f"CLOSE {sym} @ {price:,.6g} pnl={pnl:+.2f}")
            if new != 0:
                enter_trade(con, sym, new, price, priced_at)
                changes.append(f"OPEN {'LONG' if new > 0 else 'SHORT'} {sym} @ {price:,.6g} (E[r]={sig['exp_ret'] * 1e4:+.0f}bps)")
        con.execute("INSERT INTO signals VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (str(now), sym, str(sig["bar_open"]), sig["prob_up"], sig["exp_ret"], sig["hurdle"],
                     cur, new, price, sig["bar_close_price"], bundle["meta"]["trained_at"], str(priced_at)))
    con.commit()
    holding = [r[0] for r in con.execute("SELECT symbol FROM trades WHERE status='OPEN'")]
    con.close()
    best = max(sigs, key=lambda x: x["exp_ret"])
    log(f"bar {expected_bar:%H:%M}: {len(sigs)} coins, priced {(priced_at - now.floor('1h')).total_seconds():.0f}s after close, "
        f"best E[r]={best['exp_ret'] * 1e4:+.1f}bps ({best['symbol']}) vs hurdle {best['hurdle'] * 1e4:.0f}bps, "
        f"holding: {', '.join(holding) or 'nothing'}")
    for c in changes:
        log("  " + c)

    # Retrain AFTER acting, so the monthly retrain never delays a decision.
    if not need_train and model_age_days(bundle) > spec["retrain_days"]:
        log("retraining model (older than retrain interval) ...")
        train_final(spec, offline=False, verbose=False)
        log("retrain done")


# --------------------------------------------------------------------------- status

def status() -> None:
    con = db()
    sleeves = pd.read_sql("SELECT * FROM sleeves", con)
    trades = pd.read_sql("SELECT * FROM trades", con)
    sigs = pd.read_sql("SELECT * FROM signals", con)
    con.close()
    if sleeves.empty:
        print("No paper-trading activity yet.")
        return
    closed = trades[trades.status == "CLOSED"]
    print(f"Signals logged : {len(sigs)}  (first {sigs.decided_at.min() if len(sigs) else '-'}, last {sigs.decided_at.max() if len(sigs) else '-'})")
    print(f"Closed trades  : {len(closed)}  win rate {100 * (closed.pnl > 0).mean() if len(closed) else 0:.1f}%")
    print(f"Realized PnL   : {closed.pnl.sum():+,.2f} USD   fees {trades.fees.sum():,.2f}   funding {trades.funding.sum():+,.2f}")
    print(f"Universe       : {DEFAULT_SPEC.get('universe')} ({len(DEFAULT_SPEC['symbols'])} coins), ledger {DB_PATH}")
    print(f"Cash (all sleeves): {sleeves.cash.sum():,.2f} USD  (start {START_CAPITAL:,.0f})")
    op = trades[trades.status == "OPEN"]
    for _, t in op.iterrows():
        print(f"  OPEN {t.symbol} {'LONG' if t.side > 0 else 'SHORT'} since {t.entry_time} @ {t.entry_price:,.2f}")
    if len(sigs):
        print("\nLatest signals:")
        last = sigs[sigs.decided_at == sigs.decided_at.max()].copy()
        last["exp_ret_bps"] = (1e4 * last["exp_ret"]).round(1)
        print(last.sort_values("exp_ret", ascending=False)[["symbol", "prob_up", "exp_ret_bps", "pos_after"]]
              .round(3).to_string(index=False))
        if "priced_at" in sigs.columns and sigs["priced_at"].notna().any():
            lat = (pd.to_datetime(sigs.priced_at) - pd.to_datetime(sigs.decided_at).dt.floor("h")).dt.total_seconds()
            slip = 1e4 * np.log(sigs.live_price / sigs.bar_close_price)
            print(f"\nLatency: median {lat.median():.0f}s after close; live vs bar-close price: "
                  f"mean {slip.mean():+.1f} bps, mean |diff| {slip.abs().mean():.1f} bps")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--delay", type=int, default=15, help="seconds after the hour to act")
    a = ap.parse_args()
    spec = dict(DEFAULT_SPEC)
    if a.status:
        return status()
    if a.once:
        return cycle(spec)
    log(f"paper trader started: {spec.get('universe')} {len(spec['symbols'])} coins model={spec['model']} h={spec['horizon']} policy={spec['policy']}")
    while True:
        now = pd.Timestamp.now(tz="UTC")
        nxt = now.floor("1h") + pd.Timedelta(hours=1, seconds=a.delay)
        time.sleep(max(1.0, (nxt - now).total_seconds()))
        try:
            cycle(spec)
        except Exception as e:  # keep running; never trade on missing/synthetic data
            log(f"[!] cycle failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    sys.exit(main())
