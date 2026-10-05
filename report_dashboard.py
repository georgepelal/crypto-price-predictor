"""Real-Time Terminal Paper Trading & Model Performance Dashboard.

Displays portfolio equity, open positions, win rate, realized PnL,
recent executions, and latest model inference signals with clean ANSI formatting.
"""

import argparse
from datetime import datetime, timezone
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

DEFAULT_DB_PATH = "data/paper_trading.db"

# ANSI Terminal Color Helpers
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def format_usd(val: float, signed: bool = False) -> str:
    sign = "+" if signed and val > 0 else ""
    return f"{sign}${val:,.2f}"


def format_pct(val: float, signed: bool = False) -> str:
    sign = "+" if signed and val > 0 else ""
    return f"{sign}{val:.2f}%"


def color_pnl(val: float, text: str) -> str:
    if val > 0:
        return f"{GREEN}{text}{RESET}"
    elif val < 0:
        return f"{RED}{text}{RESET}"
    return f"{DIM}{text}{RESET}"


def fetch_dashboard_data(db_path: str = DEFAULT_DB_PATH, starting_capital: float = 10000.0) -> Dict[str, Any]:
    """Fetch structured performance data from SQLite database."""
    if not os.path.exists(db_path):
        return {"error": f"Database '{db_path}' does not exist yet. Please run live_predictor.py first."}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        trades_df = pd.read_sql_query("SELECT * FROM trades ORDER BY id ASC;", conn)
        signals_df = pd.read_sql_query("SELECT * FROM signals_log ORDER BY id DESC LIMIT 5;", conn)
    except Exception as e:
        conn.close()
        return {"error": f"Database read error: {e}"}

    conn.close()

    if trades_df.empty:
        return {
            "starting_capital": starting_capital,
            "realized_pnl": 0.0,
            "equity": starting_capital,
            "win_rate": 0.0,
            "total_trades": 0,
            "closed_trades": 0,
            "open_positions": [],
            "recent_trades": [],
            "recent_signals": signals_df.to_dict(orient="records") if not signals_df.empty else [],
            "profit_factor": 0.0,
            "total_fees": 0.0,
        }

    closed_df = trades_df[trades_df["status"] == "CLOSED"]
    open_df = trades_df[trades_df["status"] == "OPEN"]

    winning = closed_df[closed_df["pnl_usd"] > 0]
    losing = closed_df[closed_df["pnl_usd"] < 0]

    win_count = len(winning)
    loss_count = len(losing)
    total_closed = len(closed_df)
    win_rate = (win_count / total_closed * 100.0) if total_closed > 0 else 0.0

    gross_win = winning["pnl_usd"].sum() if not winning.empty else 0.0
    gross_loss = abs(losing["pnl_usd"].sum()) if not losing.empty else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)

    realized_pnl = float(closed_df["pnl_usd"].sum()) if total_closed > 0 else 0.0
    total_fees = float(trades_df["fees_paid"].sum()) if not trades_df.empty else 0.0
    current_equity = starting_capital + realized_pnl

    return {
        "starting_capital": starting_capital,
        "realized_pnl": realized_pnl,
        "equity": current_equity,
        "win_rate": win_rate,
        "win_count": win_count,
        "loss_count": loss_count,
        "total_trades": len(trades_df),
        "closed_trades": total_closed,
        "profit_factor": profit_factor,
        "total_fees": total_fees,
        "open_positions": open_df.to_dict(orient="records"),
        "recent_trades": closed_df.tail(6).to_dict(orient="records"),
        "recent_signals": signals_df.to_dict(orient="records") if not signals_df.empty else [],
    }


def render_dashboard(data: Dict[str, Any]) -> None:
    """Print beautifully formatted ANSI terminal dashboard."""
    if "error" in data:
        print(f"\n{RED}[!] {data['error']}{RESET}\n")
        return

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    pnl = data["realized_pnl"]
    pnl_pct = (pnl / data["starting_capital"]) * 100.0

    pnl_str = color_pnl(pnl, f"{format_usd(pnl, signed=True)} ({format_pct(pnl_pct, signed=True)})")
    wr_str = f"{data['win_rate']:.1f}% ({data['win_count']}W / {data['loss_count']}L)"
    if data["win_rate"] >= 60.0:
        wr_str = f"{GREEN}{wr_str}{RESET}"
    elif data["win_rate"] < 45.0 and data["closed_trades"] > 0:
        wr_str = f"{RED}{wr_str}{RESET}"

    pf_str = f"{data['profit_factor']:.2f}"
    if data["profit_factor"] >= 1.5:
        pf_str = f"{GREEN}{pf_str}{RESET}"

    # Clear terminal screen cleanly if watching
    print("\n" + "=" * 78)
    print(f" {BOLD}QUANT CRYPTO PAPER TRADING DASHBOARD{RESET}        {DIM}{now_utc}{RESET}")
    print("=" * 78)

    # 1. Performance Summary Box
    print(f" {BOLD}PORTFOLIO PERFORMANCE{RESET}")
    print(f"   Initial Capital : {format_usd(data['starting_capital'])}      Equity Balance : {format_usd(data['equity'])}")
    print(f"   Realized PnL    : {pnl_str}     Profit Factor  : {pf_str}")
    print(f"   Trade Win Rate  : {wr_str}         Total Fees     : {format_usd(data['total_fees'])}")
    print(f"   Total Orders    : {data['total_trades']} (Closed: {data['closed_trades']}, Open: {len(data['open_positions'])})")
    print("-" * 78)

    # 2. Active Open Positions
    print(f" {BOLD}ACTIVE POSITIONS ({len(data['open_positions'])}){RESET}")
    if not data["open_positions"]:
        print(f"   {DIM}No positions currently open. (Market in Neutral / Waiting for Setup){RESET}")
    else:
        print(f"   {'ID':<4} {'Symbol':<10} {'Action':<6} {'Qty':<10} {'Entry':<12} {'Stop-Loss':<14} {'Take-Profit':<12} {'Status'}")
        print(f"   {'-'*72}")
        for pos in data["open_positions"]:
            action_color = GREEN if pos["action"] == "BUY" else RED
            action_lbl = f"{action_color}{pos['action']}{RESET}"
            sl_val = pos.get("stop_loss")
            tp_val = pos.get("take_profit")
            sl_str = f"${sl_val:,.2f}" if sl_val else "N/A"
            tp_str = f"${tp_val:,.2f}" if tp_val else "N/A"

            # Check if profit-lock is active
            status_tag = "ACTIVE"
            if pos["action"] == "BUY" and sl_val and sl_val > pos["entry_price"]:
                status_tag = f"{GREEN}PROFIT-LOCK{RESET}"
            elif pos["action"] == "SELL" and sl_val and sl_val < pos["entry_price"]:
                status_tag = f"{GREEN}PROFIT-LOCK{RESET}"

            print(f"   #{pos['id']:<3} {pos['symbol']:<10} {action_lbl:<15} {pos['quantity']:<10.4f} ${pos['entry_price']:<11,.2f} {sl_str:<14} {tp_str:<12} {status_tag}")
    print("-" * 78)

    # 3. Recent Trades
    print(f" {BOLD}RECENT COMPLETED TRADES{RESET}")
    if not data["recent_trades"]:
        print(f"   {DIM}No closed trades yet.{RESET}")
    else:
        print(f"   {'ID':<4} {'Symbol':<10} {'Type':<6} {'Entry':<11} {'Exit':<11} {'Reason':<16} {'Net PnL':<18}")
        print(f"   {'-'*72}")
        for t in reversed(data["recent_trades"]):
            act_col = GREEN if t["action"] == "BUY" else RED
            pnl_val = t.get("pnl_usd", 0.0)
            pct_val = t.get("pnl_pct", 0.0)
            pnl_repr = color_pnl(pnl_val, f"{format_usd(pnl_val, signed=True)} ({pct_val:+.2f}%)")
            reason_str = t.get("exit_reason", "")
            if reason_str in ["TAKE_PROFIT", "PROFIT_LOCK"]:
                reason_str = f"{GREEN}{reason_str}{RESET}"
            elif reason_str == "STOP_LOSS":
                reason_str = f"{RED}{reason_str}{RESET}"

            print(f"   #{t['id']:<3} {t['symbol']:<10} {act_col}{t['action']:<6}{RESET} ${t['entry_price']:<10,.2f} ${t['exit_price']:<10,.2f} {reason_str:<25} {pnl_repr}")
    print("-" * 78)

    # 4. Latest Logged Inference Signals
    print(f" {BOLD}LATEST MODEL SIGNALS (Inference Log){RESET}")
    if not data["recent_signals"]:
        print(f"   {DIM}No signals logged yet.{RESET}")
    else:
        print(f"   {'Timestamp':<20} {'Price':<11} {'P(Down)':<9} {'P(Flat)':<9} {'P(Up)':<9} {'Edge':<9} {'Signal'}")
        print(f"   {'-'*72}")
        for s in data["recent_signals"]:
            edge_val = s.get("edge", 0.0)
            sig_text = s.get("recommended_action", "")
            col = GREEN if ("BUY" in sig_text or "LONG" in sig_text) else (RED if ("SELL" in sig_text or "SHORT" in sig_text) else YELLOW)
            sig_fmt = f"{col}{sig_text[:24]}{RESET}"
            p_down_str = f"{s['p_down']*100:.1f}%"
            p_flat_str = f"{s['p_neutral']*100:.1f}%"
            p_up_str = f"{s['p_up']*100:.1f}%"
            edge_str = f"{edge_val*100:+.1f}%"
            print(f"   {s['timestamp'][5:19]:<20} ${s['price']:<10,.2f} {p_down_str:<9} {p_flat_str:<9} {p_up_str:<9} {edge_str:<9} {sig_fmt}")

    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Live Crypto Quant Paper Trading Dashboard")
    parser.add_argument("--db", type=str, default=DEFAULT_DB_PATH, help="Path to SQLite database")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial virtual capital")
    parser.add_argument("--watch", action="store_true", help="Auto-refresh dashboard continuously")
    parser.add_argument("--interval", type=int, default=10, help="Refresh interval in seconds when --watch is active")
    args = parser.parse_args()

    if not args.watch:
        data = fetch_dashboard_data(db_path=args.db, starting_capital=args.capital)
        render_dashboard(data)
        return

    print(f"[*] Starting Real-Time Dashboard Watcher (Refreshes every {args.interval}s)... Press Ctrl+C to exit.")
    while True:
        try:
            # Clear screen ANSI sequence
            sys.stdout.write("\033[H\033[J")
            data = fetch_dashboard_data(db_path=args.db, starting_capital=args.capital)
            render_dashboard(data)
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[*] Exiting dashboard.")
            sys.exit(0)


if __name__ == "__main__":
    main()
