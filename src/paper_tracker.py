"""SQLite-Backed Automated Paper Trading & Real-Time Performance Tracker.

Manages virtual orders, position sizing, dynamic ATR Stop-Loss / Take-Profit tracking,
trade execution simulation, and PnL logging with zero external server dependencies.
"""

from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
from typing import Dict, Any, Optional, List
import pandas as pd


DEFAULT_DB_PATH = "data/paper_trading.db"


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Return an active SQLite database connection with row factory."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initialize SQLite database tables for trades, signals, and account equity."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        # 1. Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                status TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity REAL NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                exit_time TEXT,
                exit_price REAL,
                exit_reason TEXT,
                pnl_usd REAL DEFAULT 0.0,
                pnl_pct REAL DEFAULT 0.0,
                fees_paid REAL DEFAULT 0.0
            );
        """)

        # 2. Signals Log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                p_down REAL NOT NULL,
                p_neutral REAL NOT NULL,
                p_up REAL NOT NULL,
                edge REAL NOT NULL,
                recommended_action TEXT NOT NULL,
                executed INTEGER NOT NULL DEFAULT 0
            );
        """)

        # 3. Account History table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS account_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cash_balance REAL NOT NULL,
                equity REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                open_positions INTEGER NOT NULL
            );
        """)
        conn.commit()


def get_open_position(symbol: str, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieve current OPEN position for a symbol, if any."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM trades WHERE symbol = ? AND status = 'OPEN' ORDER BY id DESC LIMIT 1;",
            (symbol,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def execute_paper_candle(
    symbol: str,
    candle_time: datetime,
    open_price: float,
    high_price: float,
    low_price: float,
    close_price: float,
    signal: str,
    sl_price: Optional[float] = None,
    tp_price: Optional[float] = None,
    prob_dict: Optional[Dict[int, float]] = None,
    edge: float = 0.0,
    starting_capital: float = 10000.0,
    risk_per_trade: float = 0.02,
    taker_fee: float = 0.0005,
    slippage: float = 0.0002,
    db_path: str = DEFAULT_DB_PATH,
) -> Dict[str, Any]:
    """Evaluate candle against open positions (SL/TP) and execute new entries.

    Returns summary of action taken on this candle.
    """
    init_db(db_path)
    time_str = candle_time.strftime("%Y-%m-%d %H:%M:%S UTC")
    action_log = []

    probs = prob_dict or {}
    p_down = probs.get(0, 0.0)
    p_neutral = probs.get(1, 0.0)
    p_up = probs.get(2, 0.0)

    cost_rate = taker_fee + slippage

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        # 1. Log the signal
        cursor.execute(
            """INSERT INTO signals_log (timestamp, symbol, price, p_down, p_neutral, p_up, edge, recommended_action, executed)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);""",
            (time_str, symbol, close_price, p_down, p_neutral, p_up, edge, signal, 0),
        )
        signal_id = cursor.lastrowid

        # 2. Check open position for this symbol
        cursor.execute(
            "SELECT * FROM trades WHERE symbol = ? AND status = 'OPEN' ORDER BY id DESC LIMIT 1;",
            (symbol,),
        )
        pos = cursor.fetchone()

        if pos:
            pos_dict = dict(pos)
            pos_id = pos_dict["id"]
            pos_action = pos_dict["action"]
            entry_p = pos_dict["entry_price"]
            pos_sl = pos_dict["stop_loss"]
            pos_tp = pos_dict["take_profit"]
            qty = pos_dict["quantity"]

            closed = False
            exit_p = close_price
            reason = ""

            if pos_action == "BUY":
                # Check Stop-Loss hit
                if pos_sl and low_price <= pos_sl:
                    closed = True
                    exit_p = pos_sl
                    reason = "STOP_LOSS"
                # Check Take-Profit hit
                elif pos_tp and high_price >= pos_tp:
                    closed = True
                    exit_p = pos_tp
                    reason = "TAKE_PROFIT"
                # Check opposite strong sell signal
                elif "SHORT" in signal:
                    closed = True
                    exit_p = close_price
                    reason = "SIGNAL_REVERSAL"

                if closed:
                    gross_pnl = (exit_p - entry_p) * qty
                    fees = (entry_p * qty * cost_rate) + (exit_p * qty * cost_rate)
                    net_pnl = gross_pnl - fees
                    pnl_pct = (net_pnl / (entry_p * qty)) * 100.0

                    # If closed via stop-loss above entry price, mark as PROFIT_LOCK
                    if reason == "STOP_LOSS" and exit_p > entry_p:
                        reason = "PROFIT_LOCK"

                    cursor.execute(
                        """UPDATE trades SET
                           status = 'CLOSED', exit_time = ?, exit_price = ?,
                           exit_reason = ?, pnl_usd = ?, pnl_pct = ?, fees_paid = ?
                           WHERE id = ?;""",
                        (time_str, exit_p, reason, net_pnl, pnl_pct, fees, pos_id),
                    )
                    action_log.append(f"Closed LONG #{pos_id} via {reason} at ${exit_p:,.2f} | PnL: ${net_pnl:+,.2f} ({pnl_pct:+.2f}%)")
                elif pos_tp and pos_tp > entry_p:
                    # Dynamic Profit Lock: if trade moved >= 65% toward TP, elevate SL to lock profit
                    target_gain = pos_tp - entry_p
                    if high_price >= (entry_p + 0.65 * target_gain):
                        locked_sl = entry_p + (0.25 * target_gain)
                        if pos_sl is None or locked_sl > pos_sl:
                            cursor.execute("UPDATE trades SET stop_loss = ? WHERE id = ?;", (locked_sl, pos_id))
                            pos_sl = locked_sl
                            action_log.append(f"Profit-Lock Activated on LONG #{pos_id}: Stop-Loss raised to ${locked_sl:,.2f} (+{((locked_sl/entry_p)-1)*100:+.2f}%)")

            elif pos_action == "SELL":
                # Check Stop-Loss hit
                if pos_sl and high_price >= pos_sl:
                    closed = True
                    exit_p = pos_sl
                    reason = "STOP_LOSS"
                # Check Take-Profit hit
                elif pos_tp and low_price <= pos_tp:
                    closed = True
                    exit_p = pos_tp
                    reason = "TAKE_PROFIT"
                # Check opposite buy signal
                elif "LONG" in signal:
                    closed = True
                    exit_p = close_price
                    reason = "SIGNAL_REVERSAL"

                if closed:
                    gross_pnl = (entry_p - exit_p) * qty
                    fees = (entry_p * qty * cost_rate) + (exit_p * qty * cost_rate)
                    net_pnl = gross_pnl - fees
                    pnl_pct = (net_pnl / (entry_p * qty)) * 100.0

                    # If closed via stop-loss below entry price, mark as PROFIT_LOCK
                    if reason == "STOP_LOSS" and exit_p < entry_p:
                        reason = "PROFIT_LOCK"

                    cursor.execute(
                        """UPDATE trades SET
                           status = 'CLOSED', exit_time = ?, exit_price = ?,
                           exit_reason = ?, pnl_usd = ?, pnl_pct = ?, fees_paid = ?
                           WHERE id = ?;""",
                        (time_str, exit_p, reason, net_pnl, pnl_pct, fees, pos_id),
                    )
                    action_log.append(f"Closed SHORT #{pos_id} via {reason} at ${exit_p:,.2f} | PnL: ${net_pnl:+,.2f} ({pnl_pct:+.2f}%)")
                elif pos_tp and entry_p > pos_tp:
                    # Dynamic Profit Lock: if trade moved >= 65% toward TP, lower SL to lock profit
                    target_gain = entry_p - pos_tp
                    if low_price <= (entry_p - 0.65 * target_gain):
                        locked_sl = entry_p - (0.25 * target_gain)
                        if pos_sl is None or locked_sl < pos_sl:
                            cursor.execute("UPDATE trades SET stop_loss = ? WHERE id = ?;", (locked_sl, pos_id))
                            pos_sl = locked_sl
                            action_log.append(f"Profit-Lock Activated on SHORT #{pos_id}: Stop-Loss lowered to ${locked_sl:,.2f} ({((locked_sl/entry_p)-1)*100:+.2f}%)")

        # 3. Check for new entry if no position is open
        cursor.execute("SELECT COUNT(*) FROM trades WHERE symbol = ? AND status = 'OPEN';", (symbol,))
        has_open = cursor.fetchone()[0] > 0

        if not has_open:
            if "BUY" in signal or "LONG" in signal:
                risk_amt = starting_capital * risk_per_trade
                sl_distance = abs(close_price - sl_price) if sl_price else (close_price * 0.02)
                qty = risk_amt / (sl_distance + 1e-8)
                max_pos_usd = starting_capital * 0.50
                if (qty * close_price) > max_pos_usd:
                    qty = max_pos_usd / close_price

                cursor.execute(
                    """INSERT INTO trades (symbol, action, status, entry_time, entry_price, quantity, stop_loss, take_profit, fees_paid)
                       VALUES (?, 'BUY', 'OPEN', ?, ?, ?, ?, ?, ?);""",
                    (symbol, time_str, close_price, qty, sl_price, tp_price, close_price * qty * cost_rate),
                )
                cursor.execute("UPDATE signals_log SET executed = 1 WHERE id = ?;", (signal_id,))
                action_log.append(f"Opened virtual LONG at ${close_price:,.2f} (Qty: {qty:.4f}, SL: ${sl_price:,.2f}, TP: ${tp_price:,.2f})")

            elif "SELL" in signal or "SHORT" in signal:
                risk_amt = starting_capital * risk_per_trade
                sl_distance = abs(sl_price - close_price) if sl_price else (close_price * 0.02)
                qty = risk_amt / (sl_distance + 1e-8)
                max_pos_usd = starting_capital * 0.50
                if (qty * close_price) > max_pos_usd:
                    qty = max_pos_usd / close_price

                cursor.execute(
                    """INSERT INTO trades (symbol, action, status, entry_time, entry_price, quantity, stop_loss, take_profit, fees_paid)
                       VALUES (?, 'SELL', 'OPEN', ?, ?, ?, ?, ?, ?);""",
                    (symbol, time_str, close_price, qty, sl_price, tp_price, close_price * qty * cost_rate),
                )
                cursor.execute("UPDATE signals_log SET executed = 1 WHERE id = ?;", (signal_id,))
                action_log.append(f"Opened virtual SHORT at ${close_price:,.2f} (Qty: {qty:.4f}, SL: ${sl_price:,.2f}, TP: ${tp_price:,.2f})")

        conn.commit()

    return {
        "timestamp": time_str,
        "symbol": symbol,
        "actions": action_log,
    }


def get_paper_trading_summary(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Retrieve full performance metrics and history from the SQLite paper trading database."""
    if not Path(db_path).exists():
        return {"error": "Paper trading database not yet initialized."}

    with get_db_connection(db_path) as conn:
        trades_df = pd.read_sql_query("SELECT * FROM trades;", conn)

    if trades_df.empty:
        return {
            "total_trades": 0,
            "open_positions": 0,
            "realized_pnl_usd": 0.0,
            "win_rate_pct": 0.0,
        }

    closed = trades_df[trades_df["status"] == "CLOSED"]
    open_pos = trades_df[trades_df["status"] == "OPEN"]

    total_closed = len(closed)
    winning_trades = len(closed[closed["pnl_usd"] > 0])
    losing_trades = len(closed[closed["pnl_usd"] < 0])
    win_rate = (winning_trades / total_closed * 100.0) if total_closed > 0 else 0.0
    realized_pnl = float(closed["pnl_usd"].sum()) if total_closed > 0 else 0.0
    total_fees = float(trades_df["fees_paid"].sum())

    return {
        "total_trades": len(trades_df),
        "closed_trades": total_closed,
        "open_positions": len(open_pos),
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate_pct": round(win_rate, 2),
        "realized_pnl_usd": round(realized_pnl, 2),
        "total_fees_paid": round(total_fees, 2),
        "recent_trades": closed.tail(5).to_dict(orient="records"),
    }
