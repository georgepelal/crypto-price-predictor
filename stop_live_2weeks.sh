#!/bin/bash
# ==============================================================================
# Stop 2-Week Autonomous Multi-Asset Paper Trading Daemon
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/data/trader.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "[!] No trader.pid file found. The bot does not appear to be running."
    exit 0
fi

PID=$(cat "$PID_FILE")

if ps -p "$PID" > /dev/null 2>&1; then
    echo "[*] Stopping trading bot (PID: $PID)..."
    kill "$PID"
    sleep 2
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "[*] Forcing process termination..."
        kill -9 "$PID"
    fi
    rm -f "$PID_FILE"
    echo "[+] Trading bot successfully stopped."
else
    echo "[!] Process $PID was not running. Removed stale PID file."
    rm -f "$PID_FILE"
fi
