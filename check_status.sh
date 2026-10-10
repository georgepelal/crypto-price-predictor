#!/bin/bash
# ==============================================================================
# Quick Status & Portfolio Health Check for 2-Week Paper Trader
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/data/trader.pid"
LOG_FILE="$SCRIPT_DIR/logs/live_trader.log"
PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python3"

echo "=================================================================="
echo " TRADING BOT STATUS CHECK"
echo " Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=================================================================="

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo " [✓] BOT STATUS: ACTIVE & RUNNING (PID: $PID)"
    else
        echo " [!] BOT STATUS: NOT RUNNING (Dead PID $PID)"
    fi
else
    echo " [!] BOT STATUS: INACTIVE (No PID file)"
fi

echo "------------------------------------------------------------------"
echo " PAPER LEDGER"
cd "$SCRIPT_DIR" && "$PYTHON_BIN" paper_trader.py --status

echo "------------------------------------------------------------------"
echo " LAST 8 LOG LINES:"
if [ -f "$LOG_FILE" ]; then
    tail -n 8 "$LOG_FILE"
else
    echo " (No log file created yet)"
fi
echo "=================================================================="
