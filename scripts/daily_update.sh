#!/bin/bash
# E1: Daily stock predictor update script
# Run this via cron or launchd for automated daily updates

# Exit on error
set -e

# Script directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Log file
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/daily_update_$(date +%Y%m%d).log"

# Python path (modify if using a virtual environment)
PYTHON_PATH="${PYTHON_PATH:-python3}"

# Redirect output to log file
exec >> "$LOG_FILE" 2>&1

echo "=========================================="
echo "Stock Top-10 Daily Update"
echo "Started: $(date)"
echo "=========================================="

cd "$PROJECT_DIR"

# Run the daily update
echo ""
echo "[1/3] Running daily data update..."
$PYTHON_PATH app/update_daily.py

# Optional: Run backtest
if [ "$RUN_BACKTEST" = "true" ]; then
    echo ""
    echo "[2/3] Running backtest..."
    $PYTHON_PATH app/update_daily.py --skip-data --backtest
fi

# Optional: Generate report
echo ""
echo "[3/3] Update complete!"
echo ""
echo "Latest predictions saved to: $PROJECT_DIR/outputs/top10_latest.parquet"

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
