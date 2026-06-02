#!/usr/bin/env bash
# Train SALAD + JEPA distillation on GSV-Cities overnight.
#
# Usage:
#   bash train_overnight.sh <gsvcities_path>
#
# Example:
#   bash train_overnight.sh /data/GSVCities
#
# The script activates the salad virtual environment, exports the dataset
# path, runs training, and tees all output to a timestamped log file under
# logs/runs/ so results are preserved even if the tmux session is closed.
#
# Checkpoints are saved by PyTorch Lightning to ./logs/ (best 3 by R@1 on
# pitts30k_val plus the last epoch).

set -euo pipefail

# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------
if [ -z "${1:-}" ]; then
    echo "Usage: $0 <gsvcities_path>"
    echo "  gsvcities_path  absolute path to the GSVCities dataset root"
    exit 1
fi

export GSVCITIES_PATH="$1"

# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$REPO_ROOT/salad_env/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "Error: virtual env not found at $REPO_ROOT/salad_env"
    echo "Create it with:  conda env create -f environment.yml"
    exit 1
fi

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
LOG_DIR="$REPO_ROOT/logs/runs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train.log"

echo "========================================"
echo "SALAD + JEPA Distillation Training"
echo "========================================"
echo "Start time  : $(date)"
echo "GSVCities   : $GSVCITIES_PATH"
echo "Checkpoint  : $REPO_ROOT/logs/"
echo "Log file    : $LOG_FILE"
echo "========================================"

# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------
cd "$REPO_ROOT"
"$PYTHON" main.py 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================"
if [ "$EXIT_CODE" -eq 0 ]; then
    echo "Training finished successfully at $(date)"
    echo "Checkpoints saved to: $REPO_ROOT/logs/"
else
    echo "Training FAILED with exit code $EXIT_CODE at $(date)"
fi
echo "Log saved to: $LOG_FILE"
echo "========================================"

exit "$EXIT_CODE"
