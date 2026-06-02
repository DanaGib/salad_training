#!/usr/bin/env bash
# Evaluate a trained SALAD checkpoint on Pittsburgh 30k.
#
# Usage:
#   bash eval_pitts30k.sh <checkpoint_path> <pitts_path>
#
# Example:
#   bash eval_pitts30k.sh \
#     logs/dinov2_vitb14_(03)_R1[0.9210]_R5[0.9610].ckpt \
#     /data/Pittsburgh
#
# The checkpoint path printed at the end of training (best R@1 filename) is
# the value to pass as <checkpoint_path>.
#
# Results (R@1, R@5, R@10, R@15, R@20, R@25) are printed to stdout and
# teed to logs/eval/<timestamp>.log.

set -euo pipefail

# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------
if [ -z "${1:-}" ] || [ -z "${2:-}" ]; then
    echo "Usage: $0 <checkpoint_path> <pitts_path>"
    echo ""
    echo "  checkpoint_path  path to the .ckpt file produced by training"
    echo "  pitts_path       absolute path to the Pittsburgh dataset root"
    echo "                   (must contain ref/ and query/ subdirectories)"
    exit 1
fi

CKPT_PATH="$1"
export PITTS_PATH="$2"

# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$REPO_ROOT/salad_env/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "Error: virtual env not found at $REPO_ROOT/salad_env"
    exit 1
fi

if [ ! -f "$CKPT_PATH" ]; then
    echo "Error: checkpoint not found: $CKPT_PATH"
    exit 1
fi

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
LOG_DIR="$REPO_ROOT/logs/eval"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_pitts30k.log"

echo "========================================"
echo "SALAD Evaluation — Pittsburgh 30k"
echo "========================================"
echo "Start time  : $(date)"
echo "Checkpoint  : $CKPT_PATH"
echo "Pittsburgh  : $PITTS_PATH"
echo "Log file    : $LOG_FILE"
echo "========================================"

# --------------------------------------------------------------------------
# Evaluate
# --------------------------------------------------------------------------
cd "$REPO_ROOT"
"$PYTHON" eval.py \
    --ckpt_path "$CKPT_PATH" \
    --val_datasets pitts30k_test \
    --batch_size 256 \
    --image_size 224 224 \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================"
if [ "$EXIT_CODE" -eq 0 ]; then
    echo "Evaluation finished at $(date)"
else
    echo "Evaluation FAILED with exit code $EXIT_CODE at $(date)"
fi
echo "Results saved to: $LOG_FILE"
echo "========================================"

exit "$EXIT_CODE"
