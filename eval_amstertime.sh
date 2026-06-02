#!/usr/bin/env bash
# Evaluate a trained SALAD checkpoint on AmsterTime.
#
# Usage:
#   bash eval_amstertime.sh <checkpoint_path> <amstertime_path>
#
# Example:
#   bash eval_amstertime.sh \
#     logs/lightning_logs/version_0/checkpoints/dinov2_vitb14_(epochXX).ckpt \
#     /home/eng/giborda/delavpr/datasets/amstertime
#
# <amstertime_path> must contain a test/ subdirectory with:
#   database/   — modern street-view images
#   queries/    — historical images (same filenames as database)
# Ground truth is 1:1 (query i matches database i).
#
# Results (R@1, R@5, R@10, R@15, R@20, R@25) are printed to stdout and
# teed to logs/eval/<timestamp>_amstertime.log.

set -euo pipefail

# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------
if [ -z "${1:-}" ] || [ -z "${2:-}" ]; then
    echo "Usage: $0 <checkpoint_path> <amstertime_path>"
    echo ""
    echo "  checkpoint_path   path to the .ckpt file produced by training"
    echo "  amstertime_path   absolute path to the AmsterTime dataset root"
    echo "                    (must contain test/ with database/ and queries/)"
    exit 1
fi

CKPT_PATH="$1"
export AMSTERTIME_PATH="$2"

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
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_amstertime.log"

echo "========================================"
echo "SALAD Evaluation — AmsterTime"
echo "========================================"
echo "Start time  : $(date)"
echo "Checkpoint  : $CKPT_PATH"
echo "AmsterTime  : $AMSTERTIME_PATH"
echo "Log file    : $LOG_FILE"
echo "========================================"

# --------------------------------------------------------------------------
# Evaluate
# --------------------------------------------------------------------------
cd "$REPO_ROOT"
"$PYTHON" eval.py \
    --ckpt_path "$CKPT_PATH" \
    --val_datasets amstertime \
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
