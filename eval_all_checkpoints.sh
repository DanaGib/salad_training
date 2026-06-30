#!/usr/bin/env bash
# Run pitts30k_test + amstertime eval on every epoch checkpoint and print a
# summary table so you can pick the best epoch.
#
# Usage:
#   bash eval_all_checkpoints.sh <pitts_path> <amstertime_path>
#
# Example:
#   bash eval_all_checkpoints.sh \
#     /home/eng/giborda/delavpr/datasets/pitts30k \
#     /home/eng/giborda/delavpr/datasets/amstertime
#
# Results are teed to logs/eval/<timestamp>_all_checkpoints.log
# and a summary table is printed at the end.

set -euo pipefail

# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------
if [ -z "${1:-}" ] || [ -z "${2:-}" ]; then
    echo "Usage: $0 <pitts_path> <amstertime_path>"
    exit 1
fi

export PITTS_PATH="$1"
export AMSTERTIME_PATH="$2"

# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$REPO_ROOT/salad_env/bin/python"
CKPT_DIR="$REPO_ROOT/logs/lightning_logs/version_0/checkpoints"
LOG_DIR="$REPO_ROOT/logs/eval"
mkdir -p "$LOG_DIR"
SUMMARY_LOG="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_all_checkpoints.log"

if [ ! -f "$PYTHON" ]; then
    echo "Error: virtual env not found at $REPO_ROOT/salad_env"
    exit 1
fi

# --------------------------------------------------------------------------
# Collect checkpoints (skip last.ckpt — it duplicates the final epoch)
# --------------------------------------------------------------------------
mapfile -t CKPTS < <(ls "$CKPT_DIR"/dinov2_vitb14_*.ckpt 2>/dev/null | sort)

if [ ${#CKPTS[@]} -eq 0 ]; then
    echo "Error: no epoch checkpoints found in $CKPT_DIR"
    exit 1
fi

echo "Found ${#CKPTS[@]} checkpoint(s):"
for c in "${CKPTS[@]}"; do echo "  $c"; done
echo ""

# --------------------------------------------------------------------------
# Eval loop — one log file per checkpoint per dataset
# --------------------------------------------------------------------------
declare -A PITTS_R1 PITTS_R5 PITTS_R10
declare -A AMT_R1   AMT_R5   AMT_R10

for CKPT in "${CKPTS[@]}"; do
    BASENAME="$(basename "$CKPT" .ckpt)"
    echo "========================================"
    echo "Checkpoint: $BASENAME"
    echo "========================================"

    # --- Pitts30k test ---
    P_LOG="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_${BASENAME}_pitts30k.log"
    echo "  Running pitts30k_test ..."
    cd "$REPO_ROOT"
    "$PYTHON" eval.py \
        --ckpt_path "$CKPT" \
        --val_datasets pitts30k_test \
        --batch_size 256 \
        --image_size 224 224 \
        2>&1 | tee "$P_LOG"

    # Parse the machine-parseable "RECALLS <dataset> R@1=XX R@5=XX R@10=XX" line
    RECALLS_LINE="$(grep -m1 '^RECALLS pitts30k_test' "$P_LOG" || echo '')"
    PITTS_R1["$BASENAME"]="$(echo "$RECALLS_LINE" | grep -oP 'R@1=\K[\d.]+' || echo 'N/A')"
    PITTS_R5["$BASENAME"]="$(echo "$RECALLS_LINE" | grep -oP 'R@5=\K[\d.]+' || echo 'N/A')"
    PITTS_R10["$BASENAME"]="$(echo "$RECALLS_LINE" | grep -oP 'R@10=\K[\d.]+' || echo 'N/A')"

    # --- AmsterTime ---
    A_LOG="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_${BASENAME}_amstertime.log"
    echo "  Running amstertime ..."
    "$PYTHON" eval.py \
        --ckpt_path "$CKPT" \
        --val_datasets amstertime \
        --batch_size 256 \
        --image_size 224 224 \
        2>&1 | tee "$A_LOG"

    RECALLS_LINE="$(grep -m1 '^RECALLS amstertime' "$A_LOG" || echo '')"
    AMT_R1["$BASENAME"]="$(echo "$RECALLS_LINE"  | grep -oP 'R@1=\K[\d.]+' || echo 'N/A')"
    AMT_R5["$BASENAME"]="$(echo "$RECALLS_LINE"  | grep -oP 'R@5=\K[\d.]+' || echo 'N/A')"
    AMT_R10["$BASENAME"]="$(echo "$RECALLS_LINE" | grep -oP 'R@10=\K[\d.]+' || echo 'N/A')"

    echo ""
done

# --------------------------------------------------------------------------
# Summary table
# --------------------------------------------------------------------------
SUMMARY="
==============================================================================
 Checkpoint Comparison Summary
==============================================================================
 Dataset         Checkpoint                          R@1      R@5      R@10
------------------------------------------------------------------------------"

for CKPT in "${CKPTS[@]}"; do
    BASENAME="$(basename "$CKPT" .ckpt)"
    SUMMARY+="
 pitts30k_test   $BASENAME   ${PITTS_R1[$BASENAME]}   ${PITTS_R5[$BASENAME]}   ${PITTS_R10[$BASENAME]}
 amstertime      $BASENAME   ${AMT_R1[$BASENAME]}   ${AMT_R5[$BASENAME]}   ${AMT_R10[$BASENAME]}
------------------------------------------------------------------------------"
done

SUMMARY+="
=============================================================================="

echo "$SUMMARY"
echo "$SUMMARY" | tee -a "$SUMMARY_LOG"

echo ""
echo "Full per-checkpoint logs saved to: $LOG_DIR"
echo "Summary saved to: $SUMMARY_LOG"
