#!/usr/bin/env bash
# Standalone re-evaluation script for train_trials.sh Blocks A/B/C runs.
#
# Loops over every pred_global_*, global_depth_*, and global_local_* checkpoint
# directory, evaluates last.ckpt on pitts30k_test, amstertime, and MSLS_Test,
# and appends recall results to a single CSV at logs/eval/trials.csv.
# MSLS_Test has no ground truth: eval.py writes a .preds.txt submission file
# next to each checkpoint instead of adding rows to the CSV.
#
# Usage:
#   bash eval_trials.sh
#
# No arguments are required. The script discovers runs automatically from
# logs/checkpoints/ inside the repository root.
# Override MSLS_PATH before calling to use a different Mapillary image root.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$REPO_ROOT/env/bin/python"
CKPT_BASE="$REPO_ROOT/logs/checkpoints"
CSV_OUT="$REPO_ROOT/logs/eval/trials.csv"
LOG_DIR="$REPO_ROOT/logs/runs"

export MSLS_PATH="${MSLS_PATH:-/home/shared/datasets/msls_challenge/}"

if [ ! -f "$PYTHON" ]; then
    echo "Error: virtualenv not found at $REPO_ROOT/env"
    exit 1
fi

mkdir -p "$(dirname "$CSV_OUT")" "$LOG_DIR"

OVERALL=0

for ckpt_dir in \
    "$CKPT_BASE"/pred_global_* \
    "$CKPT_BASE"/global_depth_* \
    "$CKPT_BASE"/global_local_*; do

    [ -d "$ckpt_dir" ] || continue
    ckpt="$ckpt_dir/last.ckpt"
    [ -f "$ckpt" ] || continue

    dir_label=$(basename "$ckpt_dir")
    # Strip trailing _YYYYMMDD_HHMMSS timestamp to recover the original run name.
    run_name=$(echo "$dir_label" | sed 's/_[0-9]\{8\}_[0-9]\{6\}$//')

    echo "========================================"
    echo "Evaluating : $run_name"
    echo "Checkpoint : $ckpt"
    echo "Time       : $(date)"
    echo "========================================"

    "$PYTHON" eval.py \
        --ckpt_path "$ckpt" \
        --val_datasets pitts30k_test amstertime MSLS_Test \
        --image_size 322 322 \
        --batch_size 256 \
        --run_name "$run_name" \
        --csv_path "$CSV_OUT" \
        2>&1 | tee -a "$LOG_DIR/${dir_label}_eval.log" \
        || OVERALL=$?

    echo "--- Done: $run_name at $(date) ---"
    echo ""
done

echo "========================================"
echo "All evaluations complete."
echo "Results written to: $CSV_OUT"
echo "Overall exit code : $OVERALL"
echo "========================================"
exit "$OVERALL"
