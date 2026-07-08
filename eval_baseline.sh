#!/usr/bin/env bash
# eval_baseline.sh — Evaluate the 3 baseline training runs on 4 benchmarks.
#
# Evaluates checkpoints produced by train_baseline.sh:
#   baseline_bs80_ep6_es    (bs=80, epochs=6, early stopping ON)
#   baseline_bs80_ep6_no_es (bs=80, epochs=6, early stopping OFF)
#   baseline_bs60_ep4       (bs=60, epochs=4, fixed)
#
# Datasets: Nordland, MSLS, amstertime, pitts30k_test
# Results:  logs/eval/baseline_eval.csv  (all runs appended to one file)
# Logs:     logs/runs/<label>_eval.log
#
# Usage: bash eval_baseline.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$REPO_ROOT/env/bin/python"
CKPT_BASE="$REPO_ROOT/logs/checkpoints"
CSV_OUT="$REPO_ROOT/logs/eval/baseline_eval.csv"
LOG_DIR="$REPO_ROOT/logs/runs"
DATASETS="Nordland MSLS amstertime pitts30k_test"

export MSLS_PATH="${MSLS_PATH:-/home/shared/datasets/msls_challenge/}"
export AMSTERTIME_PATH="${AMSTERTIME_PATH:-/home/eng/giborda/delavpr/datasets/amstertime/}"

if [ ! -f "$PYTHON" ]; then
    echo "Error: virtualenv not found at $REPO_ROOT/env"
    exit 1
fi

mkdir -p "$(dirname "$CSV_OUT")" "$LOG_DIR"
OVERALL=0

# ---------------------------------------------------------------------------
# eval_run <run_name> <extra_params_json>
#   Discovers the most recent checkpoint dir matching <run_name>_*, loads
#   last.ckpt, and evaluates on all 4 datasets. Skips if no checkpoint found.
# ---------------------------------------------------------------------------
eval_run() {
    local run_name="$1" extra_params="$2"
    local ckpt_dir
    ckpt_dir=$(ls -td "$CKPT_BASE/${run_name}_"* 2>/dev/null | head -1 || true)
    if [ -z "$ckpt_dir" ] || [ ! -f "$ckpt_dir/last.ckpt" ]; then
        echo "SKIP (checkpoint not found): $run_name"
        return 0
    fi
    echo ""
    echo "======== Eval: $run_name | $(date) ========"
    # shellcheck disable=SC2086
    "$PYTHON" eval.py \
        --ckpt_path "$ckpt_dir/last.ckpt" \
        --val_datasets $DATASETS \
        --image_size 322 322 \
        --batch_size 256 \
        --run_name "$run_name" \
        --csv_path "$CSV_OUT" \
        --extra_params "$extra_params" \
        2>&1 | tee -a "$LOG_DIR/${run_name}_eval.log" || OVERALL=$?
    echo "--- Done: $run_name | $(date) ---"
}

# ---------------------------------------------------------------------------
# Run 1: bs=80, max_epochs=6, early stopping ON (patience=2)
# ---------------------------------------------------------------------------
eval_run baseline_bs80_ep6_es \
    '{"model_type":"salad_baseline","batch_size":80,"max_epochs":6,"early_stop_patience":2}'

# ---------------------------------------------------------------------------
# Run 2: bs=80, max_epochs=6, early stopping OFF
# ---------------------------------------------------------------------------
eval_run baseline_bs80_ep6_no_es \
    '{"model_type":"salad_baseline","batch_size":80,"max_epochs":6,"early_stop_patience":0}'

# ---------------------------------------------------------------------------
# Run 3: bs=60, max_epochs=4, fixed epochs
# ---------------------------------------------------------------------------
eval_run baseline_bs60_ep4 \
    '{"model_type":"salad_baseline","batch_size":60,"max_epochs":4,"early_stop_patience":0}'

echo ""
echo "=== eval_baseline.sh complete | $(date) | Results: $CSV_OUT | Exit: $OVERALL ==="
exit "$OVERALL"
