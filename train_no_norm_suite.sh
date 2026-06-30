#!/usr/bin/env bash
# Train joint-depth experiments with no MLP normalisation, plus a baseline.
# Results from all three runs are written to a single CSV for easy comparison.
#
# Runs (in order):
#   1. salad_baseline            — standard SALAD, no depth branch
#   2. salad_joint_depth         — MSE loss,    no normalisation
#   3. salad_joint_depth         — cosine loss, no normalisation
#
# After each training run, last.ckpt is evaluated on pitts30k_test and
# amstertime. All recall rows accumulate in:
#   logs/eval/no_norm_suite.csv
#
# Usage:
#   bash train_no_norm_suite.sh <gsvcities_path>

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$REPO_ROOT/env/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "Error: virtualenv not found at $REPO_ROOT/env"
    echo "Create it with: conda env create -f environment.yml"
    exit 1
fi

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <gsvcities_path>"
    exit 1
fi

export GSVCITIES_PATH="$1"

mkdir -p "$REPO_ROOT/logs/runs"
cd "$REPO_ROOT"

CSV_NAME="no_norm_suite"

run_experiment() {
    local label="" log ts ckpt_dir rc=0
    for arg in "$@"; do
        [[ "$arg" == wandb.run_name=* ]] && label="${arg#wandb.run_name=}" && break
    done
    [ -z "$label" ] && label="run"
    ts=$(date +%Y%m%d_%H%M%S)
    log="$REPO_ROOT/logs/runs/${label}_${ts}.log"

    echo "========================================"
    echo "Starting : $label"
    echo "Log      : $log"
    echo "Time     : $(date)"
    echo "========================================"

    set +e
    "$PYTHON" main.py "$@" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    set -e

    if [ "$rc" -ne 0 ]; then
        echo "--- FAILED: $label (exit $rc) at $(date) ---" | tee -a "$log"
        return "$rc"
    fi

    echo "--- Training complete: $label at $(date) ---" | tee -a "$log"

    ckpt_dir=$(ls -td "$REPO_ROOT/logs/checkpoints/${label}_"* 2>/dev/null | head -1 || true)

    if [ -z "$ckpt_dir" ] || [ ! -f "$ckpt_dir/last.ckpt" ]; then
        echo "--- WARNING: last.ckpt not found under $ckpt_dir, skipping eval ---" | tee -a "$log"
        return 0
    fi

    echo "--- Evaluating $label on pitts30k_test and amstertime ---" | tee -a "$log"
    "$PYTHON" eval.py \
        --ckpt_path "$ckpt_dir/last.ckpt" \
        --val_datasets pitts30k_test amstertime \
        --image_size 322 322 --batch_size 256 \
        --run_name "$label" \
        --csv_name "$CSV_NAME" \
        2>&1 | tee -a "$log"

    echo "--- All done: $label at $(date) ---" | tee -a "$log"
}

OVERALL=0

run_experiment \
    "model.type=salad_baseline" \
    "wandb.run_name=baseline" \
    || OVERALL=$?

run_experiment \
    "model.type=salad_joint_depth" \
    "model.mlp.normalization=none" \
    "loss.alignment_loss_type=mse" \
    "wandb.run_name=joint_depth_mse_none" \
    || OVERALL=$?

run_experiment \
    "model.type=salad_joint_depth" \
    "model.mlp.normalization=none" \
    "loss.alignment_loss_type=cosine" \
    "wandb.run_name=joint_depth_cosine_none" \
    || OVERALL=$?

echo "========================================"
echo "Suite complete. Combined results: $REPO_ROOT/logs/eval/${CSV_NAME}.csv"
echo "========================================"

exit "$OVERALL"
