#!/usr/bin/env bash
# Train 9 linear-projection experiments and evaluate all on pitts30k_test
# and amstertime. All recall results accumulate in one CSV for easy comparison.
#
# Runs (in order):
#   1.  baseline                         — standard SALAD, no depth branch
#   2.  linear_depth_mse_none            — linear MLP, MSE,    no norm
#   3.  linear_depth_mse_after           — linear MLP, MSE,    norm after
#   4.  linear_depth_cosine_none         — linear MLP, cosine, no norm
#   5.  linear_depth_cosine_after        — linear MLP, cosine, norm after
#   6.  linear_depth_mse_none_alpha100   — as 2, alpha=100
#   7.  linear_depth_mse_after_alpha100  — as 3, alpha=100
#   8.  linear_depth_mse_none_alpha500   — as 2, alpha=500
#   9.  linear_depth_mse_after_alpha500  — as 3, alpha=500
#
# Output CSV: logs/eval/linear_trials.csv  (18 rows: 9 runs x 2 datasets)
#
# Usage:
#   bash train_linear_trials.sh <gsvcities_path>

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

CSV_NAME="linear_trials"

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
    "model.mlp.type=linear" \
    "model.mlp.normalization=none" \
    "loss.alignment_loss_type=mse" \
    "wandb.run_name=linear_depth_mse_none" \
    || OVERALL=$?

run_experiment \
    "model.type=salad_joint_depth" \
    "model.mlp.type=linear" \
    "model.mlp.normalization=after" \
    "loss.alignment_loss_type=mse" \
    "wandb.run_name=linear_depth_mse_after" \
    || OVERALL=$?

run_experiment \
    "model.type=salad_joint_depth" \
    "model.mlp.type=linear" \
    "model.mlp.normalization=none" \
    "loss.alignment_loss_type=cosine" \
    "wandb.run_name=linear_depth_cosine_none" \
    || OVERALL=$?

run_experiment \
    "model.type=salad_joint_depth" \
    "model.mlp.type=linear" \
    "model.mlp.normalization=after" \
    "loss.alignment_loss_type=cosine" \
    "wandb.run_name=linear_depth_cosine_after" \
    || OVERALL=$?

run_experiment \
    "model.type=salad_joint_depth" \
    "model.mlp.type=linear" \
    "model.mlp.normalization=none" \
    "loss.alignment_loss_type=mse" \
    "loss.alpha_local=100" \
    "wandb.run_name=linear_depth_mse_none_alpha100" \
    || OVERALL=$?

run_experiment \
    "model.type=salad_joint_depth" \
    "model.mlp.type=linear" \
    "model.mlp.normalization=after" \
    "loss.alignment_loss_type=mse" \
    "loss.alpha_local=100" \
    "wandb.run_name=linear_depth_mse_after_alpha100" \
    || OVERALL=$?

run_experiment \
    "model.type=salad_joint_depth" \
    "model.mlp.type=linear" \
    "model.mlp.normalization=none" \
    "loss.alignment_loss_type=mse" \
    "loss.alpha_local=500" \
    "wandb.run_name=linear_depth_mse_none_alpha500" \
    || OVERALL=$?

run_experiment \
    "model.type=salad_joint_depth" \
    "model.mlp.type=linear" \
    "model.mlp.normalization=after" \
    "loss.alignment_loss_type=mse" \
    "loss.alpha_local=500" \
    "wandb.run_name=linear_depth_mse_after_alpha500" \
    || OVERALL=$?

echo "========================================"
echo "Suite complete. Combined results: $REPO_ROOT/logs/eval/${CSV_NAME}.csv"
echo "========================================"

exit "$OVERALL"
