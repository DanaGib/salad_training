#!/usr/bin/env bash
# Run two joint-depth MSE experiments sweeping loss.alpha (100 and 500).
# Normalization: after_mlp. Auto-evals last.ckpt after each run.
#
# Usage:
#   bash train_alpha_sweep.sh <gsvcities_path>

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$REPO_ROOT/env/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "Error: virtualenv not found at $REPO_ROOT/env"
    exit 1
fi

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <gsvcities_path>"
    exit 1
fi

export GSVCITIES_PATH="$1"
export AMSTERTIME_PATH="${AMSTERTIME_PATH:-/home/eng/giborda/delavpr/datasets/amstertime/}"

mkdir -p "$REPO_ROOT/logs/runs"
cd "$REPO_ROOT"

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
        echo "--- WARNING: last.ckpt not found, skipping eval ---" | tee -a "$log"
        return 0
    fi

    echo "--- Evaluating $label on pitts30k_test ---" | tee -a "$log"
    "$PYTHON" eval.py --ckpt_path "$ckpt_dir/last.ckpt" \
        --val_datasets pitts30k_test --image_size 322 322 --batch_size 256 \
        2>&1 | tee -a "$log"

    echo "--- Evaluating $label on amstertime ---" | tee -a "$log"
    "$PYTHON" eval.py --ckpt_path "$ckpt_dir/last.ckpt" \
        --val_datasets amstertime --image_size 322 322 --batch_size 256 \
        2>&1 | tee -a "$log"

    echo "--- All done: $label at $(date) ---" | tee -a "$log"
}

OVERALL=0

run_experiment \
    "model.type=salad_joint_depth" \
    "model.normalization.stage=after_mlp" \
    "loss.alignment_loss_type=mse" \
    "loss.alpha=100" \
    "wandb.run_name=joint_depth_mse_after_mlp_alpha100" \
    || OVERALL=$?

run_experiment \
    "model.type=salad_joint_depth" \
    "model.normalization.stage=after_mlp" \
    "loss.alignment_loss_type=mse" \
    "loss.alpha=500" \
    "wandb.run_name=joint_depth_mse_after_mlp_alpha500" \
    || OVERALL=$?

exit "$OVERALL"