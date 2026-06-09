#!/usr/bin/env bash
# Train one or more SALAD models then evaluate each on pitts30k_test and amstertime.
#
# Usage:
#   bash train_overnight.sh <gsvcities_path> --suite
#   bash train_overnight.sh <gsvcities_path> [salad_baseline|salad_joint_depth ...] [overrides]
#
# --suite runs three experiments back-to-back:
#   1. salad_baseline
#   2. salad_joint_depth  —  mse loss,    after_mlp normalisation
#   3. salad_joint_depth  —  cosine loss, after_mlp normalisation
#
# After each training run completes, last.ckpt is automatically evaluated on
# pitts30k_test and then amstertime. Results are appended to the same log file.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$REPO_ROOT/env/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "Error: virtualenv not found at $REPO_ROOT/env"
    echo "Create it with: conda env create -f environment.yml"
    exit 1
fi

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <gsvcities_path> --suite"
    echo "       $0 <gsvcities_path> [salad_baseline|salad_joint_depth ...] [key=value overrides]"
    exit 1
fi

export GSVCITIES_PATH="$1"
export AMSTERTIME_PATH="${AMSTERTIME_PATH:-/home/eng/giborda/delavpr/datasets/amstertime/}"
shift

mkdir -p "$REPO_ROOT/logs/runs"
cd "$REPO_ROOT"

# Train one model then evaluate last.ckpt on pitts30k_test and amstertime.
# All arguments are passed as OmegaConf overrides to main.py.
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

    # Locate the checkpoint folder created by this run (newest matching label_*)
    ckpt_dir=$(ls -td "$REPO_ROOT/logs/checkpoints/${label}_"* 2>/dev/null | head -1 || true)

    if [ -z "$ckpt_dir" ] || [ ! -f "$ckpt_dir/last.ckpt" ]; then
        echo "--- WARNING: last.ckpt not found under $ckpt_dir, skipping eval ---" | tee -a "$log"
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

if [ "${1:-}" = "--suite" ]; then
    run_experiment \
        "model.type=salad_baseline" \
        "wandb.run_name=baseline" \
        || OVERALL=$?

    run_experiment \
        "model.type=salad_joint_depth" \
        "model.normalization.stage=after_mlp" \
        "loss.alignment_loss_type=mse" \
        "wandb.run_name=joint_depth_mse_after_mlp" \
        || OVERALL=$?

    run_experiment \
        "model.type=salad_joint_depth" \
        "model.normalization.stage=after_mlp" \
        "loss.alignment_loss_type=cosine" \
        "wandb.run_name=joint_depth_cosine_after_mlp" \
        || OVERALL=$?
else
    MODEL_TYPES=()
    EXTRA=()
    for arg in "$@"; do
        [[ "$arg" == salad_* && "$arg" != *=* ]] && MODEL_TYPES+=("$arg") || EXTRA+=("$arg")
    done
    [ ${#MODEL_TYPES[@]} -eq 0 ] && MODEL_TYPES=("salad_baseline")

    for mt in "${MODEL_TYPES[@]}"; do
        run_experiment "model.type=${mt}" "${EXTRA[@]+"${EXTRA[@]}"}" || OVERALL=$?
    done
fi

exit "$OVERALL"
