#!/usr/bin/env bash
# Instructor-specified depth distillation trials.
#
# Block A: salad_predictor_global — alpha_global sweep, alpha_local fixed at 0.2
#   Base config matches best previous experiment: cosine loss, no normalization
# Block B: salad_global_depth    — alpha_global sweep, with/without linear proj
# Block C: salad_global_local_depth — alpha_global x alpha_local grid
# Block D: extended dataset evaluation (adds msls_val + nordland)
#
# Usage:
#   bash train_trials.sh <gsvcities_path> [--block A|B|C|D|all]
#   Default: run all blocks sequentially.
#
# Datasets for Block D eval:
#   pitts30k_test, amstertime, msls_val, nordland

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$REPO_ROOT/env/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "Error: virtualenv not found at $REPO_ROOT/env"
    exit 1
fi

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <gsvcities_path> [--block A|B|C|D|all]"
    exit 1
fi

export GSVCITIES_PATH="$1"
export AMSTERTIME_PATH="${AMSTERTIME_PATH:-/home/eng/giborda/delavpr/datasets/amstertime/}"
BLOCK="${2:-all}"; BLOCK="${BLOCK/--block/}"; BLOCK="${BLOCK# }"
mkdir -p "$REPO_ROOT/logs/runs"
cd "$REPO_ROOT"

CSV_OUT="$REPO_ROOT/logs/eval/trials.csv"
mkdir -p "$(dirname "$CSV_OUT")"

run_experiment() {
    local label="" log ts ckpt_dir rc=0 extra_datasets="${EXTRA_EVAL_DATASETS:-pitts30k_test amstertime}"
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
    echo "--- Evaluating $label on: $extra_datasets ---" | tee -a "$log"
    # shellcheck disable=SC2086
    "$PYTHON" eval.py --ckpt_path "$ckpt_dir/last.ckpt" \
        --val_datasets $extra_datasets \
        --image_size 322 322 --batch_size 256 \
        --run_name "$label" --csv_path "$CSV_OUT" \
        2>&1 | tee -a "$log"
    echo "--- All done: $label at $(date) ---" | tee -a "$log"
}

OVERALL=0

# ---------------------------------------------------------------------------
# Block A: salad_predictor_global — explore alternative variant first
# Loss: L_MS(d_salad) + alpha_global * L_MS(SALAD(pred)) + 0.2 * L_local
# ---------------------------------------------------------------------------
if [[ "$BLOCK" == "A" || "$BLOCK" == "all" ]]; then
    echo "=== Block A: salad_predictor_global alpha_global sweep ==="
    # Base config: cosine loss, no normalization, alpha_local=0.2
    # (best-performing config from prior joint_depth experiments)
    for ag in 0.05 0.1 0.5 1.0; do
        run_experiment \
            "model.type=salad_predictor_global" \
            "loss.alpha_global=${ag}" \
            "loss.alpha_local=0.2" \
            "model.mlp.normalization=none" \
            "loss.alignment_loss_type=cosine" \
            "wandb.run_name=pred_global_cos_none_ag${ag}" \
            || OVERALL=$?
    done
fi

# ---------------------------------------------------------------------------
# Block B: salad_global_depth — direct depth features into SALAD
# Loss: L_MS(d_salad) + alpha_global * L_MS(SALAD(f_depth))
# With and without learnable linear projection (p.s. convergence fix)
# ---------------------------------------------------------------------------
if [[ "$BLOCK" == "B" || "$BLOCK" == "all" ]]; then
    echo "=== Block B: salad_global_depth alpha_global sweep ==="
    for ag in 0.05 0.1 0.5 1.0; do
        run_experiment \
            "model.type=salad_global_depth" \
            "loss.alpha_global=${ag}" \
            "loss.use_linear_proj=false" \
            "wandb.run_name=global_depth_ag${ag}_noproj" \
            || OVERALL=$?
    done
    echo "=== Block B (linear proj): salad_global_depth with use_linear_proj ==="
    for ag in 0.05 0.1 0.5 1.0; do
        run_experiment \
            "model.type=salad_global_depth" \
            "loss.alpha_global=${ag}" \
            "loss.use_linear_proj=true" \
            "wandb.run_name=global_depth_ag${ag}_proj" \
            || OVERALL=$?
    done
fi

# ---------------------------------------------------------------------------
# Block C: salad_global_local_depth — combined global + local
# Loss: L_MS(d_salad) + alpha_global * L_MS(SALAD(f_depth)) + alpha_local * L_local
# Grid: alpha_global in {0.05, 0.1} x alpha_local in {0.1, 0.5}
# (use best alpha_global from Block B; grid kept small to limit compute)
# ---------------------------------------------------------------------------
if [[ "$BLOCK" == "C" || "$BLOCK" == "all" ]]; then
    echo "=== Block C: salad_global_local_depth alpha grid ==="
    # Same base config as Block A (cosine, no normalization)
    for ag in 0.05 0.1; do
        for al in 0.1 0.5; do
            run_experiment \
                "model.type=salad_global_local_depth" \
                "loss.alpha_global=${ag}" \
                "loss.alpha_local=${al}" \
                "model.mlp.normalization=none" \
                "loss.alignment_loss_type=cosine" \
                "wandb.run_name=global_local_cos_none_ag${ag}_al${al}" \
                || OVERALL=$?
        done
    done
fi

# ---------------------------------------------------------------------------
# Block D: extended dataset evaluation
# Re-evaluates the last checkpoint of each Block A/B/C run on msls_val
# and nordland in addition to pitts30k_test and amstertime.
# ---------------------------------------------------------------------------
if [[ "$BLOCK" == "D" || "$BLOCK" == "all" ]]; then
    echo "=== Block D: extended dataset evaluation (msls_val + nordland) ==="
    EXTRA_EVAL_DATASETS="pitts30k_test amstertime msls_val nordland"
    export EXTRA_EVAL_DATASETS
    CSV_OUT="$REPO_ROOT/logs/eval/trials_extended.csv"
    for ckpt_dir in "$REPO_ROOT/logs/checkpoints/"{pred_global,global_depth,global_local}_*; do
        [ -d "$ckpt_dir" ] || continue
        label=$(basename "$ckpt_dir")
        ckpt="$ckpt_dir/last.ckpt"
        [ -f "$ckpt" ] || continue
        echo "--- Extended eval: $label ---"
        # shellcheck disable=SC2086
        "$PYTHON" eval.py --ckpt_path "$ckpt" \
            --val_datasets $EXTRA_EVAL_DATASETS \
            --image_size 322 322 --batch_size 256 \
            --run_name "${label}_extended" --csv_path "$CSV_OUT" \
            2>&1 | tee -a "$REPO_ROOT/logs/runs/${label}_extended.log" \
            || OVERALL=$?
    done
fi

echo "=== train_trials.sh finished. Overall exit code: $OVERALL ==="
exit "$OVERALL"
