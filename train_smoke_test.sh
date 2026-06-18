#!/usr/bin/env bash
# Single-run smoke test for the new depth distillation model types.
#
# Trains salad_predictor_global with one alpha_global value, then evaluates
# the saved checkpoint on four datasets and writes recall results to a CSV.
# Training is logged to W&B automatically (configure wandb.entity in config.yaml
# or set WANDB_ENTITY env var before running).
#
# Usage:
#   bash train_smoke_test.sh <gsvcities_path> [model_type] [alpha_global]
#
# Defaults:
#   model_type   = salad_predictor_global
#   alpha_global = 0.1
#
# MLP base config is fixed to the best prior experiment:
#   cosine alignment loss, no normalization, alpha_local=0.2
#
# Examples:
#   bash train_smoke_test.sh /data/gsvcities
#   bash train_smoke_test.sh /data/gsvcities salad_global_depth 0.05
#   bash train_smoke_test.sh /data/gsvcities salad_predictor_global 0.5

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$REPO_ROOT/env/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "Error: virtualenv not found at $REPO_ROOT/env"
    exit 1
fi

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <gsvcities_path> [model_type] [alpha_global]"
    echo "  model_type options: salad_predictor_global | salad_global_depth | salad_global_local_depth"
    exit 1
fi

export GSVCITIES_PATH="$1"
export AMSTERTIME_PATH="${AMSTERTIME_PATH:-/home/eng/giborda/delavpr/datasets/amstertime/}"

MODEL_TYPE="${2:-salad_predictor_global}"
ALPHA_GLOBAL="${3:-0.1}"
RUN_NAME="smoke_${MODEL_TYPE}_ag${ALPHA_GLOBAL}"

mkdir -p "$REPO_ROOT/logs/runs" "$REPO_ROOT/logs/eval"
cd "$REPO_ROOT"

TS=$(date +%Y%m%d_%H%M%S)
TRAIN_LOG="$REPO_ROOT/logs/runs/${RUN_NAME}_${TS}.log"
CSV_OUT="$REPO_ROOT/logs/eval/smoke_test.csv"

echo "========================================"
echo "Smoke test run : $RUN_NAME"
echo "Model type     : $MODEL_TYPE"
echo "alpha_global   : $ALPHA_GLOBAL"
echo "Train log      : $TRAIN_LOG"
echo "Eval CSV       : $CSV_OUT"
echo "Time           : $(date)"
echo "========================================"

# ---------------------------------------------------------------------------
# Step 1: Train
# Validates on pitts30k_val each epoch (monitored for best checkpoint).
# Also validates on msls_val each epoch to track multi-dataset recall.
# All training metrics (MS loss, local/global depth loss, LR) go to W&B.
# ---------------------------------------------------------------------------
echo "--- Step 1: Training ---"
# Base MLP config: cosine loss + no normalization (best prior experiment)
"$PYTHON" main.py \
    "model.type=${MODEL_TYPE}" \
    "loss.alpha_global=${ALPHA_GLOBAL}" \
    "loss.alpha_local=0.2" \
    "model.mlp.normalization=none" \
    "loss.alignment_loss_type=cosine" \
    "training.val_set_names=[pitts30k_val,msls_val]" \
    "wandb.run_name=${RUN_NAME}" \
    2>&1 | tee "$TRAIN_LOG"

echo "--- Training complete at $(date) ---"

# ---------------------------------------------------------------------------
# Step 2: Find the checkpoint directory for this run
# ---------------------------------------------------------------------------
CKPT_DIR=$(ls -td "$REPO_ROOT/logs/checkpoints/${RUN_NAME}_"* 2>/dev/null | head -1 || true)

if [ -z "$CKPT_DIR" ] || [ ! -f "$CKPT_DIR/last.ckpt" ]; then
    echo "ERROR: checkpoint not found at $CKPT_DIR — cannot run eval"
    exit 1
fi

echo "--- Using checkpoint: $CKPT_DIR/last.ckpt ---"

# ---------------------------------------------------------------------------
# Step 3: Evaluate on all four benchmark datasets
# Results are appended to smoke_test.csv (columns: run_name, dataset, R@1 …).
# Use last.ckpt for a consistent comparison across runs.
# ---------------------------------------------------------------------------
echo "--- Step 2: Multi-dataset evaluation ---"
"$PYTHON" eval.py \
    --ckpt_path "$CKPT_DIR/last.ckpt" \
    --val_datasets pitts30k_test amstertime msls_val nordland \
    --image_size 322 322 \
    --batch_size 256 \
    --run_name "${RUN_NAME}" \
    --csv_name "smoke_test" \
    2>&1 | tee -a "$TRAIN_LOG"

echo "========================================"
echo "Smoke test done at $(date)"
echo "Recall results : $CSV_OUT"
echo "W&B run        : $RUN_NAME"
echo "========================================"
