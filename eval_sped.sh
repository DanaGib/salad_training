#!/usr/bin/env bash
# eval_sped.sh — Evaluate baseline and v2 checkpoints on the SPED dataset.
#
# Models:
#   baseline_20260609_001439      (logs/checkpoints/baseline_20260609_001439/last.ckpt)
#   baseline_bs80_ep6_no_es       (baseline_bs80_ep6_no_es_*/last.ckpt)
#   v2_global_local_ag0.05_al0.1  (v2_global_local_ag0.05_al0.1_*/last.ckpt)
#   v2_global_local_ag0.02_al0.05 (v2_global_local_ag0.02_al0.05_*/last.ckpt)
#
# Results: logs/eval/sped_eval.csv
# Logs:    logs/runs/<run_name>_sped.log
#
# Usage:
#   tmux new -s eval_sped
#   bash eval_sped.sh
#   Ctrl+B D

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$REPO_ROOT/env/bin/python"
CKPT_BASE="$REPO_ROOT/logs/checkpoints"
CSV_OUT="$REPO_ROOT/logs/eval/sped_eval.csv"
LOG_DIR="$REPO_ROOT/logs/runs"

export SPED_PATH="${SPED_PATH:-/home/shared/datasets/SPEDTEST/}"

DATASETS="SPED"

[ -f "$PYTHON" ] || { echo "Error: virtualenv not found at $REPO_ROOT/env"; exit 1; }
mkdir -p "$(dirname "$CSV_OUT")" "$LOG_DIR"
OVERALL=0

# eval_run <run_name> <ckpt_dir_prefix> <extra_params_json>
eval_run() {
    local run_name="$1" ckpt_prefix="$2" params="$3"
    local ckpt_dir
    if [ -d "$CKPT_BASE/$ckpt_prefix" ]; then
        ckpt_dir="$CKPT_BASE/$ckpt_prefix"
    else
        ckpt_dir=$(ls -td "$CKPT_BASE/${ckpt_prefix}_"* 2>/dev/null | head -1 || true)
    fi
    if [ -z "$ckpt_dir" ] || [ ! -f "$ckpt_dir/last.ckpt" ]; then
        echo "SKIP (checkpoint not found): $ckpt_prefix"
        return 0
    fi
    echo ""
    echo "======== Eval SPED: $run_name | $(date) ========"
    # shellcheck disable=SC2086
    "$PYTHON" eval.py \
        --ckpt_path "$ckpt_dir/last.ckpt" \
        --val_datasets $DATASETS \
        --image_size 322 322 \
        --batch_size 256 \
        --run_name "$run_name" \
        --csv_path "$CSV_OUT" \
        --extra_params "$params" \
        2>&1 | tee -a "$LOG_DIR/${run_name}_sped.log" || OVERALL=$?
    echo "--- Done: $run_name | $(date) ---"
}

# --- Baselines ---
eval_run baseline_20260609_001439 \
         baseline_20260609_001439 \
         '{"model_type":"salad_baseline"}'

eval_run baseline_bs80_ep6_no_es \
         baseline_bs80_ep6_no_es \
         '{"model_type":"salad_baseline"}'

# --- v2 models ---
eval_run v2_global_local_ag0.05_al0.1 \
         v2_global_local_ag0.05_al0.1 \
         '{"model_type":"salad_global_local_depth","alpha_global":0.05,"alpha_local":0.1}'

eval_run v2_global_local_ag0.02_al0.05 \
         v2_global_local_ag0.02_al0.05 \
         '{"model_type":"salad_global_local_depth","alpha_global":0.02,"alpha_local":0.05}'

echo ""
echo "=== eval_sped.sh complete. Results: $CSV_OUT | Exit: $OVERALL ==="
exit "$OVERALL"