#!/usr/bin/env bash
# eval_baselines_on_datasets.sh — Evaluate two baseline checkpoints on 8 hard benchmarks:
#   MSLS Challenge Test, MSLS blur, MSLS weather,
#   SF-XL v1/v2/night/occlusion, SVOX.
#
# Models:
#   20260609_001439_last    (baseline_20260609_001439/last.ckpt)
#   baseline_bs80_ep6_no_es (baseline_bs80_ep6_no_es_20260706_181043/last.ckpt)
#
# Results:  logs/eval/baselines_hard_svox.csv
# Logs:     logs/runs/<run_name>_hard_svox.log
#
# Override dataset paths with env vars before calling if needed.
# Usage:
#   tmux new -s eval_baselines_on_datasets
#   bash eval_baselines_on_datasets.sh
#   Ctrl+B D

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$REPO_ROOT/env/bin/python"
CKPT_BASE="$REPO_ROOT/logs/checkpoints"
CSV_OUT="$REPO_ROOT/logs/eval/baselines_hard_svox.csv"
LOG_DIR="$REPO_ROOT/logs/runs"

export SFXL_PATH="${SFXL_PATH:-/home/shared/datasets/SF-XL/processed/test/}"
export MSLS_PATH="${MSLS_PATH:-/home/shared/datasets/msls_challenge/}"
export MSLS_VAL_PATH="${MSLS_VAL_PATH:-/home/shared/datasets/msls/val/}"
export MSLS_CHALLENGE_GT_PATH="${MSLS_CHALLENGE_GT_PATH:-/home/eng/giborda/delavpr/datasets/msls_challenge_GT/test_meta/}"
export SVOX_PATH="${SVOX_PATH:-/home/eng/giborda/delavpr/datasets/SVOX/svox/}"


# DATASETS="MSLS_Challenge_Test MSLS_blur MSLS_weather SFXL_v1 SFXL_v2 SFXL_night SFXL_occlusion SVOX"
DATASETS="SFXL_v1 SFXL_v2 SFXL_night SFXL_occlusion MSLS_Challenge_Test MSLS_blur MSLS_weather"


[ -f "$PYTHON" ] || { echo "Error: virtualenv not found at $REPO_ROOT/env"; exit 1; }
mkdir -p "$(dirname "$CSV_OUT")" "$LOG_DIR"
OVERALL=0

# eval_run <ckpt_path> <run_name> <extra_params_json>
eval_run() {
    local ckpt="$1" run_name="$2" params="$3"
    [ -f "$ckpt" ] || { echo "SKIP (not found): $ckpt"; return 0; }
    echo ""; echo "======== Eval: $run_name | $(date) ========"
    # shellcheck disable=SC2086
    "$PYTHON" eval.py \
        --ckpt_path "$ckpt" \
        --val_datasets $DATASETS \
        --image_size 322 322 \
        --batch_size 256 \
        --run_name "$run_name" \
        --csv_path "$CSV_OUT" \
        --extra_params "$params" \
        2>&1 | tee -a "$LOG_DIR/${run_name}_hard_svox.log" || OVERALL=$?
    echo "--- Done: $run_name | $(date) ---"
}

# --- Original baseline ---
eval_run "$CKPT_BASE/baseline_20260609_001439/last.ckpt" \
         "20260609_001439_last" \
         '{"model_type":"salad_baseline"}'


# --- bs80 ep6, no early stopping ---
eval_run "$CKPT_BASE/baseline_bs80_ep6_no_es_20260706_181043/last.ckpt" \
         "baseline_bs80_ep6_no_es" \
         '{"model_type":"salad_baseline","batch_size":80,"max_epochs":6,"early_stop_patience":0}'

echo ""; echo "=== eval_baselines_on_datasets.sh complete. Results: $CSV_OUT | Exit: $OVERALL ==="
exit "$OVERALL"