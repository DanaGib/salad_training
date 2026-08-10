#!/usr/bin/env bash
# eval_hard_datasets.sh — Evaluate the 5 best models (Nordland R@1 >= 80) on
# challenging benchmarks:
#   - SF-XL test: v1, v2, night, occlusion subsets
#   - MSLS Challenge Test: 6 cities with local GT (test_meta CSVs)
#   - MSLS degraded val: blur and weather (snow) query subsets
#
# Results are appended to logs/eval/hard_datasets.csv.
# Override dataset paths with env vars before calling if needed:
#   SFXL_PATH=...  MSLS_PATH=...  MSLS_CHALLENGE_GT_PATH=...
#
# Usage:
#   tmux new -s eval_hard
#   bash eval_hard_datasets.sh
#   Ctrl+B D

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$REPO_ROOT/env/bin/python"
CKPT_BASE="$REPO_ROOT/logs/checkpoints"
CSV_OUT="$REPO_ROOT/logs/eval/hard_datasets.csv"
LOG_DIR="$REPO_ROOT/logs/runs"

export SFXL_PATH="${SFXL_PATH:-/home/shared/datasets/SF-XL/processed/test/}"
export MSLS_PATH="${MSLS_PATH:-/home/shared/datasets/msls_challenge/}"
export MSLS_CHALLENGE_GT_PATH="${MSLS_CHALLENGE_GT_PATH:-/home/eng/giborda/delavpr/datasets/msls_challenge_GT/test_meta/}"
export MSLS_VAL_PATH="${MSLS_VAL_PATH:-/home/shared/datasets/msls/val/}"

DATASETS="SFXL_v1 SFXL_v2 SFXL_night SFXL_occlusion MSLS_Challenge_Test MSLS_blur MSLS_weather"

[ -f "$PYTHON" ] || { echo "Error: virtualenv not found at $REPO_ROOT/env"; exit 1; }
mkdir -p "$(dirname "$CSV_OUT")" "$LOG_DIR"
OVERALL=0

# eval_run <csv_run_name> <ckpt_dir_prefix> <extra_params_json>
# Discovers the most recent checkpoint directory matching $CKPT_BASE/<ckpt_dir_prefix>_*,
# then evaluates last.ckpt on all DATASETS. The csv_run_name gets a _hard suffix so
# results do not collide with existing rows in trials_v2.csv.
eval_run() {
    local run_name="$1" ckpt_prefix="$2" params="$3"
    local ckpt_dir; ckpt_dir=$(ls -td "$CKPT_BASE/${ckpt_prefix}_"* 2>/dev/null | head -1 || true)
    if [ -z "$ckpt_dir" ] || [ ! -f "$ckpt_dir/last.ckpt" ]; then
        echo "SKIP (checkpoint not found): $ckpt_prefix"; return 0
    fi
    echo ""; echo "======== Eval: $run_name | $(date) ========"
    # shellcheck disable=SC2086
    "$PYTHON" eval.py --ckpt_path "$ckpt_dir/last.ckpt" \
        --val_datasets $DATASETS \
        --image_size 322 322 --batch_size 256 \
        --run_name "${run_name}_hard" \
        --csv_path "$CSV_OUT" \
        --extra_params "$params" \
        2>&1 | tee -a "$LOG_DIR/${run_name}_hard.log" || OVERALL=$?
}

# Shared JSON fragments (v2 training config and v1 training config).
_TBT_V2='"mlp_type":"token_by_token","mlp_norm":"none","loss_type":"cosine","train_image_size":224,"batch_size":80,"max_epochs":6'
_GLB_V2='"mlp_type":"none","mlp_norm":"none","loss_type":"mse","train_image_size":224,"batch_size":80,"max_epochs":6'
_TBT_V1='"mlp_type":"token_by_token","mlp_norm":"none","loss_type":"cosine","train_image_size":224,"batch_size":60,"max_epochs":4'

# --- Top 5 models in descending Nordland R@1 order ---

# Rank 1 — Nordland R@1 82.56
eval_run v2_global_local_ag0.02_al0.05 \
         v2_global_local_ag0.02_al0.05 \
         "{\"model_type\":\"salad_global_local_depth\",$_TBT_V2,\"alpha_global\":0.02,\"alpha_local\":0.05,\"use_linear_proj\":false}"

# Rank 2 — Nordland R@1 82.40
eval_run v2_global_local_ag0.05_al0.1 \
         v2_global_local_ag0.05_al0.1 \
         "{\"model_type\":\"salad_global_local_depth\",$_TBT_V2,\"alpha_global\":0.05,\"alpha_local\":0.1,\"use_linear_proj\":false}"

# Rank 3 — Nordland R@1 82.37 (v1 checkpoint; dir has no _ext suffix)
eval_run global_local_cos_none_ag0.1_al0.1_ext \
         global_local_cos_none_ag0.1_al0.1 \
         "{\"model_type\":\"salad_global_local_depth\",$_TBT_V1,\"alpha_global\":0.1,\"alpha_local\":0.1,\"use_linear_proj\":false}"

# Rank 4 — Nordland R@1 82.07
eval_run v2_global_depth_ag0.05 \
         v2_global_depth_ag0.05 \
         "{\"model_type\":\"salad_global_depth\",$_GLB_V2,\"alpha_global\":0.05,\"alpha_local\":0.0,\"use_linear_proj\":false}"

# Rank 5 — Nordland R@1 80.90
eval_run v2_global_local_ag0.05_al0.05 \
         v2_global_local_ag0.05_al0.05 \
         "{\"model_type\":\"salad_global_local_depth\",$_TBT_V2,\"alpha_global\":0.05,\"alpha_local\":0.05,\"use_linear_proj\":false}"

echo ""; echo "=== eval_hard_datasets.sh complete. Results: $CSV_OUT | Exit: $OVERALL ==="
exit "$OVERALL"
