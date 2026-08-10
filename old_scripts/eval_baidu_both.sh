#!/usr/bin/env bash
# eval_baidu_both.sh — Evaluate all 9 models on both Baidu splits in one pass.
#
# Datasets per run:
#   baidu          — Berton/VPR-methods-evaluation split (5 207 DB / 3 208 Q)
#                    threshold 25 m, XY from @x@y@ filenames
#   baidu_original — Official IDL_dataset_cvpr17 split (689 DB / 2 292 Q)
#                    threshold 10 m, XYZ from .camera pose files
#
# Thresholds are passed explicitly to eval.py so the script is immune to
# BAIDU_THRESHOLD_M / BAIDU_ORIGINAL_THRESHOLD_M env-var overrides.
#
# Descriptors are saved to DESC_CACHE so eval_from_cache.py can reuse them.
# The cache check in utils/extraction.py is idempotent, making this script
# safe to resume after interruption.
#
# Usage:  bash eval_baidu_both.sh
# Override dataset paths with env vars if needed:
#   BAIDU_PATH, BAIDU_ORIGINAL_PATH

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$REPO_ROOT/env/bin/python"
CKPT_BASE="$REPO_ROOT/logs/checkpoints"
DESC_CACHE="$REPO_ROOT/logs/desc_cache"
CSV_OUT="$REPO_ROOT/logs/eval/baidu_both_eval.csv"
LOG_DIR="$REPO_ROOT/logs/runs"

export BAIDU_PATH="${BAIDU_PATH:-/home/shared/datasets/baidu/images/test/}"
export BAIDU_ORIGINAL_PATH="${BAIDU_ORIGINAL_PATH:-/home/eng/giborda/delavpr/salad/datasets/baidu_original/raw/}"

[ -f "$PYTHON" ] || { echo "Error: virtualenv not found at $REPO_ROOT/env"; exit 1; }
mkdir -p "$DESC_CACHE" "$(dirname "$CSV_OUT")" "$LOG_DIR"
OVERALL=0

eval_run() {
    local ckpt="$1" run_name="$2"
    [ -f "$ckpt" ] || { echo "SKIP (not found): $ckpt"; return 0; }
    echo ""; echo "=== Evaluating: $run_name | $(date) ==="
    "$PYTHON" eval.py \
        --ckpt_path "$ckpt" \
        --run_name "$run_name" \
        --val_datasets baidu baidu_original \
        --baidu_threshold 25.0 \
        --baidu_original_threshold 10.0 \
        --image_size 322 322 \
        --batch_size 256 \
        --save_descriptors \
        --desc_cache_dir "$DESC_CACHE" \
        --num_workers 16 \
        --csv_path "$CSV_OUT" \
        2>&1 | tee -a "$LOG_DIR/${run_name}_baidu_both.log" || OVERALL=$?
    echo "--- Done: $run_name | $(date) ---"
}

# --- Baselines ---
eval_run "$CKPT_BASE/baseline_20260609_001439/dinov2_vitb14_epoch03_R1=0.9469.ckpt" \
         "20260609_001439_epoch03"

eval_run "$CKPT_BASE/baseline_20260609_001439/last.ckpt" \
         "20260609_001439_last"

# --- Additional baseline sweeps ---
eval_run "$CKPT_BASE/baseline_bs60_ep4_20260706_200109/last.ckpt" \
         "baseline_bs60_ep4"

eval_run "$CKPT_BASE/baseline_bs80_ep6_es_20260706_165922/last.ckpt" \
         "baseline_bs80_ep6_es"

eval_run "$CKPT_BASE/baseline_bs80_ep6_no_es_20260706_181043/last.ckpt" \
         "baseline_bs80_ep6_no_es"

# --- Depth-guided models ---
eval_run "$CKPT_BASE/global_depth_ag0.5_noproj_20260619_074246/last.ckpt" \
         "global_depth_ag0.5_noproj_ext"

eval_run "$CKPT_BASE/global_local_cos_none_ag0.1_al0.1_20260619_123322/last.ckpt" \
         "global_local_cos_none_ag0.1_al0.1_ext"

# --- v2 models ---
eval_run "$CKPT_BASE/v2_global_local_ag0.02_al0.05_20260630_230727/last.ckpt" \
         "v2_global_local_ag0.02_al0.05"

eval_run "$CKPT_BASE/v2_global_local_ag0.05_al0.1_20260630_190251/last.ckpt" \
         "v2_global_local_ag0.05_al0.1"

echo ""; echo "=== eval_baidu_both.sh complete. Results: $CSV_OUT | Exit: $OVERALL ==="
exit "$OVERALL"
