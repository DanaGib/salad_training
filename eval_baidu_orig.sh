#!/usr/bin/env bash
# eval_baidu_orig.sh — Evaluate the same 9 models from eval_new_datasets.sh
# on the official Baidu Mall (IDL_dataset_cvpr17) dataset instead of the
# repackaged 'baidu' dataset, so results are directly comparable per model.
#
# Runs eval.py --save_descriptors so descriptors are cached for future
# eval_from_cache.py runs. The disk-cache check in utils/extraction.py
# skips already-saved arrays, making this script safe to resume.
#
# Usage:  bash eval_baidu_orig.sh
# Override the dataset path with an env var if needed:
#   BAIDU_ORIGINAL_PATH

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$REPO_ROOT/env/bin/python"
CKPT_BASE="$REPO_ROOT/logs/checkpoints"
DESC_CACHE="$REPO_ROOT/logs/desc_cache"
CSV_OUT="$REPO_ROOT/logs/eval/baidu_orig_eval.csv"
LOG_DIR="$REPO_ROOT/logs/runs"

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
        --val_datasets baidu_original \
        --image_size 322 322 \
        --batch_size 256 \
        --save_descriptors \
        --desc_cache_dir "$DESC_CACHE" \
        --num_workers 16 \
        --csv_path "$CSV_OUT" \
        2>&1 | tee -a "$LOG_DIR/${run_name}_baidu_orig.log" || OVERALL=$?
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

echo ""; echo "=== eval_baidu_orig.sh complete. Results: $CSV_OUT | Exit: $OVERALL ==="
exit "$OVERALL"
