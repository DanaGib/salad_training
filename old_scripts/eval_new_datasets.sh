#!/usr/bin/env bash
# eval_new_datasets.sh — Evaluate all 9 models on the 4 new datasets:
#   baidu, hawkins, laurel_caverns, meshvpr
#
# Runs eval.py --save_descriptors so descriptors are cached for future
# eval_from_cache.py runs. The disk-cache check in utils/extraction.py
# skips already-saved arrays, making this script safe to resume.
#
# Usage:  bash eval_new_datasets.sh
# Override dataset paths with env vars if needed:
#   BAIDU_PATH, HAWKINS_PATH, LAUREL_PATH, MESHVPR_PATH

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$REPO_ROOT/env/bin/python"
CKPT_BASE="$REPO_ROOT/logs/checkpoints"
DESC_CACHE="$REPO_ROOT/logs/desc_cache"
CSV_OUT="$REPO_ROOT/logs/eval/new_datasets_eval.csv"
LOG_DIR="$REPO_ROOT/logs/runs"

export BAIDU_PATH="${BAIDU_PATH:-/home/shared/datasets/baidu/images/test/}"
export HAWKINS_PATH="${HAWKINS_PATH:-/home/shared/datasets/hawkins/hawkins_long_corridor/}"
export LAUREL_PATH="${LAUREL_PATH:-/home/shared/datasets/laurel_caverns/laurel_caverns/}"
export MESHVPR_PATH="${MESHVPR_PATH:-/home/shared/datasets/mesh_vpr/synt_melbourne/}"

[ -f "$PYTHON" ] || { echo "Error: virtualenv not found at $REPO_ROOT/env"; exit 1; }
mkdir -p "$DESC_CACHE" "$(dirname "$CSV_OUT")" "$LOG_DIR"
OVERALL=0

NEW_DS="baidu hawkins laurel_caverns meshvpr"

eval_run() {
    local ckpt="$1" run_name="$2"
    [ -f "$ckpt" ] || { echo "SKIP (not found): $ckpt"; return 0; }
    echo ""; echo "=== Evaluating: $run_name | $(date) ==="
    # shellcheck disable=SC2086
    "$PYTHON" eval.py \
        --ckpt_path "$ckpt" \
        --run_name "$run_name" \
        --val_datasets $NEW_DS \
        --image_size 322 322 \
        --batch_size 256 \
        --save_descriptors \
        --desc_cache_dir "$DESC_CACHE" \
        --num_workers 16 \
        --csv_path "$CSV_OUT" \
        2>&1 | tee -a "$LOG_DIR/${run_name}_new_datasets.log" || OVERALL=$?
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

echo ""; echo "=== eval_new_datasets.sh complete. Results: $CSV_OUT | Exit: $OVERALL ==="
exit "$OVERALL"
