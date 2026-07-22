#!/usr/bin/env bash
# save_descriptors.sh — Extract and cache descriptors for 9 models on their
# remaining (not-yet-evaluated) datasets. Run overnight before eval_from_cache.py.
#
# Group A (7 models): pitts30k_val, Nordland, MSLS_blur, MSLS_weather,
#   MSLS_Challenge_Test, SFXL x4, SVOX x6  (14 datasets each)
# Group B (2 v2 models): SVOX x6 only
#
# The disk-cache check in utils/extraction.py skips already-saved arrays,
# so this script is safe to resume if interrupted.
#
# Usage:  bash save_descriptors.sh
# Override dataset paths with env vars if needed (e.g. SFXL_PATH=...).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$REPO_ROOT/env/bin/python"
CKPT_BASE="$REPO_ROOT/logs/checkpoints"
DESC_CACHE="$REPO_ROOT/logs/desc_cache"
CSV_OUT="$REPO_ROOT/logs/eval/new_evals.csv"
LOG_DIR="$REPO_ROOT/logs/runs"

export MSLS_VAL_PATH="${MSLS_VAL_PATH:-/home/shared/datasets/msls/val/}"
export MSLS_PATH="${MSLS_PATH:-/home/shared/datasets/msls_challenge/}"
export MSLS_CHALLENGE_GT_PATH="${MSLS_CHALLENGE_GT_PATH:-/home/eng/giborda/delavpr/datasets/msls_challenge_GT/test_meta/}"
export SFXL_PATH="${SFXL_PATH:-/home/shared/datasets/SF-XL/processed/test/}"
export SVOX_PATH="${SVOX_PATH:-/home/eng/giborda/delavpr/datasets/SVOX/svox/images/}"

[ -f "$PYTHON" ] || { echo "Error: virtualenv not found at $REPO_ROOT/env"; exit 1; }
mkdir -p "$DESC_CACHE" "$(dirname "$CSV_OUT")" "$LOG_DIR"
OVERALL=0

# Datasets grouped by shared DB to maximise in-memory db_cache reuse.
GROUP_A_DS="pitts30k_val Nordland MSLS_blur MSLS_weather MSLS_Challenge_Test \
    SFXL_v1 SFXL_v2 SFXL_night SFXL_occlusion \
    SVOX SVOX_robotcar_sun SVOX_robotcar_snow SVOX_robotcar_rain \
    SVOX_robotcar_night SVOX_robotcar_overcast"

GROUP_B_DS="SVOX SVOX_robotcar_sun SVOX_robotcar_snow SVOX_robotcar_rain \
    SVOX_robotcar_night SVOX_robotcar_overcast"

save_descs() {
    local ckpt="$1" run_name="$2" datasets="$3"
    [ -f "$ckpt" ] || { echo "SKIP (not found): $ckpt"; return 0; }
    echo ""; echo "=== Saving descriptors: $run_name | $(date) ==="
    # shellcheck disable=SC2086
    "$PYTHON" eval.py \
        --ckpt_path "$ckpt" \
        --run_name "$run_name" \
        --val_datasets $datasets \
        --image_size 322 322 \
        --batch_size 256 \
        --save_descriptors \
        --desc_cache_dir "$DESC_CACHE" \
        --num_workers 16 \
        --csv_path "$CSV_OUT" \
        2>&1 | tee -a "$LOG_DIR/${run_name}_save.log" || OVERALL=$?
    echo "--- Done: $run_name | $(date) ---"
}

# --- Group A: 7 models, 14 datasets each ---
save_descs "$CKPT_BASE/baseline_20260609_001439/dinov2_vitb14_epoch03_R1=0.9469.ckpt" \
           "20260609_001439_epoch03" "$GROUP_A_DS"

save_descs "$CKPT_BASE/baseline_20260609_001439/last.ckpt" \
           "20260609_001439_last" "$GROUP_A_DS"

save_descs "$CKPT_BASE/baseline_bs60_ep4_20260706_200109/last.ckpt" \
           "baseline_bs60_ep4" "$GROUP_A_DS"

save_descs "$CKPT_BASE/baseline_bs80_ep6_es_20260706_165922/last.ckpt" \
           "baseline_bs80_ep6_es" "$GROUP_A_DS"

save_descs "$CKPT_BASE/baseline_bs80_ep6_no_es_20260706_181043/last.ckpt" \
           "baseline_bs80_ep6_no_es" "$GROUP_A_DS"

save_descs "$CKPT_BASE/global_depth_ag0.5_noproj_20260619_074246/last.ckpt" \
           "global_depth_ag0.5_noproj_ext" "$GROUP_A_DS"

save_descs "$CKPT_BASE/global_local_cos_none_ag0.1_al0.1_20260619_123322/last.ckpt" \
           "global_local_cos_none_ag0.1_al0.1_ext" "$GROUP_A_DS"

# --- Group B: 2 v2 models, SVOX datasets only ---
save_descs "$CKPT_BASE/v2_global_local_ag0.02_al0.05_20260630_230727/last.ckpt" \
           "v2_global_local_ag0.02_al0.05" "$GROUP_B_DS"

save_descs "$CKPT_BASE/v2_global_local_ag0.05_al0.1_20260630_190251/last.ckpt" \
           "v2_global_local_ag0.05_al0.1" "$GROUP_B_DS"

echo ""; echo "=== save_descriptors.sh complete. Exit: $OVERALL ==="
exit "$OVERALL"
