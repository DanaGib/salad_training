#!/usr/bin/env bash
# eval_svox.sh — Evaluate checkpoints on SVOX and all 5 RobotCar domain subsets.
#
# Datasets evaluated (all share the SVOX 2012 gallery, 10 m GT threshold):
#   SVOX                  — SVOX 2014 queries      (14,278 images)
#   SVOX_robotcar_sun     — RobotCar Sun            (854 images)
#   SVOX_robotcar_snow    — RobotCar Snow           (870 images)
#   SVOX_robotcar_rain    — RobotCar Rain           (937 images)
#   SVOX_robotcar_night   — RobotCar Night          (823 images)
#   SVOX_robotcar_overcast — RobotCar Overcast      (872 images)
#
# Gallery descriptors are extracted once and cached in memory across all 6
# dataset calls (extraction.py db_cache keyed by num_references=17166).
#
# Results are appended to logs/eval/svox_eval.csv.
# Override SVOX_PATH before calling if the dataset is in a non-default location.
#
# Usage:
#   tmux new -s eval_svox
#   bash eval_svox.sh
#   Ctrl+B D

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$REPO_ROOT/env/bin/python"
CKPT_BASE="$REPO_ROOT/logs/checkpoints"
CSV_OUT="$REPO_ROOT/logs/eval/svox_eval.csv"
LOG_DIR="$REPO_ROOT/logs/runs"

# export SVOX_PATH="${SVOX_PATH:-/home/eng/giborda/delavpr/datasets/SVOX/svox/}"
export SVOX_PATH="${SVOX_PATH:-/home/eng/giborda/delavpr/datasets/SVOX/svox/images/}"

DATASETS="SVOX SVOX_robotcar_sun SVOX_robotcar_snow SVOX_robotcar_rain SVOX_robotcar_night SVOX_robotcar_overcast"

[ -f "$PYTHON" ] || { echo "Error: virtualenv not found at $REPO_ROOT/env"; exit 1; }
mkdir -p "$(dirname "$CSV_OUT")" "$LOG_DIR"
OVERALL=0

# eval_run <run_name> <ckpt_dir_prefix> <extra_params_json>
# Discovers the most recent checkpoint directory matching
# $CKPT_BASE/<ckpt_dir_prefix>_*, evaluates last.ckpt on all SVOX datasets,
# and appends a row per dataset to svox_eval.csv.
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
    echo "======== Eval SVOX: $run_name | $(date) ========"
    # shellcheck disable=SC2086
    "$PYTHON" eval.py \
        --ckpt_path "$ckpt_dir/last.ckpt" \
        --val_datasets $DATASETS \
        --image_size 322 322 \
        --batch_size 256 \
        --run_name "$run_name" \
        --csv_path "$CSV_OUT" \
        --extra_params "$params" \
        2>&1 | tee -a "$LOG_DIR/${run_name}_svox.log" || OVERALL=$?
    echo "--- Done: $run_name | $(date) ---"
}

# ---------------------------------------------------------------------------
# Add eval_run calls below, one per checkpoint to evaluate.
# Example (uncomment and fill in actual run names / extra params):
#


# --- Baselines ---


# eval_run "$CKPT_BASE/baseline_20260609_001439/last.ckpt" \
#          "20260609_001439_last"

# # --- Additional baseline sweeps ---

# eval_run "$CKPT_BASE/baseline_bs80_ep6_no_es_20260706_181043/last.ckpt" \
#          "baseline_bs80_ep6_no_es"

# # --- v2 models ---
# eval_run "$CKPT_BASE/v2_global_local_ag0.02_al0.05_20260630_230727/last.ckpt" \
#          "v2_global_local_ag0.02_al0.05"

# eval_run "$CKPT_BASE/v2_global_local_ag0.05_al0.1_20260630_190251/last.ckpt" \
#          "v2_global_local_ag0.05_al0.1"


eval_run baseline_20260609_001439 \
          baseline_20260609_001439 \
          '{"model_type":"salad_baseline"}'

 eval_run baseline_bs80_ep6_no_es \
          baseline_bs80_ep6_no_es \
          '{"model_type":"salad_baseline"}'

#  eval_run baseline \
#           baseline_20260609_001439 \
#           '{"model_type":"salad_baseline"}'

#  eval_run baseline \
#           baseline_bs80_ep6_no_es \
#           '{"model_type":"salad_baseline"}'

 eval_run v2_global_local_ag0.05_al0.1 \
         v2_global_local_ag0.05_al0.1 \
         '{"model_type":"salad_global_local_depth","alpha_global":0.05,"alpha_local":0.1}'

 eval_run v2_global_local_ag0.02_al0.05 \
         v2_global_local_ag0.02_al0.05\
         '{"model_type":"salad_global_local_depth","alpha_global":0.02,"alpha_local":0.05}'
# ---------------------------------------------------------------------------

echo ""
echo "=== eval_svox.sh complete. Results: $CSV_OUT | Exit: $OVERALL ==="
exit "$OVERALL"
