#!/usr/bin/env bash
# eval_v2.sh — Evaluate all v2 training-run checkpoints on 4 benchmarks and
# append full-hyperparam rows to logs/eval/trials_v2.csv.
#
# Run the morning after train_v2.sh completes:
#   tmux new -s eval_v2
#   bash eval_v2.sh
#   Ctrl+B D
#
# Datasets: pitts30k_test, amstertime, Nordland, MSLS (val)
# Override MSLS_PATH before calling if images are in a non-default location.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$REPO_ROOT/env/bin/python"
CKPT_BASE="$REPO_ROOT/logs/checkpoints"
CSV_OUT="$REPO_ROOT/logs/eval/trials_v2.csv"
LOG_DIR="$REPO_ROOT/logs/runs"
export MSLS_PATH="${MSLS_PATH:-/home/shared/datasets/msls_challenge/}"
DATASETS="pitts30k_test amstertime Nordland MSLS"

if [ ! -f "$PYTHON" ]; then echo "Error: virtualenv not found at $REPO_ROOT/env"; exit 1; fi
mkdir -p "$(dirname "$CSV_OUT")" "$LOG_DIR"
OVERALL=0

eval_run() {
    local run_name="$1" params="$2"
    local ckpt_dir; ckpt_dir=$(ls -td "$CKPT_BASE/${run_name}_"* 2>/dev/null | head -1 || true)
    if [ -z "$ckpt_dir" ] || [ ! -f "$ckpt_dir/last.ckpt" ]; then
        echo "SKIP (checkpoint not found): $run_name"; return 0
    fi
    echo ""; echo "======== Eval: $run_name | $(date) ========"
    # shellcheck disable=SC2086
    "$PYTHON" eval.py --ckpt_path "$ckpt_dir/last.ckpt" \
        --val_datasets $DATASETS --image_size 322 322 --batch_size 256 \
        --run_name "$run_name" --csv_path "$CSV_OUT" --extra_params "$params" \
        2>&1 | tee -a "$LOG_DIR/${run_name}_eval.log" || OVERALL=$?
}

# Shared JSON fragments
_TBT='"mlp_type":"token_by_token","mlp_norm":"none","loss_type":"cosine"'
_NONE='"mlp_type":"none","mlp_norm":"none","loss_type":"mse"'
_V2='"train_image_size":224,"batch_size":80,"max_epochs":6'
_V2_322='"train_image_size":322,"batch_size":40,"max_epochs":6'

echo "=== BLOCK C-v2: salad_global_local_depth ==="
eval_run v2_global_local_ag0.05_al0.1  "{\"model_type\":\"salad_global_local_depth\",$_TBT,\"alpha_global\":0.05,\"alpha_local\":0.1,\"use_linear_proj\":false,$_V2}"
eval_run v2_global_local_ag0.02_al0.1  "{\"model_type\":\"salad_global_local_depth\",$_TBT,\"alpha_global\":0.02,\"alpha_local\":0.1,\"use_linear_proj\":false,$_V2}"
eval_run v2_global_local_ag0.05_al0.05 "{\"model_type\":\"salad_global_local_depth\",$_TBT,\"alpha_global\":0.05,\"alpha_local\":0.05,\"use_linear_proj\":false,$_V2}"
eval_run v2_global_local_ag0.02_al0.05 "{\"model_type\":\"salad_global_local_depth\",$_TBT,\"alpha_global\":0.02,\"alpha_local\":0.05,\"use_linear_proj\":false,$_V2}"
eval_run v2_global_local_ag0.05_al0.2  "{\"model_type\":\"salad_global_local_depth\",$_TBT,\"alpha_global\":0.05,\"alpha_local\":0.2,\"use_linear_proj\":false,$_V2}"

echo "=== BLOCK E-v2: 322x322 training ==="
eval_run v2_baseline_322              "{\"model_type\":\"salad_baseline\",$_NONE,\"alpha_global\":0.0,\"alpha_local\":0.0,\"use_linear_proj\":false,$_V2_322}"
eval_run v2_global_local_322_ag0.05_al0.1 "{\"model_type\":\"salad_global_local_depth\",$_TBT,\"alpha_global\":0.05,\"alpha_local\":0.1,\"use_linear_proj\":false,$_V2_322}"

echo "=== BLOCK D-v2: pretrained SALAD init (Trial 4) ==="
eval_run v2_trial4_init_ag0.05_al0.1  "{\"model_type\":\"salad_global_local_depth\",$_TBT,\"alpha_global\":0.05,\"alpha_local\":0.1,\"use_linear_proj\":false,\"init_from_pretrained\":true,$_V2}"
eval_run v2_trial4_init_ag0.05_al0.05 "{\"model_type\":\"salad_global_local_depth\",$_TBT,\"alpha_global\":0.05,\"alpha_local\":0.05,\"use_linear_proj\":false,\"init_from_pretrained\":true,$_V2}"

echo "=== BLOCK A-v2: salad_predictor_global ==="
eval_run v2_pred_global_ag0.05 "{\"model_type\":\"salad_predictor_global\",$_TBT,\"alpha_global\":0.05,\"alpha_local\":0.2,\"use_linear_proj\":false,$_V2}"
eval_run v2_pred_global_ag0.02 "{\"model_type\":\"salad_predictor_global\",$_TBT,\"alpha_global\":0.02,\"alpha_local\":0.2,\"use_linear_proj\":false,$_V2}"
eval_run v2_pred_global_ag0.08 "{\"model_type\":\"salad_predictor_global\",$_TBT,\"alpha_global\":0.08,\"alpha_local\":0.2,\"use_linear_proj\":false,$_V2}"

echo "=== BLOCK B-v2: salad_global_depth ==="
eval_run v2_global_depth_ag0.05 "{\"model_type\":\"salad_global_depth\",$_NONE,\"alpha_global\":0.05,\"alpha_local\":0.0,\"use_linear_proj\":false,$_V2}"
eval_run v2_global_depth_ag0.02 "{\"model_type\":\"salad_global_depth\",$_NONE,\"alpha_global\":0.02,\"alpha_local\":0.0,\"use_linear_proj\":false,$_V2}"
eval_run v2_global_depth_ag1.0  "{\"model_type\":\"salad_global_depth\",$_NONE,\"alpha_global\":1.0,\"alpha_local\":0.0,\"use_linear_proj\":false,$_V2}"
eval_run v2_global_depth_ag2.0  "{\"model_type\":\"salad_global_depth\",$_NONE,\"alpha_global\":2.0,\"alpha_local\":0.0,\"use_linear_proj\":false,$_V2}"

echo ""; echo "=== eval_v2.sh complete. Results: $CSV_OUT | Exit: $OVERALL ==="
exit "$OVERALL"
