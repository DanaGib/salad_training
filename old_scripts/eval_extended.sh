#!/usr/bin/env bash
# eval_extended.sh — Re-evaluate all existing Block A/B/C (v1) checkpoints on
# pitts30k_test, amstertime, Nordland, and MSLS val. Results are appended to
# logs/eval/trials_v2.csv with full hyperparam columns so they can be compared
# directly against the new v2 runs.
#
# Usage: bash eval_extended.sh
# Override MSLS_PATH before calling if your MSLS images are in a non-default location.

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

eval_ckpt() {
    local ckpt_dir="$1" run_name="$2" params="$3"
    local ckpt="$ckpt_dir/last.ckpt"
    [ -f "$ckpt" ] || { echo "SKIP (no last.ckpt): $ckpt_dir"; return 0; }
    echo "=== Eval: $run_name | $(date) ==="
    # shellcheck disable=SC2086
    "$PYTHON" eval.py --ckpt_path "$ckpt" \
        --val_datasets $DATASETS \
        --image_size 322 322 --batch_size 256 \
        --run_name "${run_name}_ext" \
        --csv_path "$CSV_OUT" \
        --extra_params "$params" \
        2>&1 | tee -a "$LOG_DIR/${run_name}_ext.log" || OVERALL=$?
}

find_ckpt_dir() { ls -td "$CKPT_BASE/${1}_"* 2>/dev/null | head -1 || true; }

# Block A — salad_predictor_global
for ag in 0.05 0.1 0.5 1.0; do
    P="{\"model_type\":\"salad_predictor_global\",\"mlp_type\":\"token_by_token\",\"mlp_norm\":\"none\",\"alpha_global\":${ag},\"alpha_local\":0.2,\"use_linear_proj\":false,\"loss_type\":\"cosine\",\"train_image_size\":224,\"batch_size\":60,\"max_epochs\":4}"
    RUN="pred_global_cos_none_ag${ag}"
    D=$(find_ckpt_dir "$RUN"); [ -n "$D" ] && eval_ckpt "$D" "$RUN" "$P"
done

# Block B — salad_global_depth noproj and proj
for ag in 0.05 0.1 0.5 1.0; do
    for proj in false true; do
        SUFFIX=$([[ "$proj" == "true" ]] && echo "_proj" || echo "_noproj")
        P="{\"model_type\":\"salad_global_depth\",\"mlp_type\":\"none\",\"mlp_norm\":\"none\",\"alpha_global\":${ag},\"alpha_local\":0.0,\"use_linear_proj\":${proj},\"loss_type\":\"mse\",\"train_image_size\":224,\"batch_size\":60,\"max_epochs\":4}"
        RUN="global_depth_ag${ag}${SUFFIX}"
        D=$(find_ckpt_dir "$RUN"); [ -n "$D" ] && eval_ckpt "$D" "$RUN" "$P"
    done
done

# Block C — salad_global_local_depth
for ag in 0.05 0.1; do
    for al in 0.1 0.5; do
        P="{\"model_type\":\"salad_global_local_depth\",\"mlp_type\":\"token_by_token\",\"mlp_norm\":\"none\",\"alpha_global\":${ag},\"alpha_local\":${al},\"use_linear_proj\":false,\"loss_type\":\"cosine\",\"train_image_size\":224,\"batch_size\":60,\"max_epochs\":4}"
        RUN="global_local_cos_none_ag${ag}_al${al}"
        D=$(find_ckpt_dir "$RUN"); [ -n "$D" ] && eval_ckpt "$D" "$RUN" "$P"
    done
done

echo "=== eval_extended.sh complete. Results: $CSV_OUT | Exit: $OVERALL ==="
exit "$OVERALL"
