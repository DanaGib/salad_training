#!/usr/bin/env bash
# train_v2.sh — Overnight training sweep for VPR Distillation Trials V2.
#
# Runs 16 experiments sequentially (no eval). Eval is done separately via
# eval_v2.sh the next morning. Experiments run in priority order:
#   Block C-v2: salad_global_local_depth (5 runs) — refine around best v1
#   Block E-v2: 322x322 training (2 runs)         — address resolution gap
#   Block D-v2: Trial 4, pretrained SALAD init (2 runs)
#   Block A-v2: salad_predictor_global (3 runs)
#   Block B-v2: salad_global_depth (4 runs)
#
# Usage: bash train_v2.sh <gsvcities_path>
#
# Detach from tmux with Ctrl+B D after starting.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$REPO_ROOT/env/bin/python"

if [ ! -f "$PYTHON" ]; then echo "Error: virtualenv not found at $REPO_ROOT/env"; exit 1; fi
if [ -z "${1:-}" ]; then echo "Usage: $0 <gsvcities_path>"; exit 1; fi

export GSVCITIES_PATH="$1"
export AMSTERTIME_PATH="${AMSTERTIME_PATH:-/home/eng/giborda/delavpr/datasets/amstertime/}"
export MSLS_PATH="${MSLS_PATH:-/home/shared/datasets/msls_challenge/}"
mkdir -p "$REPO_ROOT/logs/runs"
cd "$REPO_ROOT"
OVERALL=0

run() {
    local label="$1"; shift
    local ts log; ts=$(date +%Y%m%d_%H%M%S); log="$REPO_ROOT/logs/runs/${label}_${ts}.log"
    echo ""; echo "======== $label | $(date) ========"; echo "Log: $log"
    "$PYTHON" main.py wandb.run_name="$label" "$@" 2>&1 | tee "$log" || OVERALL=$?
    echo "--- Done: $label | $(date) ---"
}

# Shared param groups (bash arrays)
TBT=(model.mlp.type=token_by_token model.mlp.normalization=none loss.alignment_loss_type=cosine)
V2=(training.max_epochs=6 training.early_stop_patience=2 training.batch_size=80)
V2_322=(training.max_epochs=6 training.early_stop_patience=2 training.batch_size=40 "training.image_size=[322,322]")

# ---------------------------------------------------------------------------
# Block C-v2: salad_global_local_depth — finer grid around best (ag=0.05, al=0.1)
# ---------------------------------------------------------------------------
echo ""; echo "=== BLOCK C-v2: salad_global_local_depth ==="; echo ""
run v2_global_local_ag0.05_al0.1  model.type=salad_global_local_depth loss.alpha_global=0.05 loss.alpha_local=0.1  "${TBT[@]}" "${V2[@]}"
run v2_global_local_ag0.02_al0.1  model.type=salad_global_local_depth loss.alpha_global=0.02 loss.alpha_local=0.1  "${TBT[@]}" "${V2[@]}"
run v2_global_local_ag0.05_al0.05 model.type=salad_global_local_depth loss.alpha_global=0.05 loss.alpha_local=0.05 "${TBT[@]}" "${V2[@]}"
run v2_global_local_ag0.02_al0.05 model.type=salad_global_local_depth loss.alpha_global=0.02 loss.alpha_local=0.05 "${TBT[@]}" "${V2[@]}"
run v2_global_local_ag0.05_al0.2  model.type=salad_global_local_depth loss.alpha_global=0.05 loss.alpha_local=0.2  "${TBT[@]}" "${V2[@]}"

# ---------------------------------------------------------------------------
# Block E-v2: 322x322 training — directly address 224-train / 322-eval mismatch
# ---------------------------------------------------------------------------
echo ""; echo "=== BLOCK E-v2: 322x322 training ==="; echo ""
run v2_baseline_322              model.type=salad_baseline "${V2_322[@]}"
run v2_global_local_322_ag0.05_al0.1 model.type=salad_global_local_depth loss.alpha_global=0.05 loss.alpha_local=0.1 "${TBT[@]}" "${V2_322[@]}"

# ---------------------------------------------------------------------------
# Block D-v2: Trial 4 — init SALAD aggregator from best pretrained baseline
# ---------------------------------------------------------------------------
BASELINE_CKPT=$(ls -td "$REPO_ROOT/logs/checkpoints/baseline_"* 2>/dev/null | head -1 || true)
BASELINE_CKPT="${BASELINE_CKPT}/last.ckpt"
echo ""; echo "=== BLOCK D-v2: pretrained SALAD init (from $BASELINE_CKPT) ==="; echo ""
run v2_trial4_init_ag0.05_al0.1  model.type=salad_global_local_depth loss.alpha_global=0.05 loss.alpha_local=0.1  "${TBT[@]}" "${V2[@]}" "model.init_aggregator_from=${BASELINE_CKPT}"
run v2_trial4_init_ag0.05_al0.05 model.type=salad_global_local_depth loss.alpha_global=0.05 loss.alpha_local=0.05 "${TBT[@]}" "${V2[@]}" "model.init_aggregator_from=${BASELINE_CKPT}"

# ---------------------------------------------------------------------------
# Block A-v2: salad_predictor_global — finer alpha_global sweep
# ---------------------------------------------------------------------------
echo ""; echo "=== BLOCK A-v2: salad_predictor_global ==="; echo ""
run v2_pred_global_ag0.05 model.type=salad_predictor_global loss.alpha_global=0.05 loss.alpha_local=0.2 "${TBT[@]}" "${V2[@]}"
run v2_pred_global_ag0.02 model.type=salad_predictor_global loss.alpha_global=0.02 loss.alpha_local=0.2 "${TBT[@]}" "${V2[@]}"
run v2_pred_global_ag0.08 model.type=salad_predictor_global loss.alpha_global=0.08 loss.alpha_local=0.2 "${TBT[@]}" "${V2[@]}"

# ---------------------------------------------------------------------------
# Block B-v2: salad_global_depth — push alpha_global higher and lower
# ---------------------------------------------------------------------------
echo ""; echo "=== BLOCK B-v2: salad_global_depth ==="; echo ""
run v2_global_depth_ag0.05 model.type=salad_global_depth loss.alpha_global=0.05 loss.use_linear_proj=false "${V2[@]}"
run v2_global_depth_ag0.02 model.type=salad_global_depth loss.alpha_global=0.02 loss.use_linear_proj=false "${V2[@]}"
run v2_global_depth_ag1.0  model.type=salad_global_depth loss.alpha_global=1.0  loss.use_linear_proj=false "${V2[@]}"
run v2_global_depth_ag2.0  model.type=salad_global_depth loss.alpha_global=2.0  loss.use_linear_proj=false "${V2[@]}"

echo ""; echo "=== train_v2.sh complete | $(date) | Overall exit: $OVERALL ==="
exit "$OVERALL"
