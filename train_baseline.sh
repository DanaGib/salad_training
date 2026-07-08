#!/usr/bin/env bash
# train_baseline.sh — Sequential baseline training sweep.
#
# Runs 3 experiments with model.type=salad_baseline:
#   Run 1: batch_size=80, max_epochs=6, early stopping ON  (patience=2)
#   Run 2: batch_size=80, max_epochs=6, early stopping OFF (patience=0)
#   Run 3: batch_size=60, max_epochs=4, early stopping OFF (fixed epochs)
#
# Usage: bash train_baseline.sh <gsvcities_path>
#
# Checkpoints: ./logs/checkpoints/<label>_<timestamp>/
# Stdout logs:  ./logs/runs/<label>_<timestamp>.log

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$REPO_ROOT/env/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "Error: virtualenv not found at $REPO_ROOT/env"
    exit 1
fi

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <gsvcities_path>"
    exit 1
fi

export GSVCITIES_PATH="$1"
mkdir -p "$REPO_ROOT/logs/runs"
cd "$REPO_ROOT"
OVERALL=0

# ---------------------------------------------------------------------------
# run <label> [main.py overrides...]
#   Executes main.py, tees output to a timestamped log file, and accumulates
#   any non-zero exit code into OVERALL so the sweep reports a failure at end.
# ---------------------------------------------------------------------------
run() {
    local label="$1"; shift
    local ts log
    ts=$(date +%Y%m%d_%H%M%S)
    log="$REPO_ROOT/logs/runs/${label}_${ts}.log"
    echo ""
    echo "======== $label | $(date) ========"
    echo "Log: $log"
    "$PYTHON" main.py wandb.run_name="$label" "$@" 2>&1 | tee "$log" || OVERALL=$?
    echo "--- Done: $label | $(date) ---"
}

# ---------------------------------------------------------------------------
# Run 1: batch_size=80, max_epochs=6, early stopping ON (patience=2)
# ---------------------------------------------------------------------------
echo ""
echo "=== Run 1: baseline bs=80 ep=6 early_stopping=ON ==="
run baseline_bs80_ep6_es \
    model.type=salad_baseline \
    training.batch_size=80 \
    training.max_epochs=6 \
    training.early_stop_patience=2

# ---------------------------------------------------------------------------
# Run 2: batch_size=80, max_epochs=6, early stopping OFF
# ---------------------------------------------------------------------------
echo ""
echo "=== Run 2: baseline bs=80 ep=6 early_stopping=OFF ==="
run baseline_bs80_ep6_no_es \
    model.type=salad_baseline \
    training.batch_size=80 \
    training.max_epochs=6 \
    training.early_stop_patience=0

# ---------------------------------------------------------------------------
# Run 3: batch_size=60, max_epochs=4, fixed epochs (no early stopping)
# ---------------------------------------------------------------------------
echo ""
echo "=== Run 3: baseline bs=60 ep=4 fixed_epochs ==="
run baseline_bs60_ep4 \
    model.type=salad_baseline \
    training.batch_size=60 \
    training.max_epochs=4 \
    training.early_stop_patience=0

echo ""
echo "=== train_baseline.sh complete | $(date) | Overall exit: $OVERALL ==="
exit "$OVERALL"
