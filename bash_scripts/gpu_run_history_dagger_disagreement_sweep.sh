#!/bin/bash
set -euo pipefail

# One .txt per run; basename matches wandb run name (see train_history_dagger_disagreement.py wandb name=).
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"

# ── Multi-GPU round-robin ──────────────────────────────────────────────
# Each run is assigned to the next GPU in round-robin order.
# One concurrent job per GPU — no MPS needed.
# Override: GPU_IDS="0 1 2 3" ./bash_scripts/gpu_run_history_dagger_disagreement_sweep.sh
read -ra GPU_IDS <<< "${GPU_IDS:-0 1}"
NUM_GPUS=${#GPU_IDS[@]}
MAX_PARALLEL=${MAX_PARALLEL:-$NUM_GPUS}

ENVS=("darkroom-easy-small" "junction-3" "navigation-episodic")
SEEDS=(1)
THRESHOLDS=(0.00001 0.0001 0.001 0.01 0.1 1)
LABEL_STRATEGIES=("mask" "blend")
EVAL_OOD=${EVAL_OOD:-false}
EXP_NAME=${EXP_NAME:-sweep_disagreement}
# Set SAVE_MODEL=false to skip checkpoints (saves disk with large ensembles).
SAVE_MODEL=${SAVE_MODEL:-true}

EVAL_OOD_FLAG="--eval_ood"
if [ "$EVAL_OOD" = "false" ]; then
  EVAL_OOD_FLAG="--no-eval_ood"
fi

SAVE_MODEL_FLAG=""
if [ "$SAVE_MODEL" = "false" ]; then
  SAVE_MODEL_FLAG="--no-save_model"
fi

TOTAL=$(( ${#ENVS[@]} * ${#SEEDS[@]} * ${#THRESHOLDS[@]} * ${#LABEL_STRATEGIES[@]} ))
echo "=== History DAgger Disagreement Sweep ==="
echo "Per-run logs:      ${LOG_DIR}/<wandb_run_name>.txt"
echo "Environments:      ${ENVS[*]}"
echo "Seeds:             ${SEEDS[*]}"
echo "Thresholds:        ${THRESHOLDS[*]}"
echo "Label strategies:  ${LABEL_STRATEGIES[*]}"
echo "Eval OOD:          ${EVAL_OOD}"
echo "Exp name:          ${EXP_NAME}"
echo "Save model:        ${SAVE_MODEL}"
echo "GPUs:              ${GPU_IDS[*]} (${NUM_GPUS} GPUs, round-robin)"
echo "Max parallel:      ${MAX_PARALLEL}"
echo "Total runs:        ${TOTAL}"
echo "=========================================="

# Job control: needed so `jobs -rp` counts background `python` processes in this script.
set -m

RUN=0
PIDS=()
ANY_FAIL=0

# Ctrl+C only reaches the shell; background `python` jobs keep running unless we kill them.
_sweep_kill_tree() {
  local pid=$1
  local sig=${2:-TERM}
  local c
  for c in $(pgrep -P "$pid" 2>/dev/null); do
    _sweep_kill_tree "$c" "$sig"
  done
  kill -s "$sig" "$pid" 2>/dev/null || true
}

_sweep_on_interrupt() {
  trap - INT TERM
  set +eu
  echo "" >&2
  echo "Interrupted — stopping all training jobs..." >&2
  local pid
  if [ "${#PIDS[@]}" -gt 0 ]; then
    for pid in "${PIDS[@]}"; do
      _sweep_kill_tree "$pid" TERM
    done
  fi
  for pid in $(jobs -p 2>/dev/null); do
    kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 2
  if [ "${#PIDS[@]}" -gt 0 ]; then
    for pid in "${PIDS[@]}"; do
      _sweep_kill_tree "$pid" KILL
    done
  fi
  for pid in $(jobs -p 2>/dev/null); do
    kill -KILL "$pid" 2>/dev/null || true
  done
  exit 130
}
trap '_sweep_on_interrupt' INT TERM

for ENV in "${ENVS[@]}"; do
  for THRESH in "${THRESHOLDS[@]}"; do
    for STRATEGY in "${LABEL_STRATEGIES[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        RUN=$((RUN + 1))
        echo ""
        echo "--- Run ${RUN}/${TOTAL}: env=${ENV} thresh=${THRESH} strategy=${STRATEGY} seed=${SEED} ---"

        # Match wandb: f"{exp_name}-{env_name}-thresh{disagreement_threshold}-{label_strategy}-seed{seed}"
        WANDB_RUN_NAME="$(
          EXP_NAME="$EXP_NAME" ENV_NAME="$ENV" DISAGREEMENT_THRESHOLD="$THRESH" \
          LABEL_STRATEGY="$STRATEGY" SEED="$SEED" \
          python3 - <<'PY'
import os
e = os.environ
t = float(e["DISAGREEMENT_THRESHOLD"])
print(f"{e['EXP_NAME']}-{e['ENV_NAME']}-thresh{t}-{e['LABEL_STRATEGY']}-seed{int(e['SEED'])}")
PY
        )"
        RUN_LOG="${LOG_DIR}/${WANDB_RUN_NAME}.txt"
        echo "Log file: ${RUN_LOG}"

        while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
          wait -n 2>/dev/null || sleep 0.5
        done

        GPU=${GPU_IDS[$(( (RUN - 1) % NUM_GPUS ))]}
        echo "  → GPU ${GPU}"

        CUDA_VISIBLE_DEVICES=$GPU python train_history_dagger_disagreement.py \
          --env_name "$ENV" \
          --exp_name "$EXP_NAME" \
          --dagger_steps 10 \
          --seed "$SEED" \
          --log_wandb \
          --wandb_tags seed_sweep \
          --wandb_project history_dagger_v1 \
          --num_epochs 10 \
          --batch_size 256 \
          --lr 3e-4 \
          --no-visualize \
          --disagreement_threshold "$THRESH" \
          --label_strategy "$STRATEGY" \
          $EVAL_OOD_FLAG \
          $SAVE_MODEL_FLAG >"$RUN_LOG" 2>&1 &
        PIDS+=($!)
      done
    done
  done
done

for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    ANY_FAIL=1
  fi
done

echo ""
if [ "$ANY_FAIL" -ne 0 ]; then
  echo "=== Finished with at least one failed run (see logs above) ==="
  exit 1
fi
echo "=== All ${TOTAL} runs complete ==="
