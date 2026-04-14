#!/bin/bash
set -euo pipefail

ENVS=("darkroom-easy-small" "junction-3" "navigation-episodic")
SEEDS=(1)
EVAL_OOD=${EVAL_OOD:-false}
EXP_NAME=${EXP_NAME:-history_dagger}

EVAL_OOD_FLAG="--eval_ood"
if [ "$EVAL_OOD" = "false" ]; then
  EVAL_OOD_FLAG="--no-eval_ood"
fi

SAVE_MODEL=${SAVE_MODEL:-true}
SAVE_MODEL_FLAG=""
if [ "$SAVE_MODEL" = "false" ]; then
  SAVE_MODEL_FLAG="--no-save_model"
fi

TOTAL=$(( ${#ENVS[@]} * ${#SEEDS[@]} ))
echo "=== History DAgger Sweep ==="
echo "Environments:  ${ENVS[*]}"
echo "Seeds:         ${SEEDS[*]}"
echo "Eval OOD:      ${EVAL_OOD}"
echo "Exp name:      ${EXP_NAME}"
echo "Save model:    ${SAVE_MODEL}"
echo "Total runs:    ${TOTAL}"
echo "============================= ============="

RUN=0
for ENV in "${ENVS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    RUN=$((RUN + 1))
    echo ""
    echo "--- Run ${RUN}/${TOTAL}: env=${ENV} seed=${SEED} ---"

    python train_history_dagger.py \
      --env_name "$ENV" \
      --exp_name "$EXP_NAME" \
      --dagger_steps 10 \
      --seed "$SEED" \
      --log_wandb \
      --wandb_project history_dagger_v1 \
      --num_epochs 10 \
      $EVAL_OOD_FLAG \
      $SAVE_MODEL_FLAG

  done
done

echo ""
echo "=== All ${TOTAL} runs complete ==="
