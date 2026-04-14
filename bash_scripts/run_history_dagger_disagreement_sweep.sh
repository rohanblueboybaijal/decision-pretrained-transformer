#!/bin/bash
set -euo pipefail

ENVS=("darkroom-easy-small" "junction-3" "navigation-episodic")
SEEDS=(1)
THRESHOLDS=(0.00001 0.0001 0.001 0.01 0.1 1)
LABEL_STRATEGIES=("mask" "blend")
EVAL_OOD=${EVAL_OOD:-false}
EXP_NAME=${EXP_NAME:-history_dagger_disagreement_log_per_step_query}
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
echo "Environments:      ${ENVS[*]}"
echo "Seeds:             ${SEEDS[*]}"
echo "Thresholds:        ${THRESHOLDS[*]}"
echo "Label strategies:  ${LABEL_STRATEGIES[*]}"
echo "Eval OOD:          ${EVAL_OOD}"
echo "Exp name:          ${EXP_NAME}"
echo "Save model:        ${SAVE_MODEL}"
echo "Total runs:        ${TOTAL}"
echo "=========================================="

RUN=0
for ENV in "${ENVS[@]}"; do
  for THRESH in "${THRESHOLDS[@]}"; do
    for STRATEGY in "${LABEL_STRATEGIES[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        RUN=$((RUN + 1))
        echo ""
        echo "--- Run ${RUN}/${TOTAL}: env=${ENV} thresh=${THRESH} strategy=${STRATEGY} seed=${SEED} ---"

        python train_history_dagger_disagreement.py \
          --env_name "$ENV" \
          --exp_name "$EXP_NAME" \
          --dagger_steps 10 \
          --seed "$SEED" \
          --log_wandb \
          --wandb_project history_dagger_v1 \
          --num_epochs 10 \
          --disagreement_threshold "$THRESH" \
          --label_strategy "$STRATEGY" \
          $EVAL_OOD_FLAG \
          $SAVE_MODEL_FLAG

        # To run jobs in the background instead, replace the python line above with:
        #   python train_history_dagger_disagreement.py ... &
        # and add `wait` after the innermost loop to cap parallelism.
      done
    done
  done
done

echo ""
echo "=== All ${TOTAL} runs complete ==="
