#!/bin/bash
set -euo pipefail

ENVS=("darkroom-easy-small" "junction-3" "navigation-episodic")
SEEDS=(1 2 3)
THRESHOLDS=(0.00001 0.0001 0.001 0.01 0.1 1)
LABEL_STRATEGIES=("mask")
EVAL_OOD=${EVAL_OOD:-true}
EXP_NAME=${EXP_NAME:-history_dagger_disagreement}

EVAL_OOD_FLAG="--eval_ood"
if [ "$EVAL_OOD" = "false" ]; then
  EVAL_OOD_FLAG="--no-eval_ood"
fi

TOTAL=$(( ${#ENVS[@]} * ${#SEEDS[@]} * ${#THRESHOLDS[@]} * ${#LABEL_STRATEGIES[@]} ))
echo "=== History DAgger Disagreement Sweep ==="
echo "Environments:      ${ENVS[*]}"
echo "Seeds:             ${SEEDS[*]}"
echo "Thresholds:        ${THRESHOLDS[*]}"
echo "Label strategies:  ${LABEL_STRATEGIES[*]}"
echo "Eval OOD:          ${EVAL_OOD}"
echo "Exp name:          ${EXP_NAME}"
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
          --wandb_project history_dagger \
          --num_epochs 20 \
          --disagreement_threshold "$THRESH" \
          --label_strategy "$STRATEGY" \
          $EVAL_OOD_FLAG

        # To run jobs in the background instead, replace the python line above with:
        #   python train_history_dagger_disagreement.py ... &
        # and add `wait` after the innermost loop to cap parallelism.
      done
    done
  done
done

echo ""
echo "=== All ${TOTAL} runs complete ==="
