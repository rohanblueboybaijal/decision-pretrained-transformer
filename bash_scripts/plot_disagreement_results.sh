#!/bin/bash
# From repo root: run eval-return / query-rate plots (W&B) and per-timestep disagreement
# grids (local metrics.json) for each env.
# Output layout: plots/<env_name>/return.pdf and plots/<env_name>/disagreement.pdf
#
# Override experiment name to match training (required if not using the default), e.g.:
#   EXP_NAME=history_dagger_disagreement_log_per_step_query ./bash_scripts/plot_disagreement_results.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PROJECT="${PROJECT:-history_dagger_v1}"
SEED="${SEED:-1}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/history_dagger_disagreement_results}"
# Must match train_history_dagger_disagreement.py --exp_name (and metrics.json config.exp_name)
EXP_NAME="${EXP_NAME:-history_dagger_disagreement}"

ENVS=(
  "darkroom-easy-small"
  "junction-3"
  "navigation-episodic"
)

echo "Repo root:     $REPO_ROOT"
echo "W&B project:   $PROJECT"
echo "Results dir:   $RESULTS_DIR"
echo "Timestep seed: $SEED"
echo "exp_name:      $EXP_NAME"
echo "Environments:  ${ENVS[*]}"
echo ""

for ENV in "${ENVS[@]}"; do
  OUT_DIR="$REPO_ROOT/plots/$ENV"
  mkdir -p "$OUT_DIR"

  echo "=== $ENV ==="
  python -m viz.plot_disagreement_eval_mask_blend \
    --env_name "$ENV" \
    --project "$PROJECT" \
    --exp_name "$EXP_NAME" \
    --save_path "$OUT_DIR/return.pdf"

  python -m viz.plot_disagreement_timestep_grid \
    --results_dir "$RESULTS_DIR" \
    --env_name "$ENV" \
    --seed "$SEED" \
    --exp_name "$EXP_NAME" \
    --save_path "$OUT_DIR/disagreement.pdf"

  echo ""
done

echo "Done. PDFs under plots/<env_name>/"
