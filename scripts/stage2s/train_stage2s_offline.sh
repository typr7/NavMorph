#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
OUTPUT_DIR=${OUTPUT_DIR:-data/stage2s/offline}
TRAIN_RECORDS=${TRAIN_RECORDS:-}
VAL_RECORDS=${VAL_RECORDS:-}
BATCH_SIZE=${BATCH_SIZE:-4}
EPOCHS=${EPOCHS:-4}
LATENT_DIM=${LATENT_DIM:-32}
CANDIDATE_DIM=${CANDIDATE_DIM:-16}
HIDDEN_DIM=${HIDDEN_DIM:-64}
ROLLOUT_DEPTH=${ROLLOUT_DEPTH:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-3}
DEBUG_SYNTHETIC=${DEBUG_SYNTHETIC:-0}

CMD=(
  "$PYTHON_BIN" scripts/stage2s/train_stage2s_offline.py
  --output-dir "$OUTPUT_DIR"
  --batch-size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --latent-dim "$LATENT_DIM"
  --candidate-dim "$CANDIDATE_DIM"
  --hidden-dim "$HIDDEN_DIM"
  --rollout-depth "$ROLLOUT_DEPTH"
  --learning-rate "$LEARNING_RATE"
)

if [[ "$DEBUG_SYNTHETIC" == "1" ]]; then
  CMD+=(--debug-synthetic)
else
  if [[ -z "$TRAIN_RECORDS" ]]; then
    echo "TRAIN_RECORDS must be set unless DEBUG_SYNTHETIC=1" >&2
    exit 1
  fi
  CMD+=(--train-records "$TRAIN_RECORDS")
  if [[ -n "$VAL_RECORDS" ]]; then
    CMD+=(--val-records "$VAL_RECORDS")
  fi
fi

"${CMD[@]}"
