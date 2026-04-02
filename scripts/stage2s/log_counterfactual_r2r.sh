#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
EXP_NAME=${EXP_NAME:-stage2s_counterfactual}
SPLIT=${SPLIT:-val_seen}
EPISODE_COUNT=${EPISODE_COUNT:--1}
CKPT_PATH=${CKPT_PATH:-data/checkpoints/ckpt.pth}
PRETRAINED_PATH=${PRETRAINED_PATH:-pretrained/model_step_100000.pt}
OUTPUT_DIR=${OUTPUT_DIR:-data/logs/stage2s}
PROBE_WIDTH=${PROBE_WIDTH:-4}
ROLLOUT_FRACTION=${ROLLOUT_FRACTION:-0.5}

mkdir -p "${OUTPUT_DIR}"

${PYTHON_BIN} run.py \
  --exp_name "${EXP_NAME}" \
  --run-type eval \
  --exp-config run_r2r/stage2s_counterfactual.yaml \
  EVAL.SPLIT "${SPLIT}" \
  EVAL.EPISODE_COUNT "${EPISODE_COUNT}" \
  EVAL.CKPT_PATH_DIR "${CKPT_PATH}" \
  MODEL.pretrained_path "${PRETRAINED_PATH}" \
  STAGE2S.ENABLED True \
  STAGE2S.MODE log_only \
  STAGE2S.LOG_DIR "${OUTPUT_DIR}" \
  STAGE2S.COUNTERFACTUAL.PROBE_WIDTH "${PROBE_WIDTH}" \
  STAGE2S.COUNTERFACTUAL.ROLLOUT_FRACTION "${ROLLOUT_FRACTION}"
