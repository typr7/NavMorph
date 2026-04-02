#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
SPLIT=${SPLIT:-val_unseen}
EPISODE_COUNT=${EPISODE_COUNT:-1}
TOP_K=${TOP_K:-2}
DEPTH=${DEPTH:-2}
CKPT_PATH=${CKPT_PATH:-}
PRETRAINED_PATH=${PRETRAINED_PATH:-pretrained/model_step_82500.pt}
STAGE2S_CKPT=${STAGE2S_CKPT:-}
OUTPUT_DIR=${OUTPUT_DIR:-data/logs/stage2s/eval}

if [[ -z "$CKPT_PATH" ]]; then
  echo "CKPT_PATH must be set" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

"$PYTHON_BIN" run.py   --exp_name stage2s_online_eval   --run-type eval   --exp-config run_r2r/stage2s_navmorph.yaml   SIMULATOR_GPU_IDS [0]   TORCH_GPU_IDS [0]   GPU_NUMBERS 1   NUM_ENVIRONMENTS 1   EVAL.SPLIT "$SPLIT"   EVAL.EPISODE_COUNT "$EPISODE_COUNT"   EVAL.CKPT_PATH_DIR "$CKPT_PATH"   MODEL.pretrained_path "$PRETRAINED_PATH"   STAGE2S.ENABLED True   STAGE2S.MODE online_eval   STAGE2S.LOG_DIR "$OUTPUT_DIR"   STAGE2S.PLANNER.TOP_K "$TOP_K"   STAGE2S.PLANNER.DEPTH "$DEPTH"   STAGE2S.MODEL.CHECKPOINT "$STAGE2S_CKPT"
