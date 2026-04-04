#!/usr/bin/env bash
set -euo pipefail

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MODE="${1:-}"

EXP_NAME="${EXP_NAME:-release_r2r}"
EXP_CONFIG="${EXP_CONFIG:-run_r2r/iter_train.yaml}"
build_gpu_id_list() {
  local count="${1}"
  local ids=()
  local idx
  for ((idx=0; idx<count; idx++)); do
    ids+=("${idx}")
  done
  local joined
  joined="$(IFS=,; echo "${ids[*]}")"
  printf '[%s]' "${joined}"
}

GPU_NUMBERS="${GPU_NUMBERS:-1}"
NUM_ENVIRONMENTS="${NUM_ENVIRONMENTS:-1}"
IL_BATCH_SIZE="${IL_BATCH_SIZE:-$NUM_ENVIRONMENTS}"
DEFAULT_GPU_ID_LIST="$(build_gpu_id_list "${GPU_NUMBERS}")"
SIMULATOR_GPU_IDS="${SIMULATOR_GPU_IDS:-$DEFAULT_GPU_ID_LIST}"
TORCH_GPU_IDS="${TORCH_GPU_IDS:-$DEFAULT_GPU_ID_LIST}"

TRAIN_ITERS="${TRAIN_ITERS:-29000}"
TRAIN_LR="${TRAIN_LR:-1e-5}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-500}"
TRAIN_ML_WEIGHT="${TRAIN_ML_WEIGHT:-1.0}"
TRAIN_SAMPLE_RATIO="${TRAIN_SAMPLE_RATIO:-0.75}"
TRAIN_DECAY_INTERVAL="${TRAIN_DECAY_INTERVAL:-4000}"
TRAIN_LOAD_FROM_CKPT="${TRAIN_LOAD_FROM_CKPT:-True}"
TRAIN_IS_REQUEUE="${TRAIN_IS_REQUEUE:-True}"
TRAIN_WAYPOINT_AUG="${TRAIN_WAYPOINT_AUG:-True}"
TRAIN_CKPT="${TRAIN_CKPT:-data/checkpoints/ckpt.iter25000.pth}"

MODEL_PRETRAINED_PATH="${MODEL_PRETRAINED_PATH:-pretrained/model_step_100000.pt}"
ALLOW_SLIDING="${ALLOW_SLIDING:-True}"

EVAL_CKPT_PATH="${EVAL_CKPT_PATH:-data/checkpoints/ckpt.pth}"
INFER_CKPT_PATH="${INFER_CKPT_PATH:-data/checkpoints/ckpt.pth}"
INFER_PREDICTIONS_FILE="${INFER_PREDICTIONS_FILE:-preds.json}"
BACK_ALGO="${BACK_ALGO:-control}"

common_flags=(
  --exp_name "${EXP_NAME}"
  --exp-config "${EXP_CONFIG}"
  SIMULATOR_GPU_IDS "${SIMULATOR_GPU_IDS}"
  TORCH_GPU_IDS "${TORCH_GPU_IDS}"
  GPU_NUMBERS "${GPU_NUMBERS}"
  NUM_ENVIRONMENTS "${NUM_ENVIRONMENTS}"
  IL.batch_size "${IL_BATCH_SIZE}"
  TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING "${ALLOW_SLIDING}"
  MODEL.pretrained_path "${MODEL_PRETRAINED_PATH}"
)

train_flags=(
  --run-type train
  "${common_flags[@]}"
  IL.iters "${TRAIN_ITERS}"
  IL.lr "${TRAIN_LR}"
  IL.log_every "${TRAIN_LOG_EVERY}"
  IL.ml_weight "${TRAIN_ML_WEIGHT}"
  IL.sample_ratio "${TRAIN_SAMPLE_RATIO}"
  IL.decay_interval "${TRAIN_DECAY_INTERVAL}"
  IL.load_from_ckpt "${TRAIN_LOAD_FROM_CKPT}"
  IL.is_requeue "${TRAIN_IS_REQUEUE}"
  IL.waypoint_aug "${TRAIN_WAYPOINT_AUG}"
  IL.ckpt_to_load "${TRAIN_CKPT}"
)

eval_flags=(
  --run-type eval
  "${common_flags[@]}"
  EVAL.CKPT_PATH_DIR "${EVAL_CKPT_PATH}"
  IL.back_algo "${BACK_ALGO}"
)

infer_flags=(
  --run-type inference
  "${common_flags[@]}"
  INFERENCE.CKPT_PATH "${INFER_CKPT_PATH}"
  INFERENCE.PREDICTIONS_FILE "${INFER_PREDICTIONS_FILE}"
  IL.back_algo "${BACK_ALGO}"
)

launcher=(python run.py)
if [[ "${GPU_NUMBERS}" -gt 1 ]]; then
  launcher=(torchrun --standalone --nproc_per_node "${GPU_NUMBERS}" run.py)
fi

case "${MODE}" in
  train)
    echo "###### train mode ######"
    "${launcher[@]}" "${train_flags[@]}"
    ;;
  eval)
    echo "###### eval mode ######"
    "${launcher[@]}" "${eval_flags[@]}"
    ;;
  infer)
    echo "###### infer mode ######"
    "${launcher[@]}" "${infer_flags[@]}"
    ;;
  *)
    echo "Usage: $0 {train|eval|infer}" >&2
    exit 1
    ;;
esac
