export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

PYTHON_BIN=${PYTHON_BIN:-python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
EXP_NAME=${EXP_NAME:-release_r2r}
CKPT_PATH=${CKPT_PATH:-data/checkpoints/ckpt.pth}
PRETRAINED_PATH=${PRETRAINED_PATH:-pretrained/model_step_100000.pt}

flag_train="--exp_name ${EXP_NAME}
      --run-type train
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 1
      IL.iters 29000
      IL.lr 1e-5
      IL.log_every 500
      IL.ml_weight 1.0
      IL.sample_ratio 0.75
      IL.decay_interval 4000
      IL.load_from_ckpt True
      IL.is_requeue True
      IL.waypoint_aug True
      IL.ckpt_to_load data/checkpoints/ckpt.iter25000.pth
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      MODEL.pretrained_path ${PRETRAINED_PATH}
      "

flag_eval="--exp_name ${EXP_NAME}
      --run-type eval
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 1
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      EVAL.CKPT_PATH_DIR ${CKPT_PATH}
      MODEL.pretrained_path ${PRETRAINED_PATH}
      IL.back_algo control
      "

flag_infer="--exp_name ${EXP_NAME}
      --run-type inference
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 1
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      INFERENCE.CKPT_PATH ${CKPT_PATH}
      INFERENCE.PREDICTIONS_FILE preds.json
      MODEL.pretrained_path ${PRETRAINED_PATH}
      IL.back_algo control
      "

flag_stage2s_log="--exp_name ${EXP_NAME}_stage2s_log
      --run-type eval
      --exp-config run_r2r/stage2s_counterfactual.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 1
      EVAL.CKPT_PATH_DIR ${CKPT_PATH}
      MODEL.pretrained_path ${PRETRAINED_PATH}
      STAGE2S.ENABLED True
      STAGE2S.MODE log_only
      "

flag_stage2s_train="--exp_name ${EXP_NAME}_stage2s_train
      --run-type train
      --exp-config run_r2r/stage2s_navmorph.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 1
      MODEL.pretrained_path ${PRETRAINED_PATH}
      STAGE2S.ENABLED True
      STAGE2S.MODE offline_debug
      "

flag_stage2s_eval="--exp_name ${EXP_NAME}_stage2s_eval
      --run-type eval
      --exp-config run_r2r/stage2s_navmorph.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 1
      EVAL.CKPT_PATH_DIR ${CKPT_PATH}
      MODEL.pretrained_path ${PRETRAINED_PATH}
      STAGE2S.ENABLED True
      STAGE2S.MODE online_eval
      "

mode=$1
case $mode in
      train)
      echo "###### train mode ######"
      ${PYTHON_BIN} run.py $flag_train
      ;;
      eval)
      echo "###### eval mode ######"
      ${PYTHON_BIN} run.py $flag_eval
      ;;
      infer)
      echo "###### infer mode ######"
      ${PYTHON_BIN} run.py $flag_infer
      ;;
      stage2s_log)
      echo "###### stage2s log mode ######"
      ${PYTHON_BIN} run.py $flag_stage2s_log
      ;;
      stage2s_train)
      echo "###### stage2s train mode ######"
      ${PYTHON_BIN} run.py $flag_stage2s_train
      ;;
      stage2s_eval)
      echo "###### stage2s eval mode ######"
      ${PYTHON_BIN} run.py $flag_stage2s_eval
      ;;
      *)
      echo "usage: $0 {train|eval|infer|stage2s_log|stage2s_train|stage2s_eval}"
      exit 1
      ;;
esac
