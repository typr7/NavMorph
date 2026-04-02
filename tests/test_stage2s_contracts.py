import importlib.util
import sys
import types

from yacs.config import CfgNode as CN


class StubConfig(CN):
    def __init__(self, init_dict=None, key_list=None, new_allowed=True):
        super().__init__(init_dict=init_dict, key_list=key_list, new_allowed=new_allowed)


def _build_stub_habitat_baselines_config():
    cfg = CN()
    cfg.SIMULATOR_GPU_ID = 0
    cfg.SIMULATOR_GPU_IDS = [0]
    cfg.TORCH_GPU_ID = 0
    cfg.TORCH_GPU_IDS = [0]
    cfg.GPU_NUMBERS = 1
    cfg.NUM_ENVIRONMENTS = 1
    cfg.TENSORBOARD_DIR = "data/logs/tensorboard_dirs/"
    cfg.CHECKPOINT_FOLDER = "data/logs/checkpoints/"
    cfg.EVAL_CKPT_PATH_DIR = "data/logs/checkpoints/"
    cfg.RESULTS_DIR = "data/logs/eval_results/"
    cfg.VIDEO_DIR = "data/logs/video/"
    cfg.LOG_FILE = "train.log"
    cfg.TEST_EPISODE_COUNT = -1

    cfg.INFERENCE = CN()
    cfg.INFERENCE.SPLIT = "test"
    cfg.INFERENCE.USE_CKPT_CONFIG = False
    cfg.INFERENCE.SAMPLE = False
    cfg.INFERENCE.CKPT_PATH = ""
    cfg.INFERENCE.PREDICTIONS_FILE = ""
    cfg.INFERENCE.FORMAT = "r2r"
    cfg.INFERENCE.EPISODE_COUNT = -1

    cfg.EVAL = CN()
    cfg.EVAL.USE_CKPT_CONFIG = False
    cfg.EVAL.SPLIT = "val_unseen"
    cfg.EVAL.EPISODE_COUNT = -1
    cfg.EVAL.CKPT_PATH_DIR = ""
    cfg.EVAL.fast_eval = False

    cfg.IL = CN()
    cfg.IL.iters = 30000
    cfg.IL.log_every = 500
    cfg.IL.lr = 1e-5
    cfg.IL.batch_size = 1
    cfg.IL.ml_weight = 1.0
    cfg.IL.expert_policy = "spl"
    cfg.IL.sample_ratio = 0.75
    cfg.IL.decay_interval = 3000
    cfg.IL.max_traj_len = 30
    cfg.IL.max_text_len = 80
    cfg.IL.loc_noise = 0.5
    cfg.IL.waypoint_aug = False
    cfg.IL.ghost_aug = 0.0
    cfg.IL.back_algo = "teleport"
    cfg.IL.tryout = True

    cfg.MODEL = CN()
    cfg.MODEL.task_type = "r2r"
    cfg.MODEL.policy_name = "PolicyViewSelectionETP"
    cfg.MODEL.NUM_ANGLES = 12
    cfg.MODEL.pretrained_path = "pretrained/model_step_82500.pt"
    cfg.MODEL.fix_lang_embedding = False
    cfg.MODEL.fix_pano_embedding = False
    cfg.MODEL.use_depth_embedding = True
    cfg.MODEL.use_sprels = True
    cfg.MODEL.merge_ghost = True
    cfg.MODEL.consume_ghost = True
    cfg.MODEL.spatial_output = False
    cfg.MODEL.RGB_ENCODER = CN()
    cfg.MODEL.RGB_ENCODER.output_size = 512
    cfg.MODEL.DEPTH_ENCODER = CN()
    cfg.MODEL.DEPTH_ENCODER.output_size = 256
    cfg.MODEL.VISUAL_DIM = CN()
    cfg.MODEL.VISUAL_DIM.vis_hidden = 768
    cfg.MODEL.VISUAL_DIM.directional = 128
    cfg.MODEL.INSTRUCTION_ENCODER = CN()
    cfg.MODEL.INSTRUCTION_ENCODER.bidirectional = True
    return cfg


def _install_config_stubs():
    habitat_baselines_pkg = types.ModuleType("habitat_baselines")
    habitat_baselines_config_pkg = types.ModuleType("habitat_baselines.config")
    habitat_baselines_default = types.ModuleType("habitat_baselines.config.default")
    habitat_baselines_default._C = _build_stub_habitat_baselines_config()
    habitat_baselines_config_pkg.default = habitat_baselines_default
    habitat_baselines_pkg.config = habitat_baselines_config_pkg

    habitat_pkg = types.ModuleType("habitat")
    habitat_config_pkg = types.ModuleType("habitat.config")
    habitat_default = types.ModuleType("habitat.config.default")
    habitat_default.CONFIG_FILE_SEPARATOR = ","
    habitat_default.Config = StubConfig
    habitat_config_pkg.default = habitat_default
    habitat_pkg.config = habitat_config_pkg

    habitat_extensions_pkg = types.ModuleType("habitat_extensions")
    habitat_extensions_config_pkg = types.ModuleType("habitat_extensions.config")
    habitat_extensions_default = types.ModuleType("habitat_extensions.config.default")
    habitat_extensions_default.get_extended_config = lambda path=None: CN()
    habitat_extensions_config_pkg.default = habitat_extensions_default
    habitat_extensions_pkg.config = habitat_extensions_config_pkg

    sys.modules["habitat_baselines"] = habitat_baselines_pkg
    sys.modules["habitat_baselines.config"] = habitat_baselines_config_pkg
    sys.modules["habitat_baselines.config.default"] = habitat_baselines_default
    sys.modules["habitat"] = habitat_pkg
    sys.modules["habitat.config"] = habitat_config_pkg
    sys.modules["habitat.config.default"] = habitat_default
    sys.modules["habitat_extensions"] = habitat_extensions_pkg
    sys.modules["habitat_extensions.config"] = habitat_extensions_config_pkg
    sys.modules["habitat_extensions.config.default"] = habitat_extensions_default



def _load_get_config():
    _install_config_stubs()
    module_name = "stage2s_config_default_under_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, "vlnce_baselines/config/default.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.get_config



def test_stage2s_config_node_exists():
    get_config = _load_get_config()
    config = get_config("run_r2r/iter_train.yaml")
    assert hasattr(config, "STAGE2S")
    assert config.STAGE2S.ENABLED is False
    assert config.STAGE2S.COUNTERFACTUAL.PROBE_WIDTH >= 3
from pathlib import Path


def test_stage2s_overlay_configs_exist():
    assert Path("run_r2r/stage2s_navmorph.yaml").exists()
    assert Path("run_r2r/stage2s_counterfactual.yaml").exists()



def test_run_py_has_no_hardcoded_cuda_visible_devices():
    run_text = Path("run.py").read_text()
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "4"' not in run_text



def test_main_bash_exposes_stage2s_entrypoints():
    bash_text = Path("run_r2r/main.bash").read_text()
    assert "stage2s_log)" in bash_text
    assert "stage2s_train)" in bash_text
    assert "stage2s_eval)" in bash_text
