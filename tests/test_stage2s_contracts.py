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
import subprocess


def test_ss_trainer_etp_py_compile_succeeds():
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", "vlnce_baselines/ss_trainer_ETP.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _load_stage2s_module(module_basename):
    package_name = "vlnce_baselines"
    subpackage_name = "vlnce_baselines.stage2s"
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(Path("vlnce_baselines"))]
        sys.modules[package_name] = pkg
    if subpackage_name not in sys.modules:
        subpkg = types.ModuleType(subpackage_name)
        subpkg.__path__ = [str(Path("vlnce_baselines/stage2s"))]
        sys.modules[subpackage_name] = subpkg

    module_name = f"{subpackage_name}.{module_basename}"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(
        module_name,
        Path("vlnce_baselines/stage2s") / f"{module_basename}.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_candidate_set_record_round_trip_preserves_group_identity():
    contracts = _load_stage2s_module("contracts")
    record = contracts.CandidateSetRecord(
        episode_id="ep-1",
        step_id=3,
        candidate_set_id="ep-1:3",
        candidates=[],
    )
    restored = contracts.CandidateSetRecord.from_dict(record.to_dict())
    assert restored.candidate_set_id == "ep-1:3"
    assert restored.step_id == 3


def test_build_candidate_set_record_returns_grouped_record():
    contracts = _load_stage2s_module("contracts")
    logging_mod = _load_stage2s_module("logging")
    latent_state = contracts.StructuredLatentState(history_latent=[0.1, 0.2])
    candidate = contracts.CandidateRecord(candidate_index=0)
    record = logging_mod.build_candidate_set_record(
        candidate_set_id="scene-1:ep-1:3",
        latent_state=latent_state,
        candidates=[candidate],
        episode_id="ep-1",
        step_id=3,
    )
    assert record.candidate_set_id == "scene-1:ep-1:3"
    assert len(record.candidates) == 1
    assert record.candidates[0].candidate_index == 0
import ast
from copy import deepcopy
import numpy as np


class FakeAgentState:
    def __init__(self, position, rotation):
        self.position = position
        self.rotation = rotation


class FakeSim:
    def __init__(self):
        self.position = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        self.rotation = {"yaw": 0.5}
        self.last_set_position_type = None

    def get_agent_state(self):
        return FakeAgentState(position=self.position, rotation=self.rotation)

    def set_agent_state(self, position, rotation):
        self.last_set_position_type = type(position)
        self.position = np.asarray(position, dtype=np.float32)
        self.rotation = deepcopy(rotation)



def test_choose_probe_indices_keeps_top_semantic_and_one_hard_negative():
    probing = _load_stage2s_module("probing")
    semantic_scores = [9.0, 8.8, 7.5, 1.0, 0.5]
    indices = probing.choose_probe_indices(semantic_scores, probe_width=4)
    assert 0 in indices
    assert 1 in indices
    assert 4 in indices
    assert len(indices) == 4



def test_pack_and_restore_sim_snapshot_with_fake_sim():
    probing = _load_stage2s_module("probing")
    sim = FakeSim()
    snapshot = probing.pack_sim_snapshot(sim)
    sim.set_agent_state([9.0, 9.0, 9.0], {"yaw": 9.0})
    probing.restore_sim_snapshot(sim, snapshot)
    restored = sim.get_agent_state()
    assert isinstance(snapshot["position"], np.ndarray)
    assert sim.last_set_position_type is np.ndarray
    assert restored.position.tolist() == [1.0, 2.0, 3.0]
    assert restored.rotation == {"yaw": 0.5}



def test_summarize_probe_outcome_marks_blocking_when_short_motion():
    probing = _load_stage2s_module("probing")
    outcome = probing.summarize_probe_outcome(
        intended_forward=1.0,
        executed_forward=0.2,
        start_goal_distance=5.0,
        end_goal_distance=4.9,
        tolerance_ratio=0.8,
    )
    assert outcome.blocking_failure == 1.0
    assert outcome.executable == 0.0
    assert outcome.reachable_ratio == 0.2
    assert outcome.goal_progress_delta == 0.1



def test_environments_exposes_stage2s_probe_methods():
    tree = ast.parse(Path("vlnce_baselines/common/environments.py").read_text())
    methods = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "VLNCEDaggerEnv":
            methods = {child.name for child in node.body if isinstance(child, ast.FunctionDef)}
            break
    assert "get_agent_pose_snapshot" in methods
    assert "restore_agent_pose_snapshot" in methods
    assert "probe_candidate_action" in methods



def test_ss_trainer_etp_exposes_stage2s_log_only_hooks():
    trainer_text = Path("vlnce_baselines/ss_trainer_ETP.py").read_text()
    assert "def _stage2s_log_only_enabled" in trainer_text
    assert "def _stage2s_probe_candidates" in trainer_text
