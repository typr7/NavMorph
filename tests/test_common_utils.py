import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_common_utils_module():
    module_name = "vlnce_baselines.common.utils_under_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(
        module_name,
        Path("vlnce_baselines/common/utils.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_extract_instruction_tokens_returns_array_for_batch_obs():
    utils_mod = _load_common_utils_module()
    observations = [
        {
            "instruction": {
                "tokens": [11, 22, 33],
            }
        }
    ]

    extracted = utils_mod.extract_instruction_tokens(
        observations,
        instruction_sensor_uuid="instruction",
        max_length=5,
        pad_id=0,
    )

    tokens = extracted[0]["instruction"]
    assert isinstance(tokens, np.ndarray)
    assert tokens.shape == (5,)
    assert tokens.tolist() == [11, 22, 33, 0, 0]
