import unittest
import importlib.util
from pathlib import Path

import numpy as np


UTILS_PATH = Path(__file__).resolve().parents[1] / "vlnce_baselines" / "common" / "utils.py"
spec = importlib.util.spec_from_file_location("navmorph_common_utils", UTILS_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
extract_instruction_tokens = module.extract_instruction_tokens


class ExtractInstructionTokensTest(unittest.TestCase):
    def test_instruction_tokens_become_numpy_array_with_padding(self):
        observations = [{"instruction": {"tokens": [11, 22, 33]}}]

        extract_instruction_tokens(
            observations,
            "instruction",
            max_length=5,
            pad_id=0,
        )

        self.assertIsInstance(observations[0]["instruction"], np.ndarray)
        np.testing.assert_array_equal(
            observations[0]["instruction"],
            np.array([11, 22, 33, 0, 0]),
        )


if __name__ == "__main__":
    unittest.main()
