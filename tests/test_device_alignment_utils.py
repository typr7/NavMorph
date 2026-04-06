import importlib.util
from pathlib import Path
import unittest

import torch


UTILS_PATH = Path(__file__).resolve().parents[1] / "vlnce_baselines" / "common" / "utils.py"
spec = importlib.util.spec_from_file_location("navmorph_common_utils", UTILS_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class AlignTensorsToDeviceTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this regression test")
    def test_aligns_imagination_delta_to_position_device(self):
        position = torch.tensor([1.0, 2.0, 3.0])
        imagine_outputs = torch.tensor([[0.5, 0.25, -0.75]], device="cuda")

        aligned_outputs = module.align_tensors_to_device(position, imagine_outputs)
        cur_position = position.unsqueeze(0) + aligned_outputs

        self.assertEqual(aligned_outputs.device, position.device)
        self.assertEqual(cur_position.device, position.device)
        self.assertTrue(torch.allclose(cur_position, torch.tensor([[1.5, 2.25, 2.25]])))


if __name__ == "__main__":
    unittest.main()
