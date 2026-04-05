import importlib.util
import os
import pathlib
import unittest
from unittest import mock

try:
    import numpy as np
except ImportError:  # pragma: no cover - depends on local dev environment
    np = None

try:
    import torch
except ImportError:  # pragma: no cover - depends on local dev environment
    torch = None


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / 'vlnce_baselines' / 'common' / 'parallel_utils.py'
SPEC = importlib.util.spec_from_file_location('parallel_utils_test_module', MODULE_PATH)
parallel_utils = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(parallel_utils)

batched_posref_update = parallel_utils.batched_posref_update
filter_batch_distribution_rows = parallel_utils.filter_batch_distribution_rows
filter_batch_tensor_rows = parallel_utils.filter_batch_tensor_rows
ddp_mean_equivalent_scale = parallel_utils.ddp_mean_equivalent_scale
positions_to_tensor = parallel_utils.positions_to_tensor
resolve_local_rank = parallel_utils.resolve_local_rank
shard_sequence_by_rank = parallel_utils.shard_sequence_by_rank
validate_parallel_config = parallel_utils.validate_parallel_config


class ResolveLocalRankTest(unittest.TestCase):
    def test_prefers_explicit_cli_rank(self):
        with mock.patch.dict(os.environ, {'LOCAL_RANK': '3'}, clear=True):
            self.assertEqual(resolve_local_rank(1), 1)

    def test_falls_back_to_local_rank_env(self):
        with mock.patch.dict(os.environ, {'LOCAL_RANK': '2'}, clear=True):
            self.assertEqual(resolve_local_rank(None), 2)

    def test_falls_back_to_slurm_localid(self):
        with mock.patch.dict(os.environ, {'SLURM_LOCALID': '4'}, clear=True):
            self.assertEqual(resolve_local_rank(None), 4)


class ValidateParallelConfigTest(unittest.TestCase):
    def test_accepts_valid_multi_gpu_config(self):
        with mock.patch.dict(os.environ, {'WORLD_SIZE': '2'}, clear=True):
            validate_parallel_config(
                gpu_numbers=2,
                num_environments=3,
                torch_gpu_ids=[0, 1],
                simulator_gpu_ids=[0, 1],
            )

    def test_rejects_world_size_mismatch(self):
        with mock.patch.dict(os.environ, {'WORLD_SIZE': '2'}, clear=True):
            with self.assertRaises(ValueError):
                validate_parallel_config(
                    gpu_numbers=1,
                    num_environments=1,
                    torch_gpu_ids=[0],
                    simulator_gpu_ids=[0],
                )

    def test_rejects_missing_gpu_ids(self):
        with self.assertRaises(ValueError):
            validate_parallel_config(
                gpu_numbers=2,
                num_environments=1,
                torch_gpu_ids=[0],
                simulator_gpu_ids=[0, 1],
            )


class SequenceShardingTest(unittest.TestCase):
    def test_shards_sequence_round_robin(self):
        items = list(range(8))
        self.assertEqual(shard_sequence_by_rank(items, rank=0, world_size=2), [0, 2, 4, 6])
        self.assertEqual(shard_sequence_by_rank(items, rank=1, world_size=2), [1, 3, 5, 7])


class DDPMeanEquivalentScaleTest(unittest.TestCase):
    def test_matches_expected_scale(self):
        self.assertEqual(ddp_mean_equivalent_scale(world_size=4, global_count=20), 0.2)


@unittest.skipIf(torch is None or np is None, 'torch/numpy are not installed in this environment')
class BatchTensorHelpersTest(unittest.TestCase):
    def test_positions_to_tensor_preserves_batch_shape(self):
        tensor = positions_to_tensor([[1, 2, 3], [4, 5, 6]], torch.device('cpu'))
        self.assertEqual(tuple(tensor.shape), (2, 3))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_filter_batch_tensor_rows(self):
        tensor = torch.tensor([[1.0], [2.0], [3.0]])
        filtered = filter_batch_tensor_rows(tensor, [0, 2])
        self.assertTrue(torch.equal(filtered, torch.tensor([[1.0], [3.0]])))

    def test_filter_batch_distribution_rows(self):
        distribution = {
            'mu': torch.tensor([[1.0], [2.0], [3.0]]),
            'sigma': torch.tensor([[4.0], [5.0], [6.0]]),
        }
        filtered = filter_batch_distribution_rows(distribution, [1, 2])
        self.assertTrue(torch.equal(filtered['mu'], torch.tensor([[2.0], [3.0]])))
        self.assertTrue(torch.equal(filtered['sigma'], torch.tensor([[5.0], [6.0]])))


@unittest.skipIf(torch is None or np is None, 'torch/numpy are not installed in this environment')
class BatchedPosrefUpdateTest(unittest.TestCase):
    def test_updates_each_batch_row_independently(self):
        nav_logits = torch.zeros((2, 3), dtype=torch.float32)
        updated = batched_posref_update(
            positions=[[1.0, 0.0, 0.0], [10.5, 0.0, 0.0]],
            pred_cur_positions=[[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]],
            ghost_pos_batch=[
                {'ghost_a': [0.0, 0.0, 0.0]},
                {'ghost_b': [8.0, 0.0, 0.0]},
            ],
            gmap_vp_ids_batch=[
                ['ghost_a', 'missing', None],
                ['ghost_b', 'missing', None],
            ],
            nav_logits=nav_logits,
            alpha=1.0,
        )

        self.assertGreater(updated[0, 0].item(), updated[0, 1].item())
        self.assertGreater(updated[1, 0].item(), updated[1, 1].item())
        self.assertEqual(updated[0, 2].item(), 0.0)
        self.assertEqual(updated[1, 2].item(), 0.0)


if __name__ == '__main__':
    unittest.main()
