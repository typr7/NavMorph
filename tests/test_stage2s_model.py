import importlib.util
import sys
import types
from pathlib import Path

import torch


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


def _make_fake_batch(batch_size=2, num_candidates=4, latent_dim=32, candidate_dim=16):
    torch.manual_seed(0)
    return {
        "state_latent": torch.randn(batch_size, num_candidates, latent_dim),
        "candidate_token": torch.randn(batch_size, num_candidates, candidate_dim),
        "candidate_mask": torch.ones(batch_size, num_candidates),
    }


def test_stage2s_model_outputs_transition_and_physical_heads():
    model_mod = _load_stage2s_module("model")
    fake_batch = _make_fake_batch()
    model = model_mod.Stage2SModel(latent_dim=32, candidate_dim=16, hidden_dim=64)
    outputs = model(fake_batch)
    assert outputs["next_latent"].shape == (2, 4, 32)
    assert outputs["exec_logit"].shape == (2, 4)
    assert outputs["progress_mean"].shape == (2, 4)
    assert outputs["blocking_logit"].shape == (2, 4)
    assert outputs["uncertainty_logit"].shape == (2, 4)
    assert outputs["semantic_logit"].shape == (2, 4)


def test_branch_aggregator_is_monotonic_in_exec_when_other_terms_fixed():
    calibration_mod = _load_stage2s_module("calibration")
    agg = calibration_mod.CalibratedBranchAggregator()
    worse = agg(exec_prob=0.2, progress=0.5, blocking=0.1, uncertainty=0.1, semantic=0.6)
    better = agg(exec_prob=0.8, progress=0.5, blocking=0.1, uncertainty=0.1, semantic=0.6)
    assert better > worse


def test_branch_aggregator_penalizes_blocking_and_uncertainty():
    calibration_mod = _load_stage2s_module("calibration")
    agg = calibration_mod.CalibratedBranchAggregator()
    safer = agg(exec_prob=0.8, progress=0.5, blocking=0.1, uncertainty=0.1, semantic=0.6)
    riskier = agg(exec_prob=0.8, progress=0.5, blocking=0.8, uncertainty=0.4, semantic=0.6)
    assert safer > riskier


def test_temperature_scaler_preserves_logit_ordering():
    calibration_mod = _load_stage2s_module("calibration")
    scaler = calibration_mod.TemperatureScaler(initial_temperature=2.0)
    logits = torch.tensor([-1.0, 0.5, 3.0])
    scaled = scaler(logits)
    assert torch.allclose(scaled, logits / 2.0)
    assert torch.equal(torch.argsort(logits), torch.argsort(scaled))


def test_stage2s_losses_mask_missing_successors_and_rank_candidates():
    losses_mod = _load_stage2s_module("losses")
    outputs = {
        "next_latent": torch.tensor([[[1.0, 2.0], [100.0, 200.0]]]),
        "exec_logit": torch.tensor([[2.0, -2.0]]),
        "progress_mean": torch.tensor([[0.9, -0.2]]),
        "blocking_logit": torch.tensor([[-2.0, 2.0]]),
        "uncertainty_logit": torch.tensor([[-0.5, 0.5]]),
        "semantic_logit": torch.tensor([[2.0, 0.0]]),
    }
    batch = {
        "next_latent": torch.tensor([[[0.0, 0.0], [5.0, 5.0]]]),
        "has_successor": torch.tensor([[1.0, 0.0]]),
        "exec_target": torch.tensor([[1.0, 0.0]]),
        "progress_target": torch.tensor([[1.0, -1.0]]),
        "blocking_target": torch.tensor([[0.0, 1.0]]),
        "semantic_target": torch.tensor([[1.0, 0.0]]),
        "candidate_mask": torch.tensor([[1.0, 1.0]]),
    }
    terms = losses_mod.stage2s_loss_terms(outputs, batch)
    assert torch.isclose(terms["latent_loss"], torch.tensor(2.5))
    assert torch.isfinite(terms["total_loss"])
    assert terms["ranking_loss"] < 0.2


def test_pairwise_ranking_loss_prefers_correct_semantic_ordering():
    losses_mod = _load_stage2s_module("losses")
    target = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0]])
    good = losses_mod.pairwise_ranking_loss(
        torch.tensor([[2.0, 0.0]]),
        target,
        mask,
    )
    bad = losses_mod.pairwise_ranking_loss(
        torch.tensor([[0.0, 2.0]]),
        target,
        mask,
    )
    assert good < bad
