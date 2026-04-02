from __future__ import annotations

import argparse
import json
import sys
import types
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_stage2s_namespace() -> None:
    package_name = "vlnce_baselines"
    subpackage_name = "vlnce_baselines.stage2s"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(REPO_ROOT / "vlnce_baselines")]
        sys.modules[package_name] = package
    if subpackage_name not in sys.modules:
        subpackage = types.ModuleType(subpackage_name)
        subpackage.__path__ = [str(REPO_ROOT / "vlnce_baselines/stage2s")]
        sys.modules[subpackage_name] = subpackage


def _load_stage2s_module(module_basename: str):
    import importlib.util

    _ensure_stage2s_namespace()
    module_name = f"vlnce_baselines.stage2s.{module_basename}"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPO_ROOT / "vlnce_baselines/stage2s" / f"{module_basename}.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


contracts_mod = _load_stage2s_module("contracts")
dataset_mod = _load_stage2s_module("dataset")
losses_mod = _load_stage2s_module("losses")
metrics_mod = _load_stage2s_module("metrics")
model_mod = _load_stage2s_module("model")


def _make_synthetic_records(
    num_groups: int,
    num_candidates: int,
    latent_dim: int,
    candidate_dim: int,
    seed: int = 0,
):
    generator = torch.Generator().manual_seed(seed)
    records = []
    local_feature_count = max(1, candidate_dim - 4)
    for group_index in range(num_groups):
        episode_id = f"debug-episode-{group_index // 2}"
        state = torch.randn(latent_dim, generator=generator) * 0.25
        latent_state = contracts_mod.StructuredLatentState(history_latent=state.tolist())
        candidates = []
        for candidate_index in range(num_candidates):
            candidate_features = torch.randn(local_feature_count, generator=generator)
            semantic_target = float(num_candidates - candidate_index)
            exec_target = 1.0 if candidate_index < max(1, num_candidates // 2) else 0.0
            blocking_target = 0.0 if exec_target > 0.5 else 1.0
            progress_target = float((num_candidates - candidate_index) / num_candidates)
            successor = state + 0.1 * torch.randn(latent_dim, generator=generator) + progress_target * 0.05
            token = contracts_mod.CandidateToken(
                action_token={
                    "angle": float(candidate_index) / max(num_candidates, 1),
                    "distance": 0.5 + 0.1 * candidate_index,
                },
                candidate_local={
                    f"f{i}": float(candidate_features[i].item())
                    for i in range(local_feature_count)
                },
                semantic_bundle={
                    "semantic_target": semantic_target,
                    "rank_proxy": semantic_target,
                },
            )
            outcome = contracts_mod.CounterfactualOutcome(
                executable=exec_target,
                blocking_failure=blocking_target,
                reachable_ratio=1.0 if exec_target else 0.1,
                realized_displacement=0.3 + 0.1 * progress_target,
                goal_progress_delta=progress_target,
            )
            successor_latent = contracts_mod.StructuredLatentState(history_latent=successor.tolist())
            candidates.append(
                contracts_mod.CandidateRecord(
                    candidate_index=candidate_index,
                    token=token,
                    outcome=outcome,
                    successor_latent=successor_latent,
                )
            )
        records.append(
            contracts_mod.CandidateSetRecord(
                episode_id=episode_id,
                step_id=group_index,
                candidate_set_id=f"{episode_id}:{group_index}",
                latent_state=latent_state,
                candidates=candidates,
            )
        )
    return records


def _make_dataloader(dataset, batch_size: int, latent_dim: int, candidate_dim: int, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(
            dataset_mod.collate_group_tensor_batch,
            latent_dim=latent_dim,
            candidate_dim=candidate_dim,
        ),
    )


def _move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


LOSS_TERM_KEYS = [
    "total_loss",
    "latent_loss",
    "exec_loss",
    "progress_loss",
    "blocking_loss",
    "semantic_loss",
    "ranking_loss",
    "uncertainty_loss",
]


def _aggregate_epoch_metrics(loss_rows: List[Dict[str, float]], buffers: Dict[str, List[torch.Tensor]], ranking_hits: float, ranking_groups: int):
    metrics = {}
    for key in LOSS_TERM_KEYS:
        metrics[key] = float(sum(row[key] for row in loss_rows) / max(len(loss_rows), 1))

    if buffers["exec_prob"]:
        exec_prob = torch.cat(buffers["exec_prob"])
        exec_target = torch.cat(buffers["exec_target"])
        exec_mask = torch.ones_like(exec_target)
        exec_stats = metrics_mod.binary_auc_ap(exec_prob, exec_target, exec_mask)
        metrics["exec_auroc"] = exec_stats["auroc"]
        metrics["exec_ap"] = exec_stats["ap"]
        metrics["exec_ece"] = metrics_mod.expected_calibration_error(exec_prob, exec_target, exec_mask)
    else:
        metrics["exec_auroc"] = 0.5
        metrics["exec_ap"] = 0.0
        metrics["exec_ece"] = 0.0

    if buffers["blocking_prob"]:
        blocking_prob = torch.cat(buffers["blocking_prob"])
        blocking_target = torch.cat(buffers["blocking_target"])
        blocking_mask = torch.ones_like(blocking_target)
        blocking_stats = metrics_mod.binary_auc_ap(blocking_prob, blocking_target, blocking_mask)
        metrics["blocking_auroc"] = blocking_stats["auroc"]
        metrics["blocking_ap"] = blocking_stats["ap"]
        metrics["blocking_ece"] = metrics_mod.expected_calibration_error(blocking_prob, blocking_target, blocking_mask)
    else:
        metrics["blocking_auroc"] = 0.5
        metrics["blocking_ap"] = 0.0
        metrics["blocking_ece"] = 0.0

    if buffers["progress_pred"]:
        progress_pred = torch.cat(buffers["progress_pred"])
        progress_target = torch.cat(buffers["progress_target"])
        progress_mask = torch.ones_like(progress_target)
        progress_stats = metrics_mod.regression_stats(progress_pred, progress_target, progress_mask)
        metrics["progress_mae"] = progress_stats["mae"]
        metrics["progress_corr"] = progress_stats["corr"]
    else:
        metrics["progress_mae"] = 0.0
        metrics["progress_corr"] = 0.0

    metrics["ranking_top1_acc"] = float(ranking_hits / max(ranking_groups, 1))
    return metrics


def _run_epoch(model, dataloader, optimizer=None, loss_weights=None, device=torch.device("cpu")):
    training = optimizer is not None
    model.train(training)
    loss_rows: List[Dict[str, float]] = []
    buffers = {
        "exec_prob": [],
        "exec_target": [],
        "blocking_prob": [],
        "blocking_target": [],
        "progress_pred": [],
        "progress_target": [],
    }
    ranking_hits = 0.0
    ranking_groups = 0

    for batch in dataloader:
        batch = _move_batch_to_device(batch, device)
        if training:
            outputs = model(batch)
            terms = losses_mod.stage2s_loss_terms(outputs, batch, weights=loss_weights)
            optimizer.zero_grad()
            terms["total_loss"].backward()
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(batch)
                terms = losses_mod.stage2s_loss_terms(outputs, batch, weights=loss_weights)

        loss_rows.append({key: float(terms[key].detach().cpu().item()) for key in LOSS_TERM_KEYS})

        candidate_mask = batch["candidate_mask"].detach().cpu().bool()
        exec_prob = torch.sigmoid(outputs["exec_logit"].detach().cpu())
        blocking_prob = torch.sigmoid(outputs["blocking_logit"].detach().cpu())
        progress_pred = outputs["progress_mean"].detach().cpu()
        progress_target = batch["progress_target"].detach().cpu()
        exec_target = batch["exec_target"].detach().cpu()
        blocking_target = batch["blocking_target"].detach().cpu()

        buffers["exec_prob"].append(exec_prob[candidate_mask].reshape(-1))
        buffers["exec_target"].append(exec_target[candidate_mask].reshape(-1))
        buffers["blocking_prob"].append(blocking_prob[candidate_mask].reshape(-1))
        buffers["blocking_target"].append(blocking_target[candidate_mask].reshape(-1))
        buffers["progress_pred"].append(progress_pred[candidate_mask].reshape(-1))
        buffers["progress_target"].append(progress_target[candidate_mask].reshape(-1))

        ranking_hits += metrics_mod.ranking_top1_accuracy(
            outputs["semantic_logit"].detach().cpu(),
            batch["semantic_target"].detach().cpu(),
            batch["candidate_mask"].detach().cpu(),
        ) * batch["candidate_mask"].size(0)
        ranking_groups += int(batch["candidate_mask"].size(0))

    return _aggregate_epoch_metrics(loss_rows, buffers, ranking_hits, ranking_groups)


def _loss_weights_from_args(args) -> Dict[str, float]:
    return {
        "latent": args.latent_loss_weight,
        "exec": args.exec_loss_weight,
        "progress": args.progress_loss_weight,
        "blocking": args.blocking_loss_weight,
        "semantic": args.semantic_loss_weight,
        "ranking": args.ranking_loss_weight,
        "uncertainty": args.uncertainty_loss_weight,
    }


def _train_offline(args) -> Dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.debug_synthetic:
        train_records = _make_synthetic_records(
            num_groups=args.synthetic_train_groups,
            num_candidates=args.synthetic_num_candidates,
            latent_dim=args.latent_dim,
            candidate_dim=args.candidate_dim,
            seed=args.seed,
        )
        val_records = _make_synthetic_records(
            num_groups=args.synthetic_val_groups,
            num_candidates=args.synthetic_num_candidates,
            latent_dim=args.latent_dim,
            candidate_dim=args.candidate_dim,
            seed=args.seed + 1,
        )
        calibration_records = []
    else:
        if not args.train_records:
            raise ValueError("--train-records is required unless --debug-synthetic is used")
        train_dataset_full = dataset_mod.GroupedCounterfactualDataset(args.train_records)
        if args.val_records:
            train_records = train_dataset_full.records
            val_records = dataset_mod.GroupedCounterfactualDataset(args.val_records).records
            calibration_records = []
        else:
            splits = dataset_mod.split_grouped_records(
                train_dataset_full.records,
                val_ratio=args.val_ratio,
                calibration_ratio=args.calibration_ratio,
            )
            train_records = splits["train"]
            val_records = splits["val"]
            calibration_records = splits["calibration"]

    train_dataset = dataset_mod.GroupedCounterfactualDataset.from_records(train_records)
    val_dataset = dataset_mod.GroupedCounterfactualDataset.from_records(val_records) if val_records else None

    train_loader = _make_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        candidate_dim=args.candidate_dim,
        shuffle=True,
    )
    val_loader = _make_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        candidate_dim=args.candidate_dim,
        shuffle=False,
    ) if val_dataset is not None else None

    model = model_mod.Stage2SModel(
        latent_dim=args.latent_dim,
        candidate_dim=args.candidate_dim,
        hidden_dim=args.hidden_dim,
        rollout_depth=args.rollout_depth,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_weights = _loss_weights_from_args(args)

    history = []
    best_epoch_record = None
    best_metric = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer=optimizer, loss_weights=loss_weights)
        epoch_record = {"epoch": epoch, **{f"train_{key}": value for key, value in train_metrics.items()}}
        if val_loader is not None:
            val_metrics = _run_epoch(model, val_loader, optimizer=None, loss_weights=loss_weights)
            epoch_record.update({f"val_{key}": value for key, value in val_metrics.items()})
            selection_metric = val_metrics["total_loss"]
        else:
            selection_metric = train_metrics["total_loss"]
        history.append(epoch_record)

        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "args": vars(args),
        }
        torch.save(checkpoint_payload, output_dir / "checkpoint_last.pt")
        if selection_metric <= best_metric:
            best_metric = selection_metric
            best_epoch_record = epoch_record
            torch.save(checkpoint_payload, output_dir / "checkpoint_best.pt")

    run_summary = {
        "history": history,
        "best_epoch": None if best_epoch_record is None else best_epoch_record["epoch"],
        "best_metric": best_metric,
        "train_groups": len(train_records),
        "val_groups": len(val_records),
        "calibration_groups": len(calibration_records),
    }
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2))
    return run_summary


def run_one_debug_step(output_dir) -> Dict[str, float]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _make_synthetic_records(
        num_groups=4,
        num_candidates=4,
        latent_dim=32,
        candidate_dim=16,
        seed=123,
    )
    dataset = dataset_mod.GroupedCounterfactualDataset.from_records(records)
    batch = dataset_mod.collate_group_tensor_batch(dataset.records[:2], latent_dim=32, candidate_dim=16)
    model = model_mod.Stage2SModel(latent_dim=32, candidate_dim=16, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    outputs = model(batch)
    terms = losses_mod.stage2s_loss_terms(outputs, batch)
    optimizer.zero_grad()
    terms["total_loss"].backward()
    optimizer.step()
    metrics = {key: float(value.detach().cpu().item()) for key, value in terms.items()}
    (output_dir / "debug_step_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Stage2S offline modules without the online NavMorph trainer")
    parser.add_argument("--train-records", type=str, default="")
    parser.add_argument("--val-records", type=str, default="")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--candidate-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--rollout-depth", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--calibration-ratio", type=float, default=0.0)
    parser.add_argument("--latent-loss-weight", type=float, default=1.0)
    parser.add_argument("--exec-loss-weight", type=float, default=1.0)
    parser.add_argument("--progress-loss-weight", type=float, default=1.0)
    parser.add_argument("--blocking-loss-weight", type=float, default=1.0)
    parser.add_argument("--semantic-loss-weight", type=float, default=0.5)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.5)
    parser.add_argument("--uncertainty-loss-weight", type=float, default=0.0)
    parser.add_argument("--debug-synthetic", action="store_true")
    parser.add_argument("--synthetic-train-groups", type=int, default=12)
    parser.add_argument("--synthetic-val-groups", type=int, default=6)
    parser.add_argument("--synthetic-num-candidates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, object]:
    args = build_arg_parser().parse_args(argv)
    run_summary = _train_offline(args)
    print(json.dumps(run_summary, indent=2))
    return run_summary


if __name__ == "__main__":
    main()
