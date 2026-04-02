from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

from .contracts import CandidateSetRecord, StructuredLatentState
from .logging import load_candidate_set_records


Numeric = Union[int, float]


def _safe_float(value, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


def _flatten_numeric_sequence(values: Optional[Iterable]) -> List[float]:
    if values is None:
        return []
    flattened: List[float] = []
    for value in values:
        if isinstance(value, dict):
            flattened.extend(_flatten_numeric_mapping(value))
        elif isinstance(value, (list, tuple)):
            flattened.extend(_flatten_numeric_sequence(value))
        elif isinstance(value, (int, float)):
            flattened.append(_safe_float(value))
    return flattened


def _flatten_numeric_mapping(mapping: Optional[Dict]) -> List[float]:
    if mapping is None:
        return []
    flattened: List[float] = []
    for key in sorted(mapping.keys()):
        value = mapping[key]
        if isinstance(value, dict):
            flattened.extend(_flatten_numeric_mapping(value))
        elif isinstance(value, (list, tuple)):
            flattened.extend(_flatten_numeric_sequence(value))
        elif isinstance(value, (int, float)):
            flattened.append(_safe_float(value))
    return flattened


def _pad_vector(values: Sequence[float], size: int) -> List[float]:
    truncated = list(values[:size])
    if len(truncated) < size:
        truncated.extend([0.0] * (size - len(truncated)))
    return truncated


def _state_latent_values(state: Optional[StructuredLatentState]) -> List[float]:
    if state is None:
        return []
    values: List[float] = []
    values.extend(_flatten_numeric_sequence(state.history_latent))
    values.extend(_flatten_numeric_sequence(state.stochastic_latent))
    values.extend(_flatten_numeric_sequence(state.memory_latent))
    values.extend(_flatten_numeric_sequence(state.global_latent))
    return values


def _candidate_token_values(candidate_record) -> List[float]:
    token = candidate_record.token
    if token is None:
        return []
    values: List[float] = []
    values.extend(_flatten_numeric_mapping(token.action_token))
    values.extend(_flatten_numeric_mapping(token.candidate_local))
    values.extend(_flatten_numeric_mapping(token.semantic_bundle))
    return values


def _semantic_target(candidate_record) -> float:
    token = candidate_record.token
    if token is None:
        return 0.0
    bundle = token.semantic_bundle or {}
    for key in ("semantic_target", "semantic_score", "rank_proxy"):
        if key in bundle:
            return _safe_float(bundle[key])
    return 0.0


def _outcome_target(candidate_record, field_name: str) -> float:
    outcome = candidate_record.outcome
    if outcome is None:
        return 0.0
    return _safe_float(getattr(outcome, field_name, 0.0))


def build_group_tensor_sample(record: CandidateSetRecord, latent_dim: int, candidate_dim: int) -> Dict:
    import torch

    candidate_count = max(len(record.candidates), 1)
    state_vector = _pad_vector(_state_latent_values(record.latent_state), latent_dim)
    state_latent = torch.tensor([state_vector for _ in range(candidate_count)], dtype=torch.float32)

    candidate_token_rows = []
    next_latent_rows = []
    has_successor = []
    exec_target = []
    progress_target = []
    blocking_target = []
    semantic_target = []
    candidate_mask = []

    if record.candidates:
        candidate_iterable = record.candidates
    else:
        candidate_iterable = [None]

    for candidate_record in candidate_iterable:
        if candidate_record is None:
            candidate_token_rows.append([0.0] * candidate_dim)
            next_latent_rows.append([0.0] * latent_dim)
            has_successor.append(0.0)
            exec_target.append(0.0)
            progress_target.append(0.0)
            blocking_target.append(0.0)
            semantic_target.append(0.0)
            candidate_mask.append(0.0)
            continue

        candidate_token_rows.append(_pad_vector(_candidate_token_values(candidate_record), candidate_dim))
        successor_values = _state_latent_values(candidate_record.successor_latent)
        next_latent_rows.append(_pad_vector(successor_values, latent_dim))
        has_successor.append(1.0 if successor_values else 0.0)
        exec_target.append(_outcome_target(candidate_record, "executable"))
        progress_target.append(_outcome_target(candidate_record, "goal_progress_delta"))
        blocking_target.append(_outcome_target(candidate_record, "blocking_failure"))
        semantic_target.append(_semantic_target(candidate_record))
        candidate_mask.append(1.0)

    return {
        "candidate_set_id": record.candidate_set_id,
        "episode_id": record.episode_id,
        "step_id": record.step_id,
        "state_latent": state_latent,
        "candidate_token": torch.tensor(candidate_token_rows, dtype=torch.float32),
        "next_latent": torch.tensor(next_latent_rows, dtype=torch.float32),
        "has_successor": torch.tensor(has_successor, dtype=torch.float32),
        "exec_target": torch.tensor(exec_target, dtype=torch.float32),
        "progress_target": torch.tensor(progress_target, dtype=torch.float32),
        "blocking_target": torch.tensor(blocking_target, dtype=torch.float32),
        "semantic_target": torch.tensor(semantic_target, dtype=torch.float32),
        "candidate_mask": torch.tensor(candidate_mask, dtype=torch.float32),
    }


def collate_group_tensor_batch(records: Sequence[CandidateSetRecord], latent_dim: int, candidate_dim: int) -> Dict:
    import torch

    samples = [build_group_tensor_sample(record, latent_dim=latent_dim, candidate_dim=candidate_dim) for record in records]
    max_candidates = max(sample["candidate_token"].size(0) for sample in samples)
    batch_size = len(samples)

    batch = {
        "candidate_set_id": [sample["candidate_set_id"] for sample in samples],
        "episode_id": [sample["episode_id"] for sample in samples],
        "step_id": [sample["step_id"] for sample in samples],
        "state_latent": torch.zeros(batch_size, max_candidates, latent_dim, dtype=torch.float32),
        "candidate_token": torch.zeros(batch_size, max_candidates, candidate_dim, dtype=torch.float32),
        "next_latent": torch.zeros(batch_size, max_candidates, latent_dim, dtype=torch.float32),
        "has_successor": torch.zeros(batch_size, max_candidates, dtype=torch.float32),
        "exec_target": torch.zeros(batch_size, max_candidates, dtype=torch.float32),
        "progress_target": torch.zeros(batch_size, max_candidates, dtype=torch.float32),
        "blocking_target": torch.zeros(batch_size, max_candidates, dtype=torch.float32),
        "semantic_target": torch.zeros(batch_size, max_candidates, dtype=torch.float32),
        "candidate_mask": torch.zeros(batch_size, max_candidates, dtype=torch.float32),
    }

    for row_index, sample in enumerate(samples):
        count = sample["candidate_token"].size(0)
        batch["state_latent"][row_index, :count] = sample["state_latent"]
        batch["candidate_token"][row_index, :count] = sample["candidate_token"]
        batch["next_latent"][row_index, :count] = sample["next_latent"]
        batch["has_successor"][row_index, :count] = sample["has_successor"]
        batch["exec_target"][row_index, :count] = sample["exec_target"]
        batch["progress_target"][row_index, :count] = sample["progress_target"]
        batch["blocking_target"][row_index, :count] = sample["blocking_target"]
        batch["semantic_target"][row_index, :count] = sample["semantic_target"]
        batch["candidate_mask"][row_index, :count] = sample["candidate_mask"]

    return batch


def split_grouped_records(
    records: Sequence[CandidateSetRecord],
    val_ratio: float = 0.2,
    calibration_ratio: float = 0.0,
) -> Dict[str, List[CandidateSetRecord]]:
    episode_ids = sorted({record.episode_id for record in records})
    episode_count = len(episode_ids)
    val_count = min(episode_count, max(1 if val_ratio > 0 and episode_count > 1 else 0, int(round(episode_count * val_ratio))))
    calibration_count = min(
        max(0, episode_count - val_count - 1),
        int(round(episode_count * calibration_ratio)),
    )

    val_episodes = set(episode_ids[-val_count:]) if val_count else set()
    calibration_start = max(0, episode_count - val_count - calibration_count)
    calibration_end = episode_count - val_count
    calibration_episodes = set(episode_ids[calibration_start:calibration_end]) if calibration_count else set()

    splits = {"train": [], "val": [], "calibration": []}
    for record in records:
        if record.episode_id in val_episodes:
            splits["val"].append(record)
        elif record.episode_id in calibration_episodes:
            splits["calibration"].append(record)
        else:
            splits["train"].append(record)
    return splits


class GroupedCounterfactualDataset:
    def __init__(self, path: Optional[Union[str, Path]] = None, records: Optional[Sequence[CandidateSetRecord]] = None):
        if path is None and records is None:
            raise ValueError("GroupedCounterfactualDataset requires either a path or in-memory records")
        self.path = None if path is None else Path(path)
        self.records = list(records) if records is not None else load_candidate_set_records(self.path)
        self.group_count = len(self.records)

    @classmethod
    def from_records(cls, records: Sequence[CandidateSetRecord]) -> "GroupedCounterfactualDataset":
        return cls(records=records)

    def __len__(self) -> int:
        return self.group_count

    def __getitem__(self, index: int) -> CandidateSetRecord:
        return self.records[index]

    def build_group_sample(self, index: int) -> Dict:
        record = self.records[index]
        return record.to_dict()

    def build_tensor_sample(self, index: int, latent_dim: int, candidate_dim: int) -> Dict:
        return build_group_tensor_sample(self.records[index], latent_dim=latent_dim, candidate_dim=candidate_dim)
