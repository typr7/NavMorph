from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple, Union

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only in lightweight test envs
    np = None

try:
    import torch
except ImportError:  # pragma: no cover - exercised only in lightweight test envs
    torch = None


def resolve_local_rank(local_rank: Optional[int]) -> int:
    if local_rank is not None:
        return int(local_rank)

    env_local_rank = os.environ.get("LOCAL_RANK")
    if env_local_rank is not None:
        return int(env_local_rank)

    slurm_local_rank = os.environ.get("SLURM_LOCALID")
    if slurm_local_rank is not None:
        return int(slurm_local_rank)

    return 0


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("torch is required for tensor-parallel helpers")


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("numpy is required for tensor-parallel helpers")


def normalize_gpu_id_list(
    gpu_ids: Union[int, Sequence[int]],
) -> Sequence[int]:
    if isinstance(gpu_ids, (list, tuple)):
        return list(gpu_ids)
    return [int(gpu_ids)]


def unwrap_data_parallel_module(module: object) -> object:
    while hasattr(module, "module"):
        module = getattr(module, "module")
    return module


def _resolve_attr_path(root: object, path: Tuple[str, ...]) -> Optional[object]:
    current = root
    for attr in path:
        if not hasattr(current, attr):
            return None
        current = getattr(current, attr)
    return current


def resolve_global_sap_head(module: object) -> Optional[object]:
    root = unwrap_data_parallel_module(module)
    for path in (
        ("global_sap_head",),
        ("vln_bert", "global_sap_head"),
    ):
        candidate = _resolve_attr_path(root, path)
        if candidate is not None:
            return unwrap_data_parallel_module(candidate)
    return None


def shard_sequence_by_rank(
    items: Sequence[object], rank: int, world_size: int
) -> Sequence[object]:
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(
            f"rank must be in [0, {world_size}), got rank={rank}"
        )
    return list(items)[rank::world_size]


def ddp_mean_equivalent_scale(world_size: int, global_count: float) -> float:
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if global_count <= 0:
        raise ValueError(f"global_count must be > 0, got {global_count}")
    return float(world_size) / float(global_count)


def validate_parallel_config(
    gpu_numbers: int,
    num_environments: int,
    torch_gpu_ids: Union[int, Sequence[int]],
    simulator_gpu_ids: Union[int, Sequence[int]],
) -> None:
    if gpu_numbers < 1:
        raise ValueError(f"GPU_NUMBERS must be >= 1, got {gpu_numbers}")
    if num_environments < 1:
        raise ValueError(
            f"NUM_ENVIRONMENTS must be >= 1, got {num_environments}"
        )

    torch_ids = normalize_gpu_id_list(torch_gpu_ids)
    simulator_ids = normalize_gpu_id_list(simulator_gpu_ids)

    if len(torch_ids) < gpu_numbers:
        raise ValueError(
            "TORCH_GPU_IDS must provide at least GPU_NUMBERS entries: "
            f"got {len(torch_ids)} ids for GPU_NUMBERS={gpu_numbers}"
        )

    if len(simulator_ids) < gpu_numbers:
        raise ValueError(
            "SIMULATOR_GPU_IDS must provide at least GPU_NUMBERS entries: "
            f"got {len(simulator_ids)} ids for GPU_NUMBERS={gpu_numbers}"
        )

    env_world_size = os.environ.get("WORLD_SIZE")
    if env_world_size is not None and int(env_world_size) != gpu_numbers:
        raise ValueError(
            "WORLD_SIZE and GPU_NUMBERS must match under torchrun: "
            f"WORLD_SIZE={env_world_size}, GPU_NUMBERS={gpu_numbers}"
        )


def positions_to_tensor(positions: Sequence[Sequence[float]], device: torch.device) -> torch.Tensor:
    _require_torch()
    _require_numpy()
    if len(positions) == 0:
        return torch.empty((0, 3), dtype=torch.float32, device=device)
    return torch.as_tensor(np.asarray(positions, dtype=np.float32), device=device)


def filter_batch_tensor_rows(
    tensor: Optional[torch.Tensor], keep_indices: Sequence[int]
) -> Optional[torch.Tensor]:
    _require_torch()
    if tensor is None:
        return None
    if len(keep_indices) == 0:
        shape = list(tensor.shape)
        shape[0] = 0
        return tensor.new_empty(shape)
    keep_indices = torch.as_tensor(keep_indices, dtype=torch.long, device=tensor.device)
    return tensor.index_select(0, keep_indices)


def filter_batch_distribution_rows(
    distribution: Optional[Dict[str, torch.Tensor]], keep_indices: Sequence[int]
) -> Optional[Dict[str, torch.Tensor]]:
    if distribution is None:
        return None
    return {
        key: filter_batch_tensor_rows(value, keep_indices)
        for key, value in distribution.items()
    }


def batched_posref_update(
    positions: Sequence[Sequence[float]],
    pred_cur_positions: Sequence[Sequence[float]],
    ghost_pos_batch: Sequence[Dict[str, Sequence[float]]],
    gmap_vp_ids_batch: Sequence[Sequence[Optional[str]]],
    nav_logits: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    _require_numpy()
    updated_logits = nav_logits.clone()

    for batch_idx, vp_ids in enumerate(gmap_vp_ids_batch):
        position = np.asarray(positions[batch_idx], dtype=np.float32)
        pred_cur_position = np.asarray(pred_cur_positions[batch_idx], dtype=np.float32)
        ghost_pos = ghost_pos_batch[batch_idx]

        for vp_idx, vp_id in enumerate(vp_ids):
            if vp_idx >= updated_logits.size(1):
                break
            if vp_id is None:
                continue

            if vp_id in ghost_pos:
                ref_position = np.asarray(ghost_pos[vp_id], dtype=np.float32)
            else:
                ref_position = position

            distance_ref = np.linalg.norm(pred_cur_position - ref_position)
            weight = float(np.exp(-alpha * distance_ref))
            updated_logits[batch_idx, vp_idx] += weight

    return updated_logits
