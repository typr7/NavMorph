from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from .contracts import CandidateToken, StructuredLatentState


def _flatten_numeric(value) -> List[float]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [float(x) for x in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        flattened: List[float] = []
        for item in value:
            flattened.extend(_flatten_numeric(item))
        return flattened
    if isinstance(value, (int, float)):
        return [float(value)]
    return []


def _maybe_truncate(values: List[float], max_dims: Optional[int]) -> List[float]:
    if max_dims is None:
        return values
    return values[:max_dims]


def build_stage2s_state_bundle(
    *,
    history_latent,
    stochastic_latent=None,
    memory_latent=None,
    global_latent=None,
    max_dims: Optional[int] = None,
) -> StructuredLatentState:
    return StructuredLatentState(
        history_latent=_maybe_truncate(_flatten_numeric(history_latent), max_dims),
        stochastic_latent=_maybe_truncate(_flatten_numeric(stochastic_latent), max_dims) or None,
        memory_latent=_maybe_truncate(_flatten_numeric(memory_latent), max_dims) or None,
        global_latent=_maybe_truncate(_flatten_numeric(global_latent), max_dims) or None,
    )


def _semantic_lookup(
    gmap_vp_ids: Optional[Sequence],
    nav_logits,
    origin_nav_logits=None,
):
    if gmap_vp_ids is None or nav_logits is None:
        return {}

    nav_logits_tensor = torch.as_tensor(nav_logits, dtype=torch.float32).detach().cpu().reshape(-1)
    nav_probs_tensor = torch.softmax(nav_logits_tensor, dim=0)
    origin_logits_tensor = None
    origin_probs_tensor = None
    if origin_nav_logits is not None:
        origin_logits_tensor = torch.as_tensor(origin_nav_logits, dtype=torch.float32).detach().cpu().reshape(-1)
        origin_probs_tensor = torch.softmax(origin_logits_tensor, dim=0)

    lookup = {}
    for index, vp_id in enumerate(gmap_vp_ids):
        if vp_id is None:
            continue
        bundle = {
            "gmap_index": float(index),
            "nav_logit": float(nav_logits_tensor[index].item()),
            "nav_prob": float(nav_probs_tensor[index].item()),
        }
        if origin_logits_tensor is not None and index < origin_logits_tensor.numel():
            bundle["origin_nav_logit"] = float(origin_logits_tensor[index].item())
            bundle["origin_nav_prob"] = float(origin_probs_tensor[index].item())
        lookup[vp_id] = bundle
    return lookup


def build_stage2s_candidate_tokens(
    *,
    cand_angles: Sequence[float],
    cand_distances: Sequence[float],
    cand_img_idxes: Sequence[int],
    cand_vp_ids: Optional[Sequence] = None,
    candidate_embeddings=None,
    gmap_vp_ids: Optional[Sequence] = None,
    nav_logits=None,
    origin_nav_logits=None,
    max_candidate_local_dims: int = 16,
) -> List[CandidateToken]:
    if cand_vp_ids is None:
        cand_vp_ids = [None] * len(cand_angles)

    semantic_lookup = _semantic_lookup(
        gmap_vp_ids=gmap_vp_ids,
        nav_logits=nav_logits,
        origin_nav_logits=origin_nav_logits,
    )

    embedding_rows = None
    if candidate_embeddings is not None:
        embedding_rows = torch.as_tensor(candidate_embeddings, dtype=torch.float32).detach().cpu()

    tokens: List[CandidateToken] = []
    for index, (angle, distance, img_idx, vp_id) in enumerate(
        zip(cand_angles, cand_distances, cand_img_idxes, cand_vp_ids)
    ):
        embedding_head: List[float] = []
        embedding_norm = 0.0
        if embedding_rows is not None and index < embedding_rows.size(0):
            row = embedding_rows[index]
            embedding_head = _maybe_truncate(_flatten_numeric(row), max_candidate_local_dims)
            embedding_norm = float(row.norm().item())

        semantic_bundle = {"vp_id": "" if vp_id is None else str(vp_id)}
        semantic_bundle.update(semantic_lookup.get(vp_id, {}))

        tokens.append(
            CandidateToken(
                action_token={
                    "angle": float(angle),
                    "distance": float(distance),
                    "img_idx": float(img_idx),
                },
                candidate_local={
                    "embedding_head": embedding_head,
                    "embedding_norm": embedding_norm,
                },
                semantic_bundle=semantic_bundle,
            )
        )
    return tokens
