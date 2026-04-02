from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _resolve_mask(batch: Dict[str, torch.Tensor], key: str, fallback_like: Optional[torch.Tensor] = None) -> torch.Tensor:
    if key in batch:
        return batch[key].float()
    if fallback_like is None:
        raise KeyError(f"Missing required mask key: {key}")
    return torch.ones_like(fallback_like, dtype=torch.float32)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def latent_transition_loss(pred_next: torch.Tensor, target_next: torch.Tensor, has_successor: torch.Tensor) -> torch.Tensor:
    per_candidate = (pred_next - target_next).pow(2).mean(dim=-1)
    return _masked_mean(per_candidate, has_successor)


def pointwise_regression_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _masked_mean((pred - target).pow(2), mask)


def pointwise_bce_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_candidate = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
    return _masked_mean(per_candidate, mask)


def pairwise_ranking_loss(scores: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    positive_pairs = ((target.unsqueeze(-1) > target.unsqueeze(-2)).float() * mask.unsqueeze(-1) * mask.unsqueeze(-2))
    if positive_pairs.sum() <= 0:
        return scores.new_zeros(())
    margins = scores.unsqueeze(-1) - scores.unsqueeze(-2)
    pair_losses = F.softplus(-margins)
    return (pair_losses * positive_pairs).sum() / positive_pairs.sum().clamp_min(1.0)


DEFAULT_LOSS_WEIGHTS = {
    "latent": 1.0,
    "exec": 1.0,
    "progress": 1.0,
    "blocking": 1.0,
    "semantic": 0.5,
    "ranking": 0.5,
    "uncertainty": 0.0,
}


def stage2s_loss_terms(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    weights = {**DEFAULT_LOSS_WEIGHTS, **(weights or {})}
    candidate_mask = _resolve_mask(batch, "candidate_mask", fallback_like=outputs["exec_logit"])
    has_successor = _resolve_mask(batch, "has_successor", fallback_like=outputs["exec_logit"])

    latent_loss = latent_transition_loss(outputs["next_latent"], batch["next_latent"].float(), has_successor)
    exec_loss = pointwise_bce_loss(outputs["exec_logit"], batch["exec_target"], candidate_mask)
    progress_loss = pointwise_regression_loss(outputs["progress_mean"], batch["progress_target"].float(), candidate_mask)
    blocking_loss = pointwise_bce_loss(outputs["blocking_logit"], batch["blocking_target"], candidate_mask)
    semantic_loss = pointwise_regression_loss(outputs["semantic_logit"], batch["semantic_target"].float(), candidate_mask)
    ranking_loss = pairwise_ranking_loss(outputs["semantic_logit"], batch["semantic_target"].float(), candidate_mask)

    if "uncertainty_target" in batch:
        uncertainty_loss = pointwise_bce_loss(
            outputs["uncertainty_logit"],
            batch["uncertainty_target"],
            candidate_mask,
        )
    else:
        uncertainty_loss = outputs["uncertainty_logit"].new_zeros(())

    total_loss = (
        weights["latent"] * latent_loss
        + weights["exec"] * exec_loss
        + weights["progress"] * progress_loss
        + weights["blocking"] * blocking_loss
        + weights["semantic"] * semantic_loss
        + weights["ranking"] * ranking_loss
        + weights["uncertainty"] * uncertainty_loss
    )

    return {
        "total_loss": total_loss,
        "latent_loss": latent_loss,
        "exec_loss": exec_loss,
        "progress_loss": progress_loss,
        "blocking_loss": blocking_loss,
        "semantic_loss": semantic_loss,
        "ranking_loss": ranking_loss,
        "uncertainty_loss": uncertainty_loss,
    }


def build_stage2s_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    weights: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    return stage2s_loss_terms(outputs, batch, weights=weights)["total_loss"]
