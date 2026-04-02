from __future__ import annotations

from typing import Iterable

import torch


def masked_binary_accuracy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    predictions = (torch.sigmoid(logits) >= 0.5).float()
    correct = (predictions == target.float()).float()
    mask = mask.float()
    denom = mask.sum().clamp_min(1.0)
    return (correct * mask).sum() / denom


def masked_pearsonr(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    if mask.sum() < 2:
        return pred.new_zeros(())
    pred_valid = pred[mask]
    target_valid = target[mask]
    pred_centered = pred_valid - pred_valid.mean()
    target_centered = target_valid - target_valid.mean()
    denom = pred_centered.norm() * target_centered.norm()
    if denom <= 0:
        return pred.new_zeros(())
    return (pred_centered * target_centered).sum() / denom


def intervention_rate(changed_flags: Iterable[bool]) -> float:
    changed_flags = list(changed_flags)
    if not changed_flags:
        return 0.0
    return float(sum(bool(flag) for flag in changed_flags) / len(changed_flags))
