from __future__ import annotations

from typing import Dict, Iterable

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


def _flatten_masked(values: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    valid = mask.bool().reshape(-1)
    values = values.reshape(-1)[valid]
    target = target.reshape(-1)[valid]
    return values.float(), target.float()


def binary_auc_ap(scores: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    scores, target = _flatten_masked(scores, target, mask)
    if scores.numel() == 0:
        return {"auroc": 0.5, "ap": 0.0}

    positives = target.sum().item()
    negatives = target.numel() - positives
    if positives == 0 or negatives == 0:
        return {"auroc": 0.5, "ap": float(positives / max(target.numel(), 1))}

    sorted_indices = torch.argsort(scores, descending=True)
    sorted_target = target[sorted_indices]
    tps = torch.cumsum(sorted_target, dim=0)
    fps = torch.cumsum(1.0 - sorted_target, dim=0)
    tpr = tps / positives
    fpr = fps / negatives
    zero = torch.zeros(1, dtype=tpr.dtype, device=tpr.device)
    tpr = torch.cat([zero, tpr])
    fpr = torch.cat([zero, fpr])
    auroc = torch.trapz(tpr, fpr).item()

    precision = tps / torch.arange(1, sorted_target.numel() + 1, dtype=torch.float32)
    ap = (precision * sorted_target).sum().item() / positives
    return {"auroc": float(auroc), "ap": float(ap)}


def regression_stats(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    pred_valid, target_valid = _flatten_masked(pred, target, mask)
    if pred_valid.numel() == 0:
        return {"mae": 0.0, "corr": 0.0}
    mae = torch.mean(torch.abs(pred_valid - target_valid)).item()
    corr = masked_pearsonr(pred, target, mask).item()
    return {"mae": float(mae), "corr": float(corr)}


def ranking_top1_accuracy(scores: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
        target = target.unsqueeze(0)
        mask = mask.unsqueeze(0)
    hits = []
    for row_scores, row_target, row_mask in zip(scores, target, mask):
        valid = row_mask.bool()
        if valid.sum() == 0:
            continue
        valid_scores = row_scores[valid]
        valid_target = row_target[valid]
        pred_index = int(torch.argmax(valid_scores).item())
        target_index = int(torch.argmax(valid_target).item())
        hits.append(1.0 if pred_index == target_index else 0.0)
    if not hits:
        return 0.0
    return float(sum(hits) / len(hits))


def expected_calibration_error(probabilities: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, n_bins: int = 10) -> float:
    probs, target = _flatten_masked(probabilities, target, mask)
    if probs.numel() == 0:
        return 0.0
    boundaries = torch.linspace(0.0, 1.0, steps=n_bins + 1)
    total = probs.numel()
    ece = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        if upper == 1.0:
            in_bin = (probs >= lower) & (probs <= upper)
        else:
            in_bin = (probs >= lower) & (probs < upper)
        if in_bin.sum() == 0:
            continue
        bin_prob = probs[in_bin].mean().item()
        bin_acc = target[in_bin].mean().item()
        ece += (in_bin.sum().item() / total) * abs(bin_acc - bin_prob)
    return float(ece)
