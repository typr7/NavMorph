from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
from torch import nn


class Stage2SModel(nn.Module):
    """Isolated candidate-conditioned latent branch model for Stage 2S.

    This module intentionally stays independent of Habitat/NavMorph runtime code.
    It consumes already-exported latent states and candidate tokens and predicts
    one-step latent transitions plus factorized physical / semantic heads.
    """

    def __init__(
        self,
        latent_dim: int,
        candidate_dim: int,
        hidden_dim: int,
        rollout_depth: int = 1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.candidate_dim = candidate_dim
        self.hidden_dim = hidden_dim
        self.rollout_depth = rollout_depth

        self.state_adapter = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.next_latent_head = nn.Linear(hidden_dim, latent_dim)
        self.exec_head = nn.Linear(hidden_dim, 1)
        self.progress_head = nn.Linear(hidden_dim, 1)
        self.blocking_head = nn.Linear(hidden_dim, 1)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)
        self.semantic_head = nn.Linear(hidden_dim, 1)

    def _prepare_state(self, state_latent: torch.Tensor, num_candidates: int) -> torch.Tensor:
        if state_latent.dim() == 2:
            return state_latent.unsqueeze(1).expand(-1, num_candidates, -1)
        if state_latent.dim() != 3:
            raise ValueError(f"Expected state_latent dim 2 or 3, got {tuple(state_latent.shape)}")
        if state_latent.size(1) == 1 and num_candidates > 1:
            return state_latent.expand(-1, num_candidates, -1)
        if state_latent.size(1) != num_candidates:
            raise ValueError(
                "state_latent candidate axis must match candidate_token candidate axis "
                f"({state_latent.size(1)} != {num_candidates})"
            )
        return state_latent

    def _candidate_hidden(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        candidate_token = batch["candidate_token"].float()
        state_latent = self._prepare_state(batch["state_latent"].float(), candidate_token.size(1))
        state_hidden = self.state_adapter(state_latent)
        candidate_hidden = self.candidate_encoder(candidate_token)
        return self.fusion(torch.cat([state_hidden, candidate_hidden], dim=-1))

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        hidden = self._candidate_hidden(batch)
        outputs = {
            "hidden": hidden,
            "next_latent": self.next_latent_head(hidden),
            "exec_logit": self.exec_head(hidden).squeeze(-1),
            "progress_mean": self.progress_head(hidden).squeeze(-1),
            "blocking_logit": self.blocking_head(hidden).squeeze(-1),
            "uncertainty_logit": self.uncertainty_head(hidden).squeeze(-1),
            "semantic_logit": self.semantic_head(hidden).squeeze(-1),
        }
        return outputs

    def rollout(self, state_latent: torch.Tensor, candidate_tokens: Iterable[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        current_state = state_latent
        rollout_outputs: List[Dict[str, torch.Tensor]] = []
        for step_index, candidate_token in enumerate(candidate_tokens):
            if step_index >= self.rollout_depth:
                break
            outputs = self.forward({
                "state_latent": current_state,
                "candidate_token": candidate_token,
            })
            rollout_outputs.append(outputs)
            current_state = outputs["next_latent"]
        return rollout_outputs
