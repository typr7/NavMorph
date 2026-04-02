from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


CONTRACT_VERSION = "stage2s.v1"


@dataclass
class StructuredLatentState:
    version: str = CONTRACT_VERSION
    history_latent: List[float] = field(default_factory=list)
    stochastic_latent: Optional[List[float]] = None
    memory_latent: Optional[List[float]] = None
    global_latent: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "history_latent": list(self.history_latent),
            "stochastic_latent": self.stochastic_latent,
            "memory_latent": self.memory_latent,
            "global_latent": self.global_latent,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["StructuredLatentState"]:
        if data is None:
            return None
        return cls(
            version=data.get("version", CONTRACT_VERSION),
            history_latent=list(data.get("history_latent", [])),
            stochastic_latent=data.get("stochastic_latent"),
            memory_latent=data.get("memory_latent"),
            global_latent=data.get("global_latent"),
        )


@dataclass
class CandidateToken:
    version: str = CONTRACT_VERSION
    action_token: Dict[str, Any] = field(default_factory=dict)
    candidate_local: Dict[str, Any] = field(default_factory=dict)
    semantic_bundle: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "action_token": dict(self.action_token),
            "candidate_local": dict(self.candidate_local),
            "semantic_bundle": dict(self.semantic_bundle),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["CandidateToken"]:
        if data is None:
            return None
        return cls(
            version=data.get("version", CONTRACT_VERSION),
            action_token=dict(data.get("action_token", {})),
            candidate_local=dict(data.get("candidate_local", {})),
            semantic_bundle=dict(data.get("semantic_bundle", {})),
        )


@dataclass
class CounterfactualOutcome:
    version: str = CONTRACT_VERSION
    executable: Optional[float] = None
    blocking_failure: Optional[float] = None
    reachable_ratio: Optional[float] = None
    realized_displacement: Optional[float] = None
    goal_progress_delta: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "executable": self.executable,
            "blocking_failure": self.blocking_failure,
            "reachable_ratio": self.reachable_ratio,
            "realized_displacement": self.realized_displacement,
            "goal_progress_delta": self.goal_progress_delta,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["CounterfactualOutcome"]:
        if data is None:
            return None
        return cls(
            version=data.get("version", CONTRACT_VERSION),
            executable=data.get("executable"),
            blocking_failure=data.get("blocking_failure"),
            reachable_ratio=data.get("reachable_ratio"),
            realized_displacement=data.get("realized_displacement"),
            goal_progress_delta=data.get("goal_progress_delta"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class CandidateRecord:
    version: str = CONTRACT_VERSION
    candidate_index: int = 0
    token: Optional[CandidateToken] = None
    outcome: Optional[CounterfactualOutcome] = None
    successor_latent: Optional[StructuredLatentState] = None
    rollout_summary: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "candidate_index": self.candidate_index,
            "token": None if self.token is None else self.token.to_dict(),
            "outcome": None if self.outcome is None else self.outcome.to_dict(),
            "successor_latent": None if self.successor_latent is None else self.successor_latent.to_dict(),
            "rollout_summary": self.rollout_summary,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateRecord":
        return cls(
            version=data.get("version", CONTRACT_VERSION),
            candidate_index=int(data.get("candidate_index", 0)),
            token=CandidateToken.from_dict(data.get("token")),
            outcome=CounterfactualOutcome.from_dict(data.get("outcome")),
            successor_latent=StructuredLatentState.from_dict(data.get("successor_latent")),
            rollout_summary=data.get("rollout_summary"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class CandidateSetRecord:
    version: str = CONTRACT_VERSION
    episode_id: str = ""
    step_id: int = 0
    candidate_set_id: str = ""
    latent_state: Optional[StructuredLatentState] = None
    candidates: List[CandidateRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "candidate_set_id": self.candidate_set_id,
            "latent_state": None if self.latent_state is None else self.latent_state.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateSetRecord":
        return cls(
            version=data.get("version", CONTRACT_VERSION),
            episode_id=data.get("episode_id", ""),
            step_id=int(data.get("step_id", 0)),
            candidate_set_id=data.get("candidate_set_id", ""),
            latent_state=StructuredLatentState.from_dict(data.get("latent_state")),
            candidates=[CandidateRecord.from_dict(candidate) for candidate in data.get("candidates", [])],
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class BranchScore:
    version: str = CONTRACT_VERSION
    candidate_set_id: str = ""
    candidate_index: int = 0
    score: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "candidate_set_id": self.candidate_set_id,
            "candidate_index": self.candidate_index,
            "score": self.score,
            "components": dict(self.components),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BranchScore":
        return cls(
            version=data.get("version", CONTRACT_VERSION),
            candidate_set_id=data.get("candidate_set_id", ""),
            candidate_index=int(data.get("candidate_index", 0)),
            score=float(data.get("score", 0.0)),
            components=dict(data.get("components", {})),
            metadata=dict(data.get("metadata", {})),
        )
