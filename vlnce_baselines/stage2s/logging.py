from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from .contracts import CandidateRecord, CandidateSetRecord, StructuredLatentState



def build_candidate_set_record(
    candidate_set_id: str,
    latent_state: Optional[StructuredLatentState],
    candidates: Iterable[CandidateRecord],
    episode_id: str,
    step_id: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> CandidateSetRecord:
    if not candidate_set_id:
        raise ValueError("candidate_set_id is required")
    return CandidateSetRecord(
        episode_id=episode_id,
        step_id=step_id,
        candidate_set_id=candidate_set_id,
        latent_state=latent_state,
        candidates=list(candidates),
        metadata=dict(metadata or {}),
    )
