from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

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


def append_candidate_set_record(
    path: Union[str, Path],
    record: CandidateSetRecord,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    open_fn = gzip.open if path.suffix == ".gz" else open
    with open_fn(path, "at") as f:
        f.write(json.dumps(record.to_dict()) + "\n")


def load_candidate_set_records(
    path: Union[str, Path],
) -> List[CandidateSetRecord]:
    path = Path(path)
    if not path.exists():
        return []
    open_fn = gzip.open if path.suffix == ".gz" else open
    records: List[CandidateSetRecord] = []
    with open_fn(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(CandidateSetRecord.from_dict(json.loads(line)))
    return records
