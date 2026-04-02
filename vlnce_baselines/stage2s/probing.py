from __future__ import annotations

from copy import deepcopy
from typing import Iterable, List, Optional, Sequence

from .contracts import CounterfactualOutcome



def choose_probe_indices(
    semantic_scores: Sequence[float],
    probe_width: int,
    protected_top_k: int = 2,
) -> List[int]:
    if probe_width <= 0 or len(semantic_scores) == 0:
        return []

    ranked = sorted(
        range(len(semantic_scores)),
        key=lambda idx: (float(semantic_scores[idx]), -idx),
        reverse=True,
    )
    if probe_width >= len(ranked):
        return ranked

    selected: List[int] = []
    for idx in ranked[: min(protected_top_k, probe_width)]:
        if idx not in selected:
            selected.append(idx)

    hard_negative = ranked[-1]
    if len(selected) < probe_width and hard_negative not in selected:
        selected.append(hard_negative)

    for idx in ranked:
        if len(selected) >= probe_width:
            break
        if idx not in selected:
            selected.append(idx)

    return selected



def pack_sim_snapshot(sim) -> dict:
    state = sim.get_agent_state()
    return {
        "position": deepcopy(list(state.position)),
        "rotation": deepcopy(state.rotation),
    }



def restore_sim_snapshot(sim, snapshot: dict) -> None:
    sim.set_agent_state(snapshot["position"], snapshot["rotation"])



def summarize_probe_outcome(
    intended_forward: float,
    executed_forward: float,
    start_goal_distance: Optional[float] = None,
    end_goal_distance: Optional[float] = None,
    tolerance_ratio: float = 0.8,
) -> CounterfactualOutcome:
    intended = max(float(intended_forward), 1e-6)
    executed = max(float(executed_forward), 0.0)
    reachable_ratio = round(executed / intended, 6)
    blocking_failure = 1.0 if reachable_ratio < tolerance_ratio else 0.0
    executable = 0.0 if blocking_failure else 1.0

    goal_progress_delta = None
    if start_goal_distance is not None and end_goal_distance is not None:
        goal_progress_delta = round(float(start_goal_distance) - float(end_goal_distance), 6)

    return CounterfactualOutcome(
        executable=executable,
        blocking_failure=blocking_failure,
        reachable_ratio=reachable_ratio,
        realized_displacement=round(executed, 6),
        goal_progress_delta=goal_progress_delta,
        metadata={
            "intended_forward": round(float(intended_forward), 6),
            "executed_forward": round(executed, 6),
        },
    )
