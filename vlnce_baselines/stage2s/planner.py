from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .calibration import CalibratedBranchAggregator
from .dataset import build_group_tensor_sample
from .model import Stage2SModel


class Stage2SBranchPlanner:
    def __init__(
        self,
        top_k: int = 2,
        depth: int = 2,
        aggregator: Optional[CalibratedBranchAggregator] = None,
        checkpoint_path: str = "",
        device: str = "cpu",
    ) -> None:
        self.top_k = int(top_k)
        self.depth = int(depth)
        self.aggregator = aggregator or CalibratedBranchAggregator()
        self.device = torch.device(device)
        self.model: Optional[Stage2SModel] = None
        self.model_args: Dict[str, int] = {}
        self.checkpoint_path = checkpoint_path
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        payload = torch.load(checkpoint_path, map_location=self.device)
        args = payload.get("args", {})
        self.model_args = {
            "latent_dim": int(args.get("latent_dim", 32)),
            "candidate_dim": int(args.get("candidate_dim", 16)),
            "hidden_dim": int(args.get("hidden_dim", 64)),
            "rollout_depth": int(args.get("rollout_depth", 1)),
        }
        self.model = Stage2SModel(**self.model_args)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.checkpoint_path = checkpoint_path

    @staticmethod
    def _shortlist_semantic(candidate: Dict) -> float:
        return float(candidate.get("semantic_base", candidate.get("semantic", 0.0)))

    @staticmethod
    def _aggregate_candidate_score(candidate: Dict, aggregator: CalibratedBranchAggregator) -> float:
        return aggregator(
            exec_prob=float(candidate.get("exec_prob", 0.0)),
            progress=float(candidate.get("progress", 0.0)),
            blocking=float(candidate.get("blocking", 0.0)),
            uncertainty=float(candidate.get("uncertainty", 0.0)),
            semantic=float(candidate.get("semantic", 0.0)),
        )

    def select(self, branch_bundle: Dict) -> Dict:
        candidates = list(branch_bundle.get("candidates", []))
        if not candidates:
            raise ValueError("branch_bundle must contain at least one candidate")

        ranked = sorted(candidates, key=self._shortlist_semantic, reverse=True)
        shortlist = ranked[: max(1, min(self.top_k, len(ranked)))]
        baseline_index = int(branch_bundle.get("baseline_index", ranked[0]["candidate_index"]))
        branch_scores: List[Dict] = []
        for candidate in shortlist:
            score = self._aggregate_candidate_score(candidate, self.aggregator)
            branch_scores.append(
                {
                    "candidate_index": int(candidate["candidate_index"]),
                    "gmap_index": candidate.get("gmap_index"),
                    "score": float(score),
                    "semantic": float(candidate.get("semantic", 0.0)),
                    "semantic_base": float(self._shortlist_semantic(candidate)),
                    "exec_prob": float(candidate.get("exec_prob", 0.0)),
                    "progress": float(candidate.get("progress", 0.0)),
                    "blocking": float(candidate.get("blocking", 0.0)),
                    "uncertainty": float(candidate.get("uncertainty", 0.0)),
                }
            )

        selected_branch = max(branch_scores, key=lambda item: item["score"])
        baseline_branch = next(
            (item for item in branch_scores if item["candidate_index"] == baseline_index),
            None,
        )
        if baseline_branch is None:
            baseline_candidate = next(
                candidate for candidate in candidates if int(candidate["candidate_index"]) == baseline_index
            )
            baseline_branch = {
                "candidate_index": baseline_index,
                "gmap_index": baseline_candidate.get("gmap_index"),
                "score": float(self._aggregate_candidate_score(baseline_candidate, self.aggregator)),
            }

        return {
            "selected_index": int(selected_branch["candidate_index"]),
            "selected_gmap_index": selected_branch.get("gmap_index"),
            "baseline_index": int(baseline_index),
            "baseline_gmap_index": baseline_branch.get("gmap_index"),
            "selected_score": float(selected_branch["score"]),
            "baseline_score": float(baseline_branch["score"]),
            "changed": int(selected_branch["candidate_index"]) != int(baseline_index),
            "branch_scores": branch_scores,
            "depth": self.depth,
            "top_k": self.top_k,
        }

    def _bundle_from_record(self, candidate_set_record) -> Dict:
        candidates: List[Dict] = []
        if self.model is not None:
            sample = build_group_tensor_sample(
                candidate_set_record,
                latent_dim=int(self.model_args.get("latent_dim", self.model.latent_dim)),
                candidate_dim=int(self.model_args.get("candidate_dim", self.model.candidate_dim)),
            )
            batch = {
                key: value.unsqueeze(0).to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in sample.items()
            }
            with torch.no_grad():
                outputs = self.model(batch)
            exec_probs = torch.sigmoid(outputs["exec_logit"][0]).detach().cpu()
            progress_means = outputs["progress_mean"][0].detach().cpu()
            blocking_probs = torch.sigmoid(outputs["blocking_logit"][0]).detach().cpu()
            uncertainty_probs = torch.sigmoid(outputs["uncertainty_logit"][0]).detach().cpu()
            semantic_scores = outputs["semantic_logit"][0].detach().cpu()
        else:
            exec_probs = progress_means = blocking_probs = uncertainty_probs = semantic_scores = None

        for index, candidate_record in enumerate(candidate_set_record.candidates):
            semantic_bundle = {}
            if candidate_record.token is not None:
                semantic_bundle = candidate_record.token.semantic_bundle or {}
            base_semantic = float(
                semantic_bundle.get(
                    "nav_logit",
                    semantic_bundle.get("origin_nav_logit", semantic_bundle.get("rank_proxy", 0.0)),
                )
            )
            if semantic_scores is None:
                semantic_score = base_semantic
                exec_prob = float(semantic_bundle.get("exec_prob", 0.0))
                progress = float(semantic_bundle.get("progress", 0.0))
                blocking = float(semantic_bundle.get("blocking", 0.0))
                uncertainty = float(semantic_bundle.get("uncertainty", 0.0))
            else:
                semantic_score = float(semantic_scores[index].item())
                exec_prob = float(exec_probs[index].item())
                progress = float(progress_means[index].item())
                blocking = float(blocking_probs[index].item())
                uncertainty = float(uncertainty_probs[index].item())
            candidates.append(
                {
                    "candidate_index": int(candidate_record.candidate_index),
                    "gmap_index": semantic_bundle.get("gmap_index"),
                    "semantic": semantic_score,
                    "semantic_base": base_semantic,
                    "exec_prob": exec_prob,
                    "progress": progress,
                    "blocking": blocking,
                    "uncertainty": uncertainty,
                }
            )

        baseline_index = max(candidates, key=lambda item: item.get("semantic_base", item.get("semantic", 0.0)))[
            "candidate_index"
        ]
        return {
            "candidate_set_id": candidate_set_record.candidate_set_id,
            "baseline_index": baseline_index,
            "candidates": candidates,
        }

    def select_record(self, candidate_set_record) -> Dict:
        branch_bundle = self._bundle_from_record(candidate_set_record)
        decision = self.select(branch_bundle)
        decision["candidate_set_id"] = candidate_set_record.candidate_set_id
        return decision
