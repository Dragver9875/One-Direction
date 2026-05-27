from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .path_constraints import ConstraintConfig, apply_transition_constraints, build_candidate_mask
from .viterbi import ViterbiResult, viterbi_decode


@dataclass(frozen=True)
class HMMDecoderConfig:
    invalid_score: float = -1.0e9
    use_transition_constraints: bool = True
    confidence_method: str = "margin"
    constraint: ConstraintConfig = ConstraintConfig()


class HMMDecoder:
    def __init__(self, cfg: HMMDecoderConfig = HMMDecoderConfig()) -> None:
        self.cfg = cfg

    def decode(
        self,
        emission_scores: Tensor,
        transition_scores: Tensor,
        candidate_edge_idx: Tensor,
        candidate_mask: Tensor | None = None,
        transition_mask: Tensor | None = None,
    ) -> ViterbiResult:
        if candidate_mask is None:
            candidate_mask = build_candidate_mask(candidate_edge_idx)

        constrained_transition_scores = transition_scores

        if transition_mask is not None and self.cfg.use_transition_constraints:
            constrained_transition_scores = apply_transition_constraints(
                transition_scores=transition_scores,
                transition_mask=transition_mask,
                cfg=self.cfg.constraint,
            )

        return viterbi_decode(
            emission_scores=emission_scores,
            transition_scores=constrained_transition_scores,
            candidate_edge_idx=candidate_edge_idx,
            candidate_mask=candidate_mask,
            transition_mask=None,
            invalid_score=self.cfg.invalid_score,
        )

    def decode_batch(
        self,
        emission_scores: Tensor,
        transition_scores: Tensor,
        candidate_edge_idx: Tensor,
        candidate_mask: Tensor | None = None,
        transition_mask: Tensor | None = None,
    ) -> list[ViterbiResult]:
        if emission_scores.ndim != 3:
            raise ValueError("emission_scores must have shape [B, T, K].")

        results = []
        for b in range(emission_scores.shape[0]):
            cm = candidate_mask[b] if candidate_mask is not None else None
            tm = transition_mask[b] if transition_mask is not None else None
            results.append(
                self.decode(
                    emission_scores=emission_scores[b],
                    transition_scores=transition_scores[b],
                    candidate_edge_idx=candidate_edge_idx[b],
                    candidate_mask=cm,
                    transition_mask=tm,
                )
            )
        return results

    def selected_candidate_scores(
        self,
        emission_scores: Tensor,
        transition_scores: Tensor,
        result: ViterbiResult,
    ) -> tuple[Tensor, Tensor]:
        t_count = emission_scores.shape[0]
        positions = result.path_positions

        selected_emissions = emission_scores[
            torch.arange(t_count, device=emission_scores.device),
            positions,
        ]

        if t_count <= 1:
            selected_transitions = torch.empty(
                (0,),
                dtype=emission_scores.dtype,
                device=emission_scores.device,
            )
        else:
            selected_transitions = transition_scores[
                torch.arange(t_count - 1, device=transition_scores.device),
                positions[:-1],
                positions[1:],
            ]

        return selected_emissions, selected_transitions

    def path_confidence(
        self,
        emission_scores: Tensor,
        result: ViterbiResult,
    ) -> Tensor:
        positions = result.path_positions
        scores = emission_scores.clone()

        top2 = torch.topk(scores, k=min(2, scores.shape[-1]), dim=-1).values

        if top2.shape[-1] == 1:
            margin = top2[:, 0]
        else:
            margin = top2[:, 0] - top2[:, 1]

        selected_scores = scores[
            torch.arange(scores.shape[0], device=scores.device),
            positions,
        ]
        best_scores = top2[:, 0]
        is_best = torch.isclose(selected_scores, best_scores, atol=1.0e-6, rtol=1.0e-5)

        confidence = torch.sigmoid(margin)
        confidence = confidence.masked_fill(~is_best, confidence * 0.5)
        confidence = confidence.masked_fill(positions < 0, 0.0)

        return confidence
