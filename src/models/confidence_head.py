from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn


ConfidenceMethod = Literal["viterbi_margin", "softmax_emission", "path_score"]


@dataclass(frozen=True)
class ConfidenceConfig:
    method: ConfidenceMethod = "viterbi_margin"
    temperature: float = 1.0
    eps: float = 1.0e-8


class ConfidenceHead(nn.Module):
    def __init__(self, cfg: ConfidenceConfig = ConfidenceConfig()) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        emission_scores: Tensor,
        selected_candidate_positions: Tensor,
        candidate_mask: Tensor | None = None,
        path_scores: Tensor | None = None,
    ) -> Tensor:
        if self.cfg.method == "softmax_emission":
            return self.softmax_emission_confidence(
                emission_scores,
                selected_candidate_positions,
                candidate_mask,
            )

        if self.cfg.method == "viterbi_margin":
            return self.margin_confidence(
                emission_scores,
                selected_candidate_positions,
                candidate_mask,
            )

        if self.cfg.method == "path_score":
            if path_scores is None:
                raise ValueError("path_scores is required for path_score confidence.")
            return torch.sigmoid(path_scores / max(self.cfg.temperature, self.cfg.eps))

        raise ValueError(f"Unsupported confidence method: {self.cfg.method}")

    def softmax_emission_confidence(
        self,
        emission_scores: Tensor,
        selected_candidate_positions: Tensor,
        candidate_mask: Tensor | None = None,
    ) -> Tensor:
        scores = emission_scores / max(self.cfg.temperature, self.cfg.eps)
        if candidate_mask is not None:
            scores = scores.masked_fill(~candidate_mask.bool(), -1.0e9)

        probs = torch.softmax(scores, dim=-1)
        gather_idx = selected_candidate_positions.long().clamp_min(0).unsqueeze(-1)
        confidence = probs.gather(dim=-1, index=gather_idx).squeeze(-1)
        invalid = selected_candidate_positions < 0
        return confidence.masked_fill(invalid, 0.0)

    def margin_confidence(
        self,
        emission_scores: Tensor,
        selected_candidate_positions: Tensor,
        candidate_mask: Tensor | None = None,
    ) -> Tensor:
        scores = emission_scores
        if candidate_mask is not None:
            scores = scores.masked_fill(~candidate_mask.bool(), -1.0e9)

        top2 = torch.topk(scores, k=min(2, scores.shape[-1]), dim=-1).values

        if top2.shape[-1] == 1:
            margin = top2[..., 0]
        else:
            margin = top2[..., 0] - top2[..., 1]

        selected_scores = scores.gather(
            dim=-1,
            index=selected_candidate_positions.long().clamp_min(0).unsqueeze(-1),
        ).squeeze(-1)

        best_scores = top2[..., 0]
        is_best = torch.isclose(selected_scores, best_scores, atol=1.0e-6, rtol=1.0e-5)
        confidence = torch.sigmoid(margin / max(self.cfg.temperature, self.cfg.eps))
        confidence = confidence.masked_fill(~is_best, confidence * 0.5)

        invalid = selected_candidate_positions < 0
        return confidence.masked_fill(invalid, 0.0)
