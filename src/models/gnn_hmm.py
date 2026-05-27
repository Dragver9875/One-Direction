from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .confidence_head import ConfidenceConfig, ConfidenceHead
from .emission_head import EmissionHead, EmissionHeadConfig
from .road_gnn_encoder import RoadGNNEncoder, RoadGNNEncoderConfig
from .transition_head import TransitionHead, TransitionHeadConfig


@dataclass(frozen=True)
class GNNHMMConfig:
    road_gnn: RoadGNNEncoderConfig
    emission_head: EmissionHeadConfig
    transition_head: TransitionHeadConfig
    confidence: ConfidenceConfig = ConfidenceConfig()


class GNNHMM(nn.Module):
    def __init__(self, cfg: GNNHMMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.road_encoder = RoadGNNEncoder(cfg.road_gnn)
        self.emission_head = EmissionHead(cfg.emission_head)
        self.transition_head = TransitionHead(cfg.transition_head)
        self.confidence_head = ConfidenceHead(cfg.confidence)

    def encode_roads(
        self,
        road_x_or_data: Tensor | object,
        edge_index: Tensor | None = None,
    ) -> Tensor:
        return self.road_encoder(road_x_or_data, edge_index)

    def compute_emissions(
        self,
        road_embeddings: Tensor,
        candidate_edge_idx: Tensor,
        emission_features: Tensor,
        candidate_mask: Tensor | None = None,
    ) -> Tensor:
        return self.emission_head(
            road_embeddings=road_embeddings,
            candidate_edge_idx=candidate_edge_idx,
            emission_features=emission_features,
            candidate_mask=candidate_mask,
        )

    def compute_transitions(
        self,
        road_embeddings: Tensor,
        prev_edge_idx: Tensor,
        curr_edge_idx: Tensor,
        transition_features: Tensor,
        transition_mask: Tensor | None = None,
    ) -> Tensor:
        return self.transition_head(
            road_embeddings=road_embeddings,
            prev_edge_idx=prev_edge_idx,
            curr_edge_idx=curr_edge_idx,
            transition_features=transition_features,
            transition_mask=transition_mask,
        )

    def forward(
        self,
        road_x_or_data: Tensor | object,
        candidate_edge_idx: Tensor,
        emission_features: Tensor,
        prev_edge_idx: Tensor,
        curr_edge_idx: Tensor,
        transition_features: Tensor,
        edge_index: Tensor | None = None,
        candidate_mask: Tensor | None = None,
        transition_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        road_embeddings = self.encode_roads(road_x_or_data, edge_index)

        emission_scores = self.compute_emissions(
            road_embeddings=road_embeddings,
            candidate_edge_idx=candidate_edge_idx,
            emission_features=emission_features,
            candidate_mask=candidate_mask,
        )

        transition_scores = self.compute_transitions(
            road_embeddings=road_embeddings,
            prev_edge_idx=prev_edge_idx,
            curr_edge_idx=curr_edge_idx,
            transition_features=transition_features,
            transition_mask=transition_mask,
        )

        return {
            "road_embeddings": road_embeddings,
            "emission_scores": emission_scores,
            "transition_scores": transition_scores,
        }

    @torch.no_grad()
    def score_for_decoding(
        self,
        road_x_or_data: Tensor | object,
        candidate_edge_idx: Tensor,
        emission_features: Tensor,
        prev_edge_idx: Tensor,
        curr_edge_idx: Tensor,
        transition_features: Tensor,
        edge_index: Tensor | None = None,
        candidate_mask: Tensor | None = None,
        transition_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        self.eval()
        return self.forward(
            road_x_or_data=road_x_or_data,
            candidate_edge_idx=candidate_edge_idx,
            emission_features=emission_features,
            prev_edge_idx=prev_edge_idx,
            curr_edge_idx=curr_edge_idx,
            transition_features=transition_features,
            edge_index=edge_index,
            candidate_mask=candidate_mask,
            transition_mask=transition_mask,
        )
