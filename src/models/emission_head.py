from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import torch
from torch import Tensor, nn


ActivationName = Literal["relu", "gelu", "silu", "tanh", "identity"]
NormName = Literal["layernorm", "batchnorm", "none"]


@dataclass(frozen=True)
class EmissionHeadConfig:
    road_embedding_dim: int = 128
    scalar_feature_dim: int = 9
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.1
    activation: ActivationName = "relu"
    normalization: NormName = "layernorm"
    output_dim: int = 1


def make_activation(name: ActivationName) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "identity":
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


def make_norm(name: NormName, dim: int) -> nn.Module:
    if name == "layernorm":
        return nn.LayerNorm(dim)
    if name == "batchnorm":
        return nn.BatchNorm1d(dim)
    if name == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported normalization: {name}")


def build_mlp(
    input_dim: int,
    hidden_dims: Iterable[int],
    output_dim: int,
    dropout: float,
    activation: ActivationName,
    normalization: NormName,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(make_norm(normalization, hidden_dim))
        layers.append(make_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class EmissionHead(nn.Module):
    def __init__(self, cfg: EmissionHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        input_dim = cfg.road_embedding_dim + cfg.scalar_feature_dim
        self.mlp = build_mlp(
            input_dim=input_dim,
            hidden_dims=cfg.hidden_dims,
            output_dim=cfg.output_dim,
            dropout=cfg.dropout,
            activation=cfg.activation,
            normalization=cfg.normalization,
        )

    def forward(
        self,
        road_embeddings: Tensor,
        candidate_edge_idx: Tensor,
        emission_features: Tensor,
        candidate_mask: Tensor | None = None,
    ) -> Tensor:
        if candidate_edge_idx.dtype != torch.long:
            candidate_edge_idx = candidate_edge_idx.long()

        safe_idx = candidate_edge_idx.clamp_min(0)
        selected_embeddings = road_embeddings[safe_idx]
        x = torch.cat([selected_embeddings, emission_features.float()], dim=-1)
        scores = self.mlp(x).squeeze(-1)

        invalid_by_index = candidate_edge_idx < 0
        if candidate_mask is not None:
            invalid = ~candidate_mask.bool() | invalid_by_index
        else:
            invalid = invalid_by_index

        scores = scores.masked_fill(invalid, -1.0e9)
        return scores
