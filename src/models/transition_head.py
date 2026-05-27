from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import torch
from torch import Tensor, nn


ActivationName = Literal["relu", "gelu", "silu", "tanh", "identity"]
NormName = Literal["layernorm", "batchnorm", "none"]


@dataclass(frozen=True)
class TransitionHeadConfig:
    road_embedding_dim: int = 128
    scalar_feature_dim: int = 10
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.1
    activation: ActivationName = "relu"
    normalization: NormName = "layernorm"
    output_dim: int = 1
    illegal_transition_score: float = -1.0e9


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


class TransitionHead(nn.Module):
    def __init__(self, cfg: TransitionHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        input_dim = (2 * cfg.road_embedding_dim) + cfg.scalar_feature_dim
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
        prev_edge_idx: Tensor,
        curr_edge_idx: Tensor,
        transition_features: Tensor,
        transition_mask: Tensor | None = None,
    ) -> Tensor:
        if prev_edge_idx.dtype != torch.long:
            prev_edge_idx = prev_edge_idx.long()
        if curr_edge_idx.dtype != torch.long:
            curr_edge_idx = curr_edge_idx.long()

        safe_prev = prev_edge_idx.clamp_min(0)
        safe_curr = curr_edge_idx.clamp_min(0)

        prev_emb = road_embeddings[safe_prev]
        curr_emb = road_embeddings[safe_curr]

        x = torch.cat([prev_emb, curr_emb, transition_features.float()], dim=-1)
        scores = self.mlp(x).squeeze(-1)

        invalid_by_index = (prev_edge_idx < 0) | (curr_edge_idx < 0)
        if transition_mask is not None:
            invalid = ~transition_mask.bool() | invalid_by_index
        else:
            invalid = invalid_by_index

        scores = scores.masked_fill(invalid, self.cfg.illegal_transition_score)
        return scores
