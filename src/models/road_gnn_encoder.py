from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

try:
    from torch_geometric.nn import GATConv, SAGEConv
except ImportError as exc:
    raise ImportError(
        "torch-geometric is required for RoadGNNEncoder. Install torch-geometric."
    ) from exc


ActivationName = Literal["relu", "gelu", "silu", "tanh", "identity"]
NormName = Literal["layernorm", "batchnorm", "none"]
GNNType = Literal["graphsage", "gat"]


@dataclass(frozen=True)
class RoadGNNEncoderConfig:
    input_dim: int
    hidden_dim: int = 128
    output_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    gnn_type: GNNType = "graphsage"
    activation: ActivationName = "relu"
    normalization: NormName = "layernorm"
    sage_aggr: str = "mean"
    gat_heads: int = 4
    gat_concat: bool = True
    gat_negative_slope: float = 0.2


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


class RoadGNNEncoder(nn.Module):
    def __init__(self, cfg: RoadGNNEncoderConfig) -> None:
        super().__init__()
        if cfg.num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.cfg = cfg
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activation = make_activation(cfg.activation)
        self.dropout = nn.Dropout(cfg.dropout)

        in_dim = cfg.input_dim

        for layer_idx in range(cfg.num_layers):
            is_last = layer_idx == cfg.num_layers - 1
            out_dim = cfg.output_dim if is_last else cfg.hidden_dim

            if cfg.gnn_type == "graphsage":
                conv = SAGEConv(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    aggr=cfg.sage_aggr,
                )
                conv_out_dim = out_dim
            elif cfg.gnn_type == "gat":
                if cfg.gat_concat:
                    per_head_dim = max(1, out_dim // cfg.gat_heads)
                    conv = GATConv(
                        in_channels=in_dim,
                        out_channels=per_head_dim,
                        heads=cfg.gat_heads,
                        concat=True,
                        dropout=cfg.dropout,
                        negative_slope=cfg.gat_negative_slope,
                    )
                    conv_out_dim = per_head_dim * cfg.gat_heads
                else:
                    conv = GATConv(
                        in_channels=in_dim,
                        out_channels=out_dim,
                        heads=cfg.gat_heads,
                        concat=False,
                        dropout=cfg.dropout,
                        negative_slope=cfg.gat_negative_slope,
                    )
                    conv_out_dim = out_dim
            else:
                raise ValueError(f"Unsupported gnn_type: {cfg.gnn_type}")

            self.layers.append(conv)
            self.norms.append(make_norm(cfg.normalization, conv_out_dim))
            in_dim = conv_out_dim

        if in_dim != cfg.output_dim:
            self.output_projection = nn.Linear(in_dim, cfg.output_dim)
        else:
            self.output_projection = nn.Identity()

    def forward(
        self,
        x_or_data: Tensor | object,
        edge_index: Tensor | None = None,
    ) -> Tensor:
        if edge_index is None:
            if not hasattr(x_or_data, "x") or not hasattr(x_or_data, "edge_index"):
                raise ValueError("Pass either (x, edge_index) or a PyG Data-like object.")
            x = getattr(x_or_data, "x")
            edge_index = getattr(x_or_data, "edge_index")
        else:
            x = x_or_data

        if not isinstance(x, Tensor):
            raise TypeError("x must be a torch.Tensor.")
        if not isinstance(edge_index, Tensor):
            raise TypeError("edge_index must be a torch.Tensor.")

        h = x.float()

        for idx, layer in enumerate(self.layers):
            h = layer(h, edge_index)
            h = self.norms[idx](h)
            is_last = idx == len(self.layers) - 1
            if not is_last:
                h = self.activation(h)
                h = self.dropout(h)

        h = self.output_projection(h)
        return h
