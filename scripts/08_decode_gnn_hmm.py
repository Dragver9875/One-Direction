from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv, SAGEConv
except ImportError as exc:
    raise ImportError("torch-geometric is required.") from exc


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class RoadGNNEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float, gnn_type: str = "graphsage"):
        super().__init__()
        self.dropout = dropout
        dims = [input_dim] + [hidden_dim] * max(num_layers - 1, 0) + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(GATConv(dims[i], dims[i + 1], heads=1, concat=False) if gnn_type == "gat" else SAGEConv(dims[i], dims[i + 1]))
        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in dims[1:]])

    def forward(self, x, edge_index):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index)
            h = self.norms[i](h)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers += [nn.LayerNorm(dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class OneDirectionModel(nn.Module):
    def __init__(self, road_feat_dim: int, road_hidden_dim: int, emission_feat_dim: int, transition_feat_dim: int, gnn_type: str = "graphsage"):
        super().__init__()
        self.encoder = RoadGNNEncoder(road_feat_dim, road_hidden_dim, road_hidden_dim, 2, 0.1, gnn_type)
        self.emission_head = MLP(road_hidden_dim + emission_feat_dim, [128, 64], 1, 0.1)
        self.transition_head = MLP(2 * road_hidden_dim + transition_feat_dim, [128, 64], 1, 0.1)

    def encode_roads(self, road_x, edge_index):
        return self.encoder(road_x, edge_index)

    def score_emissions(self, road_emb, item):
        edge_idx = item["candidate_edge_idx"].to(road_emb.device)
        mask = item["candidate_mask"].to(road_emb.device)
        feats = item["emission_features"].to(road_emb.device)
        emb = road_emb[edge_idx.clamp_min(0)]
        logits = self.emission_head(torch.cat([emb, feats], dim=-1))
        return logits.masked_fill(~mask, -1e9)

    def score_transitions(self, road_emb, item):
        edge_idx = item["candidate_edge_idx"].to(road_emb.device)
        trans_feats = item["transition_features"].to(road_emb.device)
        trans_mask = item["transition_mask"].to(road_emb.device)
        if trans_feats.numel() == 0:
            return torch.empty(0, device=road_emb.device)
        prev = road_emb[edge_idx[:-1].clamp_min(0)]
        curr = road_emb[edge_idx[1:].clamp_min(0)]
        Tm1, K, H = prev.shape
        prev_ex = prev[:, :, None, :].expand(Tm1, K, K, H)
        curr_ex = curr[:, None, :, :].expand(Tm1, K, K, H)
        logits = self.transition_head(torch.cat([prev_ex, curr_ex, trans_feats], dim=-1))
        return logits.masked_fill(~trans_mask, -1e9)


def viterbi_decode(emissions: torch.Tensor, transitions: torch.Tensor) -> tuple[list[int], list[float], list[float]]:
    T, K = emissions.shape
    if T == 0:
        return [], [], []
    dp = emissions[0].clone()
    back = []
    trans_chosen = [0.0]
    for t in range(1, T):
        scores = dp[:, None] + transitions[t - 1]
        best_score, best_prev = scores.max(dim=0)
        dp = emissions[t] + best_score
        back.append(best_prev)
    last = int(dp.argmax().item())
    path = [last]
    for b in reversed(back):
        prev = int(b[last].item())
        path.append(prev)
        last = prev
    path = list(reversed(path))
    em_scores = [float(emissions[t, c].detach().cpu().item()) for t, c in enumerate(path)]
    tr_scores = [0.0]
    for t in range(1, len(path)):
        tr_scores.append(float(transitions[t - 1, path[t - 1], path[t]].detach().cpu().item()))
    return path, em_scores, tr_scores


def softmax_confidence(emissions: torch.Tensor, path: list[int]) -> list[float]:
    probs = torch.softmax(emissions, dim=-1)
    return [float(probs[t, c].detach().cpu().item()) for t, c in enumerate(path)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/gnn_hmm_best.pt"))
    parser.add_argument("--line-graph", type=Path, default=Path("data/processed/line_graph/line_graph.pt"))
    parser.add_argument("--idx-to-edge-id", type=Path, default=Path("data/processed/line_graph/idx_to_edge_id.json"))
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/tensors/test_dataset.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/matches/gnn_hmm_matches.parquet"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    ensure_dir(args.output.parent)

    graph = torch.load(args.line_graph, map_location="cpu")
    road_x = graph["x"].float().to(device)
    edge_index = graph["edge_index"].long().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = OneDirectionModel(
        ckpt["road_feat_dim"], ckpt["hidden_dim"], ckpt["emission_feat_dim"], ckpt["transition_feat_dim"], ckpt.get("gnn_type", "graphsage")
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    dataset = torch.load(args.dataset, map_location="cpu")
    with args.idx_to_edge_id.open("r", encoding="utf-8") as f:
        idx_to_edge_id = json.load(f)

    rows = []
    with torch.no_grad():
        road_emb = model.encode_roads(road_x, edge_index)
        for item in dataset:
            emissions = model.score_emissions(road_emb, item)
            transitions = model.score_transitions(road_emb, item)
            if transitions.numel() == 0:
                path = emissions.argmax(dim=-1).detach().cpu().tolist()
                em_scores = [float(emissions[t, c].cpu()) for t, c in enumerate(path)]
                tr_scores = [0.0] * len(path)
            else:
                path, em_scores, tr_scores = viterbi_decode(emissions, transitions)
            conf = softmax_confidence(emissions, path)
            cand_edges = item["candidate_edge_idx"]
            cand_proj = item["candidate_proj_xy"]
            gt_edges = item["gt_edge_idx"]
            gt_proj = item["gt_proj_xy"]
            for t, cand_pos in enumerate(path):
                pred_idx = int(cand_edges[t, cand_pos].item())
                pred_edge_id = idx_to_edge_id.get(str(pred_idx), str(pred_idx))
                rows.append(
                    {
                        "trajectory_id": int(item["trajectory_id"]),
                        "t": int(item["timesteps"][t].item()),
                        "timestamp": item["timestamps"][t],
                        "pred_edge_idx": pred_idx,
                        "pred_edge_id": pred_edge_id,
                        "pred_proj_x": float(cand_proj[t, cand_pos, 0].item()),
                        "pred_proj_y": float(cand_proj[t, cand_pos, 1].item()),
                        "confidence": conf[t],
                        "emission_score": em_scores[t],
                        "transition_score": tr_scores[t],
                        "total_path_score": em_scores[t] + tr_scores[t],
                        "gt_edge_idx": int(gt_edges[t].item()),
                        "gt_proj_x": float(gt_proj[t, 0].item()),
                        "gt_proj_y": float(gt_proj[t, 1].item()),
                    }
                )

    out = pd.DataFrame(rows)
    out.to_parquet(args.output, index=False)
    print(f"[OK] Decoded {out['trajectory_id'].nunique()} trajectories, {len(out)} points")
    print(f"[OK] Output: {args.output}")


if __name__ == "__main__":
    main()
