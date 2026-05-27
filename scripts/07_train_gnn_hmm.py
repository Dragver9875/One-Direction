from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from torch_geometric.nn import GATConv, SAGEConv
except ImportError as exc:
    raise ImportError("torch-geometric is required. Install torch-geometric for your PyTorch/CUDA version.") from exc


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def collate_trajectories(batch):
    return batch


class RoadGNNEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float, gnn_type: str = "graphsage"):
        super().__init__()
        self.dropout = dropout
        self.gnn_type = gnn_type
        dims = [input_dim] + [hidden_dim] * max(num_layers - 1, 0) + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            if gnn_type == "gat":
                layers.append(GATConv(dims[i], dims[i + 1], heads=1, concat=False))
            else:
                layers.append(SAGEConv(dims[i], dims[i + 1]))
        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in dims[1:]])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
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
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class OneDirectionModel(nn.Module):
    def __init__(self, road_feat_dim: int, road_hidden_dim: int, emission_feat_dim: int, transition_feat_dim: int, gnn_type: str = "graphsage"):
        super().__init__()
        self.encoder = RoadGNNEncoder(
            input_dim=road_feat_dim,
            hidden_dim=road_hidden_dim,
            output_dim=road_hidden_dim,
            num_layers=2,
            dropout=0.1,
            gnn_type=gnn_type,
        )
        self.emission_head = MLP(road_hidden_dim + emission_feat_dim, [128, 64], 1, 0.1)
        self.transition_head = MLP(2 * road_hidden_dim + transition_feat_dim, [128, 64], 1, 0.1)

    def encode_roads(self, road_x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(road_x, edge_index)

    def score_emissions(self, road_emb: torch.Tensor, item: dict) -> torch.Tensor:
        edge_idx = item["candidate_edge_idx"].to(road_emb.device)
        mask = item["candidate_mask"].to(road_emb.device)
        feats = item["emission_features"].to(road_emb.device)
        safe_idx = edge_idx.clamp_min(0)
        emb = road_emb[safe_idx]
        x = torch.cat([emb, feats], dim=-1)
        logits = self.emission_head(x)
        return logits.masked_fill(~mask, -1e9)

    def score_transitions(self, road_emb: torch.Tensor, item: dict) -> torch.Tensor:
        edge_idx = item["candidate_edge_idx"].to(road_emb.device)
        trans_feats = item["transition_features"].to(road_emb.device)
        trans_mask = item["transition_mask"].to(road_emb.device)
        if trans_feats.numel() == 0:
            return torch.empty(0, device=road_emb.device)
        prev_idx = edge_idx[:-1].clamp_min(0)
        curr_idx = edge_idx[1:].clamp_min(0)
        prev_emb = road_emb[prev_idx]  # [T-1,K,H]
        curr_emb = road_emb[curr_idx]
        Tm1, K, H = prev_emb.shape
        prev_expand = prev_emb[:, :, None, :].expand(Tm1, K, K, H)
        curr_expand = curr_emb[:, None, :, :].expand(Tm1, K, K, H)
        x = torch.cat([prev_expand, curr_expand, trans_feats], dim=-1)
        logits = self.transition_head(x)
        return logits.masked_fill(~trans_mask, -1e9)


def emission_loss(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    valid = labels >= 0
    if valid.sum() == 0:
        return logits.sum() * 0.0, 0, 0
    loss = F.cross_entropy(logits[valid], labels[valid].to(logits.device))
    pred = logits[valid].argmax(dim=-1)
    correct = int((pred.cpu() == labels[valid]).sum().item())
    total = int(valid.sum().item())
    return loss, correct, total


def transition_loss(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    if logits.numel() == 0 or labels.numel() < 2:
        return logits.sum() * 0.0, 0, 0
    prev = labels[:-1]
    curr = labels[1:]
    valid = (prev >= 0) & (curr >= 0)
    losses = []
    correct = 0
    total = 0
    for ti in torch.where(valid)[0].tolist():
        row = logits[ti, int(prev[ti])]  # [K]
        target = curr[ti].view(1).to(row.device)
        losses.append(F.cross_entropy(row.view(1, -1), target))
        correct += int(row.argmax().cpu().item() == int(curr[ti]))
        total += 1
    if not losses:
        return logits.sum() * 0.0, 0, 0
    return torch.stack(losses).mean(), correct, total


def viterbi_decode(emissions: torch.Tensor, transitions: torch.Tensor) -> list[int]:
    T, K = emissions.shape
    if T == 0:
        return []
    dp = emissions[0].clone()
    back = []
    for t in range(1, T):
        scores = dp[:, None] + transitions[t - 1]
        best_score, best_prev = scores.max(dim=0)
        dp = emissions[t] + best_score
        back.append(best_prev)
    last = int(dp.argmax().item())
    path = [last]
    for b in reversed(back):
        last = int(b[last].item())
        path.append(last)
    return list(reversed(path))


def run_epoch(model, road_x, edge_index, loader, optimizer, device, train: bool, transition_weight: float) -> dict:
    model.train(train)
    total_loss = 0.0
    em_correct = em_total = tr_correct = tr_total = 0
    seq_correct = seq_total = 0
    n_batches = 0
    for batch in loader:
        if train:
            optimizer.zero_grad(set_to_none=True)
        road_emb = model.encode_roads(road_x, edge_index)
        batch_losses = []
        for item in batch:
            labels = item["gt_candidate_pos"].to(device)
            emissions = model.score_emissions(road_emb, item)
            transitions = model.score_transitions(road_emb, item)
            le, c, n = emission_loss(emissions, labels)
            lt, tc, tn = transition_loss(transitions, labels)
            batch_losses.append(le + transition_weight * lt)
            em_correct += c
            em_total += n
            tr_correct += tc
            tr_total += tn
            if transitions.numel() > 0:
                path = viterbi_decode(emissions.detach(), transitions.detach())
                valid = labels.cpu().numpy() >= 0
                if len(path) == len(labels):
                    pred = np.array(path)
                    gt = labels.cpu().numpy()
                    seq_correct += int((pred[valid] == gt[valid]).sum())
                    seq_total += int(valid.sum())
        loss = torch.stack(batch_losses).mean()
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total_loss += float(loss.detach().cpu().item())
        n_batches += 1
    return {
        "loss": total_loss / max(n_batches, 1),
        "emission_acc": em_correct / max(em_total, 1),
        "transition_acc": tr_correct / max(tr_total, 1),
        "viterbi_candidate_acc": seq_correct / max(seq_total, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--line-graph", type=Path, default=Path("data/processed/line_graph/line_graph.pt"))
    parser.add_argument("--train", type=Path, default=Path("data/processed/tensors/train_dataset.pt"))
    parser.add_argument("--val", type=Path, default=Path("data/processed/tensors/val_dataset.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--report", type=Path, default=Path("data/reports/training_report.json"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--transition-weight", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gnn-type", choices=["graphsage", "gat"], default="graphsage")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = choose_device(args.device)
    ensure_dir(args.output)
    ensure_dir(args.report.parent)

    graph = torch.load(args.line_graph, map_location="cpu")
    road_x = graph["x"].float().to(device)
    edge_index = graph["edge_index"].long().to(device)
    train_data = torch.load(args.train, map_location="cpu")
    val_data = torch.load(args.val, map_location="cpu")
    em_feat_dim = int(train_data[0]["emission_features"].shape[-1])
    tr_feat_dim = int(train_data[0]["transition_features"].shape[-1])

    model = OneDirectionModel(
        road_feat_dim=int(road_x.shape[-1]),
        road_hidden_dim=args.hidden_dim,
        emission_feat_dim=em_feat_dim,
        transition_feat_dim=tr_feat_dim,
        gnn_type=args.gnn_type,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_trajectories)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_trajectories)

    history = []
    best_score = -1.0
    best_path = args.output / "gnn_hmm_best.pt"
    last_path = args.output / "gnn_hmm_last.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, road_x, edge_index, train_loader, optimizer, device, True, args.transition_weight)
        with torch.no_grad():
            val_metrics = run_epoch(model, road_x, edge_index, val_loader, optimizer, device, False, args.transition_weight)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_em={val_metrics['emission_acc']:.4f} val_vit={val_metrics['viterbi_candidate_acc']:.4f}"
        )
        score = val_metrics["viterbi_candidate_acc"]
        ckpt = {
            "model_state": model.state_dict(),
            "road_feat_dim": int(road_x.shape[-1]),
            "hidden_dim": args.hidden_dim,
            "emission_feat_dim": em_feat_dim,
            "transition_feat_dim": tr_feat_dim,
            "gnn_type": args.gnn_type,
            "epoch": epoch,
            "val_metrics": val_metrics,
        }
        torch.save(ckpt, last_path)
        if score > best_score:
            best_score = score
            torch.save(ckpt, best_path)

    report = {"best_score": best_score, "history": history, "best_checkpoint": str(best_path), "last_checkpoint": str(last_path)}
    with args.report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Training complete. Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
