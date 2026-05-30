#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from torch_geometric.nn import GATConv, SAGEConv
except ImportError as exc:
    raise ImportError(
        "torch-geometric is required. Install torch-geometric for your PyTorch/CUDA version."
    ) from exc


MASK_VALUE = -1.0e9
MASK_THRESHOLD = -1.0e8


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


def resolve_checkpoint_paths(output: Path) -> tuple[Path, Path, Path]:
    if output.suffix == ".pt":
        checkpoint_dir = output.parent
        best_path = output
        if output.name.endswith("_best.pt"):
            last_path = output.with_name(output.name.replace("_best.pt", "_last.pt"))
        else:
            last_path = output.with_name(output.stem + "_last.pt")
    else:
        checkpoint_dir = output
        best_path = checkpoint_dir / "gnn_hmm_best.pt"
        last_path = checkpoint_dir / "gnn_hmm_last.pt"
    return checkpoint_dir, best_path, last_path


def print_json_flush(prefix: str, data: dict) -> None:
    def convert(value):
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(v) for v in value]
        if hasattr(value, "item"):
            return value.item()
        return value

    print(prefix + json.dumps(convert(data), indent=2), flush=True)


def collate_trajectories(batch):
    return batch


class RoadGNNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
        gnn_type: str = "graphsage",
    ):
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
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
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
    def __init__(
        self,
        road_feat_dim: int,
        road_hidden_dim: int,
        emission_feat_dim: int,
        transition_feat_dim: int,
        gnn_type: str = "graphsage",
    ):
        super().__init__()

        self.encoder = RoadGNNEncoder(
            input_dim=road_feat_dim,
            hidden_dim=road_hidden_dim,
            output_dim=road_hidden_dim,
            num_layers=2,
            dropout=0.1,
            gnn_type=gnn_type,
        )

        self.emission_head = MLP(
            road_hidden_dim + emission_feat_dim,
            [128, 64],
            1,
            0.1,
        )

        self.transition_head = MLP(
            2 * road_hidden_dim + transition_feat_dim,
            [128, 64],
            1,
            0.1,
        )

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
        return logits.masked_fill(~mask, MASK_VALUE)

    def score_transitions(self, road_emb: torch.Tensor, item: dict) -> torch.Tensor:
        edge_idx = item["candidate_edge_idx"].to(road_emb.device)
        trans_feats = item["transition_features"].to(road_emb.device)
        trans_mask = item["transition_mask"].to(road_emb.device)

        if trans_feats.numel() == 0:
            return torch.empty(0, device=road_emb.device)

        prev_idx = edge_idx[:-1].clamp_min(0)
        curr_idx = edge_idx[1:].clamp_min(0)

        prev_emb = road_emb[prev_idx]
        curr_emb = road_emb[curr_idx]

        t_minus_1, k, h = prev_emb.shape

        prev_expand = prev_emb[:, :, None, :].expand(t_minus_1, k, k, h)
        curr_expand = curr_emb[:, None, :, :].expand(t_minus_1, k, k, h)

        x = torch.cat([prev_expand, curr_expand, trans_feats], dim=-1)
        logits = self.transition_head(x)

        return logits.masked_fill(~trans_mask, MASK_VALUE)


def masked_cross_entropy(
    scores: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, int]:
    if scores.ndim == 1:
        scores = scores.view(1, -1)

    target = target.to(scores.device).long().view(-1)
    if scores.shape[0] != target.shape[0]:
        raise ValueError("scores batch dimension and target dimension do not match.")

    valid_action_mask = scores > MASK_THRESHOLD

    target_in_range = (target >= 0) & (target < scores.shape[1])
    safe_target = target.clamp(0, scores.shape[1] - 1)

    target_valid = torch.zeros_like(target_in_range)
    if target.numel() > 0:
        target_valid = valid_action_mask.gather(1, safe_target.view(-1, 1)).squeeze(1)

    keep = target_in_range & target_valid

    if keep.sum() == 0:
        return scores.sum() * 0.0, int(target.numel())

    scores = scores[keep]
    target = target[keep]
    valid_action_mask = valid_action_mask[keep]

    masked_scores = scores.masked_fill(~valid_action_mask, MASK_VALUE)
    log_probs = F.log_softmax(masked_scores, dim=-1)

    smoothing = float(max(0.0, min(label_smoothing, 0.999)))

    if smoothing == 0.0:
        return F.nll_loss(log_probs, target), int((~keep).sum().item())

    n_valid = valid_action_mask.sum(dim=1).clamp_min(1)
    n_non_target = (n_valid - 1).clamp_min(1)

    target_dist = torch.zeros_like(log_probs)
    smooth_mass = (smoothing / n_non_target.float()).unsqueeze(1)
    target_dist = target_dist + valid_action_mask.float() * smooth_mass

    target_prob = torch.full_like(n_valid.float(), 1.0 - smoothing)
    target_prob = torch.where(n_valid == 1, torch.ones_like(target_prob), target_prob)

    target_dist.scatter_(1, target.view(-1, 1), target_prob.view(-1, 1))
    target_dist = target_dist * valid_action_mask.float()

    row_sums = target_dist.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    target_dist = target_dist / row_sums

    loss = -(target_dist * log_probs).sum(dim=1).mean()
    return loss, int((~keep).sum().item())


def emission_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float = 0.0,
    margin_weight: float = 0.0,
    margin: float = 1.0,
) -> dict:
    labels = labels.to(logits.device)
    valid = labels >= 0

    if valid.sum() == 0:
        zero = logits.sum() * 0.0
        return {
            "loss": zero,
            "ce_loss": zero,
            "margin_loss": zero,
            "correct": 0,
            "total": 0,
            "used": 0,
            "skipped": int(labels.numel()),
        }

    scores = logits[valid]
    target = labels[valid].long()

    ce_loss, skipped = masked_cross_entropy(scores, target, label_smoothing=label_smoothing)

    valid_action_mask = scores > MASK_THRESHOLD
    target_in_range = (target >= 0) & (target < scores.shape[1])
    safe_target = target.clamp(0, scores.shape[1] - 1)
    target_valid = valid_action_mask.gather(1, safe_target.view(-1, 1)).squeeze(1)
    keep = target_in_range & target_valid

    if keep.sum() == 0:
        zero = logits.sum() * 0.0
        return {
            "loss": zero,
            "ce_loss": zero,
            "margin_loss": zero,
            "correct": 0,
            "total": int(target.numel()),
            "used": 0,
            "skipped": int(target.numel()),
        }

    kept_scores = scores[keep]
    kept_target = target[keep]
    kept_mask = valid_action_mask[keep]

    pred_scores = kept_scores.masked_fill(~kept_mask, MASK_VALUE)
    pred = pred_scores.argmax(dim=-1)
    correct = int((pred == kept_target).sum().detach().cpu().item())

    margin_loss = logits.sum() * 0.0

    if margin_weight > 0:
        gt_scores = kept_scores.gather(1, kept_target.view(-1, 1)).squeeze(1)

        negative_mask = kept_mask.clone()
        negative_mask.scatter_(1, kept_target.view(-1, 1), False)
        has_negative = negative_mask.any(dim=1)

        if has_negative.any():
            hardest_neg = kept_scores.masked_fill(~negative_mask, MASK_VALUE).max(dim=1).values
            margin_loss = F.relu(margin - gt_scores + hardest_neg)[has_negative].mean()

    total_loss = ce_loss + margin_weight * margin_loss

    return {
        "loss": total_loss,
        "ce_loss": ce_loss,
        "margin_loss": margin_loss,
        "correct": correct,
        "total": int(target.numel()),
        "used": int(keep.sum().item()),
        "skipped": int((~keep).sum().item() + skipped),
    }


def transition_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float = 0.0,
) -> dict:
    labels = labels.to(logits.device)

    if logits.numel() == 0 or labels.numel() < 2:
        zero = logits.sum() * 0.0
        return {
            "loss": zero,
            "ce_loss": zero,
            "correct": 0,
            "total": 0,
            "used": 0,
            "skipped": 0,
        }

    prev = labels[:-1]
    curr = labels[1:]
    valid = (prev >= 0) & (curr >= 0)

    losses = []
    correct = 0
    total = 0
    used = 0
    skipped = 0

    for ti in torch.where(valid)[0].tolist():
        prev_idx = int(prev[ti].detach().cpu().item())
        curr_idx = int(curr[ti].detach().cpu().item())

        if prev_idx < 0 or prev_idx >= logits.shape[1]:
            skipped += 1
            total += 1
            continue

        row = logits[ti, prev_idx]

        if curr_idx < 0 or curr_idx >= row.numel() or row[curr_idx] <= MASK_THRESHOLD:
            skipped += 1
            total += 1
            continue

        ce_loss, local_skipped = masked_cross_entropy(
            row.view(1, -1),
            torch.tensor([curr_idx], device=logits.device),
            label_smoothing=label_smoothing,
        )

        losses.append(ce_loss)
        used += 1
        skipped += local_skipped
        total += 1

        pred = row.masked_fill(row <= MASK_THRESHOLD, MASK_VALUE).argmax()
        correct += int(pred.detach().cpu().item() == curr_idx)

    if not losses:
        zero = logits.sum() * 0.0
        return {
            "loss": zero,
            "ce_loss": zero,
            "correct": correct,
            "total": total,
            "used": used,
            "skipped": skipped,
        }

    ce = torch.stack(losses).mean()

    return {
        "loss": ce,
        "ce_loss": ce,
        "correct": correct,
        "total": total,
        "used": used,
        "skipped": skipped,
    }


def viterbi_decode(emissions: torch.Tensor, transitions: torch.Tensor) -> list[int]:
    sequence_length, num_candidates = emissions.shape

    if sequence_length == 0:
        return []

    dp = emissions[0].clone()
    back = []

    for t in range(1, sequence_length):
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


def scalar(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def run_epoch(
    model,
    road_x,
    edge_index,
    loader,
    optimizer,
    device,
    train: bool,
    transition_weight: float,
    emission_weight: float,
    label_smoothing: float,
    margin_weight: float,
    margin: float,
    grad_clip_norm: float,
    epoch: int,
    phase: str,
    log_every_batches: int,
) -> dict:
    model.train(train)

    total_batch_loss = 0.0
    n_batches = 0

    em_ce_sum = 0.0
    em_margin_sum = 0.0
    em_loss_sum = 0.0
    tr_loss_sum = 0.0

    em_correct = 0
    em_total = 0
    em_used = 0
    em_skipped = 0

    tr_correct = 0
    tr_total = 0
    tr_used = 0
    tr_skipped = 0

    seq_correct = 0
    seq_total = 0

    for batch_idx, batch in enumerate(loader, start=1):
        if train:
            optimizer.zero_grad(set_to_none=True)

        road_emb = model.encode_roads(road_x, edge_index)

        item_losses = []

        for item in batch:
            labels = item["gt_candidate_pos"].to(device)

            emissions = model.score_emissions(road_emb, item)
            transitions = model.score_transitions(road_emb, item)

            em = emission_loss(
                emissions,
                labels,
                label_smoothing=label_smoothing,
                margin_weight=margin_weight,
                margin=margin,
            )

            tr = transition_loss(
                transitions,
                labels,
                label_smoothing=label_smoothing,
            )

            item_loss = emission_weight * em["loss"] + transition_weight * tr["loss"]
            item_losses.append(item_loss)

            em_correct += int(em["correct"])
            em_total += int(em["total"])
            em_used += int(em["used"])
            em_skipped += int(em["skipped"])

            tr_correct += int(tr["correct"])
            tr_total += int(tr["total"])
            tr_used += int(tr["used"])
            tr_skipped += int(tr["skipped"])

            if em["used"] > 0:
                em_ce_sum += scalar(em["ce_loss"]) * int(em["used"])
                em_margin_sum += scalar(em["margin_loss"]) * int(em["used"])
                em_loss_sum += scalar(em["loss"]) * int(em["used"])

            if tr["used"] > 0:
                tr_loss_sum += scalar(tr["loss"]) * int(tr["used"])

            if transitions.numel() > 0:
                path = viterbi_decode(emissions.detach(), transitions.detach())

                if len(path) == len(labels):
                    gt = labels.detach().cpu().numpy()
                    valid = gt >= 0
                    pred = np.array(path)

                    seq_correct += int((pred[valid] == gt[valid]).sum())
                    seq_total += int(valid.sum())

        if not item_losses:
            continue

        loss = torch.stack(item_losses).mean()

        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        batch_loss = scalar(loss)
        total_batch_loss += batch_loss
        n_batches += 1

        if log_every_batches > 0 and batch_idx % log_every_batches == 0:
            print(
                f"[{phase}] epoch={epoch:03d} batch={batch_idx:04d}/{len(loader):04d} "
                f"batch_loss={batch_loss:.6f} "
                f"em_acc={em_correct / max(em_total, 1):.4f} "
                f"tr_acc={tr_correct / max(tr_total, 1):.4f}",
                flush=True,
            )

    return {
        "loss": total_batch_loss / max(n_batches, 1),
        "emission_loss": em_loss_sum / max(em_used, 1),
        "emission_ce_loss": em_ce_sum / max(em_used, 1),
        "emission_margin_loss": em_margin_sum / max(em_used, 1),
        "transition_loss": tr_loss_sum / max(tr_used, 1),
        "weighted_emission_loss": emission_weight * em_loss_sum / max(em_used, 1),
        "weighted_transition_loss": transition_weight * tr_loss_sum / max(tr_used, 1),
        "emission_acc": em_correct / max(em_total, 1),
        "transition_acc": tr_correct / max(tr_total, 1),
        "viterbi_candidate_acc": seq_correct / max(seq_total, 1),
        "emission_supervised_total": em_total,
        "emission_supervised_used": em_used,
        "emission_supervised_skipped": em_skipped,
        "transition_supervised_total": tr_total,
        "transition_supervised_used": tr_used,
        "transition_supervised_skipped": tr_skipped,
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
    parser.add_argument("--emission-weight", type=float, default=1.0)
    parser.add_argument("--transition-weight", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--margin-weight", type=float, default=0.0)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gnn-type", choices=["graphsage", "gat"], default="graphsage")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every-batches", type=int, default=0)

    args = parser.parse_args()

    set_seed(args.seed)

    device = choose_device(args.device)
    checkpoint_dir, best_path, last_path = resolve_checkpoint_paths(args.output)

    ensure_dir(checkpoint_dir)
    ensure_dir(args.report.parent)

    print(f"[INFO] device={device}", flush=True)
    print(f"[INFO] checkpoint_dir={checkpoint_dir}", flush=True)
    print(f"[INFO] best_path={best_path}", flush=True)
    print(f"[INFO] last_path={last_path}", flush=True)

    graph = torch.load(args.line_graph, map_location="cpu")
    road_x = graph["x"].float().to(device)
    edge_index = graph["edge_index"].long().to(device)

    train_data = torch.load(args.train, map_location="cpu", weights_only=False)
    val_data = torch.load(args.val, map_location="cpu", weights_only=False)

    em_feat_dim = int(train_data[0]["emission_features"].shape[-1])
    tr_feat_dim = int(train_data[0]["transition_features"].shape[-1])

    print(f"[INFO] train_trajectories={len(train_data)} val_trajectories={len(val_data)}", flush=True)
    print(f"[INFO] road_nodes={road_x.shape[0]} road_feat_dim={road_x.shape[-1]}", flush=True)
    print(f"[INFO] emission_feat_dim={em_feat_dim} transition_feat_dim={tr_feat_dim}", flush=True)

    model = OneDirectionModel(
        road_feat_dim=int(road_x.shape[-1]),
        road_hidden_dim=args.hidden_dim,
        emission_feat_dim=em_feat_dim,
        transition_feat_dim=tr_feat_dim,
        gnn_type=args.gnn_type,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_trajectories,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_trajectories,
    )

    history = []
    best_score = -1.0

    for epoch in range(1, args.epochs + 1):
        print(f"[train] Starting epoch {epoch}/{args.epochs}", flush=True)

        train_metrics = run_epoch(
            model=model,
            road_x=road_x,
            edge_index=edge_index,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            train=True,
            transition_weight=args.transition_weight,
            emission_weight=args.emission_weight,
            label_smoothing=args.label_smoothing,
            margin_weight=args.margin_weight,
            margin=args.margin,
            grad_clip_norm=args.grad_clip_norm,
            epoch=epoch,
            phase="train",
            log_every_batches=args.log_every_batches,
        )

        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                road_x=road_x,
                edge_index=edge_index,
                loader=val_loader,
                optimizer=optimizer,
                device=device,
                train=False,
                transition_weight=args.transition_weight,
                emission_weight=args.emission_weight,
                label_smoothing=args.label_smoothing,
                margin_weight=args.margin_weight,
                margin=args.margin,
                grad_clip_norm=args.grad_clip_norm,
                epoch=epoch,
                phase="val",
                log_every_batches=0,
            )

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }

        history.append(row)

        print(f"[train] Finished epoch {epoch}/{args.epochs}", flush=True)
        print_json_flush("[metrics] ", row)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_em={train_metrics['emission_loss']:.6f} "
            f"train_tr={train_metrics['transition_loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_em_loss={val_metrics['emission_loss']:.6f} "
            f"val_tr_loss={val_metrics['transition_loss']:.6f} "
            f"val_em_acc={val_metrics['emission_acc']:.4f} "
            f"val_tr_acc={val_metrics['transition_acc']:.4f} "
            f"val_vit={val_metrics['viterbi_candidate_acc']:.4f}",
            flush=True,
        )

        score = val_metrics["viterbi_candidate_acc"]

        ckpt = {
            "model_state": model.state_dict(),
            "road_feat_dim": int(road_x.shape[-1]),
            "hidden_dim": args.hidden_dim,
            "emission_feat_dim": em_feat_dim,
            "transition_feat_dim": tr_feat_dim,
            "gnn_type": args.gnn_type,
            "train_args": vars(args),
            "epoch": epoch,
            "val_metrics": val_metrics,
        }

        torch.save(ckpt, last_path)

        if score > best_score:
            best_score = score
            torch.save(ckpt, best_path)
            print(f"[checkpoint] Updated best checkpoint at epoch {epoch}: {best_path}", flush=True)

    report = {
        "best_score": best_score,
        "history": history,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
    }

    with args.report.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2)

    print(f"[OK] Training complete. Best checkpoint: {best_path}", flush=True)
    print(f"[OK] Last checkpoint: {last_path}", flush=True)
    print(f"[OK] Report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
