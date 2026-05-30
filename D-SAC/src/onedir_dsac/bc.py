from __future__ import annotations

import json
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from .data import RLDataset
from .features import action_mask, actor_observation, observation_dims
from .models import DiscreteSACModel


def build_bc_tensors(dataset: RLDataset, k_max: int | None = None):
    obs, masks, labels = [], [], []
    for sample in dataset.episodes:
        previous_action = None
        for t in range(sample.length):
            label = int(sample.gt_candidate_pos[t].item())
            if label < 0:
                continue
            obs.append(actor_observation(sample, t, previous_action, k_max))
            masks.append(action_mask(sample, t, k_max))
            labels.append(label)
            previous_action = label
    return torch.stack(obs), torch.stack(masks).bool(), torch.tensor(labels, dtype=torch.long)


def train_behavior_cloning(train_dataset: RLDataset, val_dataset: RLDataset, output_path: str | Path, device: str = "cpu", epochs: int = 10, batch_size: int = 512, lr: float = 1.0e-3, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1, use_privileged_critic: bool = True, seed: int = 42, k_max: int | None = None) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)
    actor_dim, critic_dim, action_dim = observation_dims(train_dataset[0], k_max)
    model = DiscreteSACModel(actor_dim, critic_dim, action_dim, hidden_dim, num_layers, dropout, use_privileged_critic).to(device)
    train_obs, train_masks, train_labels = build_bc_tensors(train_dataset, k_max)
    val_obs, val_masks, val_labels = build_bc_tensors(val_dataset, k_max)
    loader = DataLoader(TensorDataset(train_obs, train_masks, train_labels), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.actor.parameters(), lr=lr)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_acc, history = -1.0, []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch_obs, batch_masks, batch_labels in tqdm(loader, desc=f"bc epoch {epoch}", leave=False):
            loss = F.cross_entropy(model.logits(batch_obs.to(device), batch_masks.to(device)), batch_labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        model.eval()
        with torch.no_grad():
            pred = model.logits(val_obs.to(device), val_masks.to(device)).argmax(dim=-1).cpu()
            acc = float((pred == val_labels).float().mean().item())
        metrics = {"epoch": epoch, "train_loss": sum(losses) / max(len(losses), 1), "val_action_accuracy": acc}
        history.append(metrics)
        print(json.dumps(metrics, indent=2), flush=True)
        if acc > best_acc:
            best_acc = acc
            torch.save({"model_state_dict": model.state_dict(), "actor_obs_dim": actor_dim, "critic_obs_dim": critic_dim, "action_dim": action_dim, "model_config": {"hidden_dim": hidden_dim, "num_layers": num_layers, "dropout": dropout, "use_privileged_critic": use_privileged_critic}, "metrics": metrics}, output_path)
    return {"history": history, "best_val_action_accuracy": best_acc}
