from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass
class TransitionBatch:
    actor_obs: Tensor
    critic_obs: Tensor
    action_mask: Tensor
    actions: Tensor
    rewards: Tensor
    next_actor_obs: Tensor
    next_critic_obs: Tensor
    next_action_mask: Tensor
    dones: Tensor


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.storage = deque(maxlen=int(capacity))

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, actor_obs: Tensor, critic_obs: Tensor, action_mask: Tensor, action: int, reward: float, next_actor_obs: Tensor, next_critic_obs: Tensor, next_action_mask: Tensor, done: bool) -> None:
        self.storage.append((actor_obs.cpu(), critic_obs.cpu(), action_mask.cpu(), int(action), float(reward), next_actor_obs.cpu(), next_critic_obs.cpu(), next_action_mask.cpu(), bool(done)))

    def sample(self, batch_size: int, device: str) -> TransitionBatch:
        batch = random.sample(self.storage, min(batch_size, len(self.storage)))
        return TransitionBatch(
            actor_obs=torch.stack([b[0] for b in batch]).to(device),
            critic_obs=torch.stack([b[1] for b in batch]).to(device),
            action_mask=torch.stack([b[2] for b in batch]).bool().to(device),
            actions=torch.tensor([b[3] for b in batch], dtype=torch.long, device=device),
            rewards=torch.tensor([b[4] for b in batch], dtype=torch.float32, device=device),
            next_actor_obs=torch.stack([b[5] for b in batch]).to(device),
            next_critic_obs=torch.stack([b[6] for b in batch]).to(device),
            next_action_mask=torch.stack([b[7] for b in batch]).bool().to(device),
            dones=torch.tensor([b[8] for b in batch], dtype=torch.float32, device=device),
        )
