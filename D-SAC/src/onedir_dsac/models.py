from __future__ import annotations

import copy
import torch
from torch import Tensor, nn
from torch.distributions import Categorical


def build_mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for _ in range(num_layers):
        layers += [nn.Linear(current, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current = hidden_dim
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class DiscreteSACModel(nn.Module):
    def __init__(self, actor_obs_dim: int, critic_obs_dim: int, action_dim: int, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1, use_privileged_critic: bool = True) -> None:
        super().__init__()
        self.actor_obs_dim = actor_obs_dim
        self.critic_obs_dim = critic_obs_dim
        self.action_dim = action_dim
        self.use_privileged_critic = use_privileged_critic
        critic_input_dim = critic_obs_dim if use_privileged_critic else actor_obs_dim
        self.actor = build_mlp(actor_obs_dim, hidden_dim, action_dim, num_layers, dropout)
        self.q1 = build_mlp(critic_input_dim, hidden_dim, action_dim, num_layers, dropout)
        self.q2 = build_mlp(critic_input_dim, hidden_dim, action_dim, num_layers, dropout)
        self.target_q1 = copy.deepcopy(self.q1)
        self.target_q2 = copy.deepcopy(self.q2)
        for p in list(self.target_q1.parameters()) + list(self.target_q2.parameters()):
            p.requires_grad_(False)

    def critic_input(self, actor_obs: Tensor, critic_obs: Tensor) -> Tensor:
        if actor_obs.ndim == 1:
            actor_obs = actor_obs.unsqueeze(0)
        if critic_obs.ndim == 1:
            critic_obs = critic_obs.unsqueeze(0)
        return critic_obs if self.use_privileged_critic else actor_obs

    def logits(self, actor_obs: Tensor, action_mask: Tensor) -> Tensor:
        if actor_obs.ndim == 1:
            actor_obs = actor_obs.unsqueeze(0)
        if action_mask.ndim == 1:
            action_mask = action_mask.unsqueeze(0)
        return self.actor(actor_obs.float()).masked_fill(~action_mask.bool(), -1.0e9)

    def policy(self, actor_obs: Tensor, action_mask: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.logits(actor_obs, action_mask)
        probs = torch.softmax(logits, dim=-1) * action_mask.float()
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        log_probs = torch.where(action_mask.bool(), torch.log(probs.clamp_min(1.0e-8)), torch.zeros_like(probs))
        return probs, log_probs

    def distribution(self, actor_obs: Tensor, action_mask: Tensor) -> Categorical:
        return Categorical(logits=self.logits(actor_obs, action_mask))

    def q_values(self, actor_obs: Tensor, critic_obs: Tensor) -> tuple[Tensor, Tensor]:
        x = self.critic_input(actor_obs, critic_obs).float()
        return self.q1(x), self.q2(x)

    def target_q_values(self, actor_obs: Tensor, critic_obs: Tensor) -> tuple[Tensor, Tensor]:
        x = self.critic_input(actor_obs, critic_obs).float()
        return self.target_q1(x), self.target_q2(x)

    @torch.no_grad()
    def act(self, actor_obs: Tensor, action_mask: Tensor, greedy: bool = False) -> tuple[int, float]:
        dist = self.distribution(actor_obs, action_mask)
        action = torch.argmax(dist.logits, dim=-1) if greedy else dist.sample()
        probs = torch.softmax(dist.logits, dim=-1)
        return int(action.item()), float(probs.squeeze(0)[int(action.item())].detach().cpu().item())

    @torch.no_grad()
    def soft_update_targets(self, tau: float) -> None:
        for target, source in zip(self.target_q1.parameters(), self.q1.parameters()):
            target.data.mul_(1.0 - tau).add_(source.data, alpha=tau)
        for target, source in zip(self.target_q2.parameters(), self.q2.parameters()):
            target.data.mul_(1.0 - tau).add_(source.data, alpha=tau)
