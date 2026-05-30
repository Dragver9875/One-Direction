from __future__ import annotations
import torch
from torch import Tensor, nn
from torch.distributions import Categorical


def mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    cur = input_dim
    for _ in range(num_layers):
        layers += [nn.Linear(cur, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        cur = hidden_dim
    layers.append(nn.Linear(cur, output_dim))
    return nn.Sequential(*layers)


class PPOActorCritic(nn.Module):
    def __init__(self, actor_obs_dim: int, critic_obs_dim: int, action_dim: int, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1, use_privileged_critic: bool = True) -> None:
        super().__init__()
        self.actor_obs_dim = actor_obs_dim
        self.critic_obs_dim = critic_obs_dim
        self.action_dim = action_dim
        self.use_privileged_critic = use_privileged_critic
        self.actor = mlp(actor_obs_dim, hidden_dim, action_dim, num_layers, dropout)
        self.critic = mlp(critic_obs_dim if use_privileged_critic else actor_obs_dim, hidden_dim, 1, num_layers, dropout)

    def masked_logits(self, actor_obs: Tensor, mask: Tensor) -> Tensor:
        if actor_obs.ndim == 1:
            actor_obs = actor_obs.unsqueeze(0)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        return self.actor(actor_obs.float()).masked_fill(~mask.bool(), -1e9)

    def distribution(self, actor_obs: Tensor, mask: Tensor) -> Categorical:
        return Categorical(logits=self.masked_logits(actor_obs, mask))

    def value(self, actor_obs: Tensor, critic_obs: Tensor) -> Tensor:
        if actor_obs.ndim == 1:
            actor_obs = actor_obs.unsqueeze(0)
        if critic_obs.ndim == 1:
            critic_obs = critic_obs.unsqueeze(0)
        x = critic_obs if self.use_privileged_critic else actor_obs
        return self.critic(x.float()).squeeze(-1)

    def act(self, actor_obs: Tensor, critic_obs: Tensor, mask: Tensor, greedy: bool = False):
        dist = self.distribution(actor_obs, mask)
        action = torch.argmax(dist.logits, dim=-1) if greedy else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.value(actor_obs, critic_obs)

    def evaluate_actions(self, actor_obs: Tensor, critic_obs: Tensor, mask: Tensor, actions: Tensor):
        dist = self.distribution(actor_obs, mask)
        return dist.log_prob(actions), dist.entropy(), self.value(actor_obs, critic_obs)
