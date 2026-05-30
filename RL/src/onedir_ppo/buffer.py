from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass
class RolloutBatch:
    actor_obs: Tensor
    critic_obs: Tensor
    masks: Tensor
    actions: Tensor
    old_log_probs: Tensor
    returns: Tensor
    advantages: Tensor
    old_values: Tensor


class RolloutBuffer:
    def __init__(self) -> None:
        self.actor_obs=[]; self.critic_obs=[]; self.masks=[]; self.actions=[]; self.log_probs=[]; self.values=[]; self.rewards=[]; self.dones=[]

    def add(self, actor_obs, critic_obs, mask, action, log_prob, value, reward: float, done: bool) -> None:
        self.actor_obs.append(actor_obs.detach().cpu())
        self.critic_obs.append(critic_obs.detach().cpu())
        self.masks.append(mask.detach().cpu())
        self.actions.append(action.detach().cpu().reshape(()))
        self.log_probs.append(log_prob.detach().cpu().reshape(()))
        self.values.append(value.detach().cpu().reshape(()))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def __len__(self) -> int:
        return len(self.rewards)

    def compute(self, gamma: float, gae_lambda: float) -> RolloutBatch:
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        dones = torch.tensor(self.dones, dtype=torch.float32)
        values = torch.stack(self.values).float()
        adv = torch.zeros_like(rewards)
        last = torch.zeros(())
        next_value = torch.zeros(())
        for t in reversed(range(len(rewards))):
            nonterm = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * nonterm - values[t]
            last = delta + gamma * gae_lambda * nonterm * last
            adv[t] = last
            next_value = values[t]
        returns = adv + values
        if float(adv.std(unbiased=False).item()) > 1e-8:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        return RolloutBatch(torch.stack(self.actor_obs), torch.stack(self.critic_obs), torch.stack(self.masks).bool(), torch.stack(self.actions).long(), torch.stack(self.log_probs).float(), returns, adv, values)
