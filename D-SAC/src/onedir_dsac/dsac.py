from __future__ import annotations

import json
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm.auto import trange
from .data import RLDataset
from .evaluate import evaluate_policy
from .features import action_mask, actor_observation, observation_dims, privileged_observation
from .models import DiscreteSACModel
from .replay import ReplayBuffer
from .reward import RewardConfig, step_reward, terminal_reward


def load_bc_actor(model: DiscreteSACModel, path: str | Path, device: str) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_state = model.state_dict()
    compatible = {k: v for k, v in ckpt["model_state_dict"].items() if k in model_state and model_state[k].shape == v.shape}
    model_state.update(compatible)
    model.load_state_dict(model_state)
    model.target_q1.load_state_dict(model.q1.state_dict())
    model.target_q2.load_state_dict(model.q2.state_dict())
    return True


def collect_rollouts(model: DiscreteSACModel, dataset: RLDataset, replay: ReplayBuffer, device: str, num_episodes: int, reward_cfg: RewardConfig, k_max: int | None, greedy: bool = False) -> dict:
    model.eval()
    total_reward = total_acc = total_legal = total_steps = 0.0
    for _ in range(num_episodes):
        sample = random.choice(dataset.episodes)
        previous_action = None
        correct_flags, legal_flags = [], []
        ep_reward = 0.0
        for t in range(sample.length):
            actor_obs = actor_observation(sample, t, previous_action, k_max)
            critic_obs = privileged_observation(sample, t, previous_action, k_max)
            mask = action_mask(sample, t, k_max)
            with torch.no_grad():
                action, _ = model.act(actor_obs.to(device), mask.to(device), greedy=greedy)
            reward, info = step_reward(sample, t, action, previous_action, reward_cfg)
            done = t == sample.length - 1
            correct_flags.append(bool(info["correct"]))
            legal_flags.append(bool(info["legal_transition"]))
            if done:
                reward += terminal_reward(correct_flags, legal_flags, reward_cfg)
                next_actor_obs = torch.zeros_like(actor_obs)
                next_critic_obs = torch.zeros_like(critic_obs)
                next_mask = torch.zeros_like(mask)
            else:
                next_actor_obs = actor_observation(sample, t + 1, action, k_max)
                next_critic_obs = privileged_observation(sample, t + 1, action, k_max)
                next_mask = action_mask(sample, t + 1, k_max)
            replay.add(actor_obs, critic_obs, mask, action, reward, next_actor_obs, next_critic_obs, next_mask, done)
            previous_action = action
            ep_reward += reward
            total_steps += 1
        total_reward += ep_reward
        total_acc += sum(correct_flags) / max(len(correct_flags), 1)
        total_legal += sum(legal_flags) / max(len(legal_flags), 1)
    return {"rollout_mean_reward": total_reward / max(num_episodes, 1), "rollout_mean_accuracy": total_acc / max(num_episodes, 1), "rollout_legal_transition_rate": total_legal / max(num_episodes, 1), "rollout_steps": int(total_steps), "replay_size": len(replay)}


def dsac_update(model: DiscreteSACModel, replay: ReplayBuffer, actor_optimizer, critic_optimizer, log_alpha: torch.Tensor, alpha_optimizer, device: str, batch_size: int, gamma: float, tau: float, grad_clip_norm: float, target_entropy_scale: float, auto_alpha: bool) -> dict:
    batch = replay.sample(batch_size, device=device)
    alpha = log_alpha.exp().detach()
    with torch.no_grad():
        next_probs, next_log_probs = model.policy(batch.next_actor_obs, batch.next_action_mask)
        tq1, tq2 = model.target_q_values(batch.next_actor_obs, batch.next_critic_obs)
        next_value = (next_probs * (torch.min(tq1, tq2) - alpha * next_log_probs)).sum(dim=-1)
        target_q = batch.rewards + gamma * (1.0 - batch.dones) * next_value
    q1, q2 = model.q_values(batch.actor_obs, batch.critic_obs)
    q1_a = q1.gather(1, batch.actions.view(-1, 1)).squeeze(1)
    q2_a = q2.gather(1, batch.actions.view(-1, 1)).squeeze(1)
    critic_loss = F.mse_loss(q1_a, target_q) + F.mse_loss(q2_a, target_q)
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(list(model.q1.parameters()) + list(model.q2.parameters()), grad_clip_norm)
    critic_optimizer.step()
    probs, log_probs = model.policy(batch.actor_obs, batch.action_mask)
    q1_pi, q2_pi = model.q_values(batch.actor_obs, batch.critic_obs)
    actor_loss = (probs * (log_alpha.exp().detach() * log_probs - torch.min(q1_pi, q2_pi))).sum(dim=-1).mean()
    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.actor.parameters(), grad_clip_norm)
    actor_optimizer.step()
    entropy = -(probs * log_probs).sum(dim=-1)
    n_valid = batch.action_mask.float().sum(dim=-1).clamp_min(1.0)
    target_entropy = target_entropy_scale * torch.log(n_valid).mean().detach()
    alpha_loss_value = 0.0
    if auto_alpha and alpha_optimizer is not None:
        alpha_loss = (log_alpha.exp() * (entropy.detach().mean() - target_entropy)).mean()
        alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        alpha_optimizer.step()
        alpha_loss_value = float(alpha_loss.detach().cpu().item())
    model.soft_update_targets(tau)
    return {"critic_loss": float(critic_loss.detach().cpu().item()), "actor_loss": float(actor_loss.detach().cpu().item()), "alpha_loss": alpha_loss_value, "alpha": float(log_alpha.exp().detach().cpu().item()), "entropy": float(entropy.mean().detach().cpu().item()), "target_entropy": float(target_entropy.detach().cpu().item()), "mean_q": float(q1_a.mean().detach().cpu().item())}


def train_discrete_sac_asym(train_dataset: RLDataset, val_dataset: RLDataset, output_dir: str | Path, device: str = "cpu", seed: int = 42, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1, use_privileged_critic: bool = True, epochs: int = 60, rollouts_per_epoch: int = 200, updates_per_epoch: int = 800, batch_size: int = 512, replay_size: int = 200000, warmup_rollouts: int = 200, gamma: float = 0.98, tau: float = 0.005, actor_lr: float = 3.0e-4, critic_lr: float = 3.0e-4, alpha_lr: float = 3.0e-4, init_alpha: float = 0.2, auto_alpha: bool = True, target_entropy_scale: float = 0.7, grad_clip_norm: float = 5.0, reward_cfg: RewardConfig = RewardConfig(), load_bc_checkpoint: bool = True, bc_checkpoint: str | Path = "D-SAC/outputs/checkpoints/bc_actor.pt", k_max: int | None = None):
    random.seed(seed)
    torch.manual_seed(seed)
    actor_dim, critic_dim, action_dim = observation_dims(train_dataset[0], k_max)
    model = DiscreteSACModel(actor_dim, critic_dim, action_dim, hidden_dim, num_layers, dropout, use_privileged_critic).to(device)
    if load_bc_checkpoint:
        print(f"[INFO] Loaded BC checkpoint: {load_bc_actor(model, bc_checkpoint, device)}", flush=True)
    actor_optimizer = torch.optim.AdamW(model.actor.parameters(), lr=actor_lr)
    critic_optimizer = torch.optim.AdamW(list(model.q1.parameters()) + list(model.q2.parameters()), lr=critic_lr)
    log_alpha = torch.tensor(float(init_alpha)).log().to(device).requires_grad_(True)
    alpha_optimizer = torch.optim.AdamW([log_alpha], lr=alpha_lr) if auto_alpha else None
    replay = ReplayBuffer(replay_size)
    output_dir = Path(output_dir)
    ckpt_dir = output_dir / "checkpoints"
    report_dir = output_dir / "reports"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Warming replay buffer with {warmup_rollouts} rollouts", flush=True)
    collect_rollouts(model, train_dataset, replay, device, warmup_rollouts, reward_cfg, k_max, greedy=False)
    history, best_val = [], -1.0
    for epoch in range(1, epochs + 1):
        rollout_metrics = collect_rollouts(model, train_dataset, replay, device, rollouts_per_epoch, reward_cfg, k_max, greedy=False)
        updates = []
        iterator = trange(updates_per_epoch, desc=f"dsac epoch {epoch}", leave=False)
        for _ in iterator:
            update = dsac_update(model, replay, actor_optimizer, critic_optimizer, log_alpha, alpha_optimizer, device, batch_size, gamma, tau, grad_clip_norm, target_entropy_scale, auto_alpha)
            updates.append(update)
            iterator.set_postfix(alpha=update["alpha"], entropy=update["entropy"])
        update_metrics = {k: sum(u[k] for u in updates) / max(len(updates), 1) for k in updates[0]}
        _, val_metrics = evaluate_policy(model, val_dataset, device=device, greedy=True, k_max=k_max)
        metrics = {"epoch": epoch, **rollout_metrics, **update_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(metrics)
        print(json.dumps(metrics, indent=2), flush=True)
        ckpt = {"model_state_dict": model.state_dict(), "actor_obs_dim": actor_dim, "critic_obs_dim": critic_dim, "action_dim": action_dim, "model_config": {"hidden_dim": hidden_dim, "num_layers": num_layers, "dropout": dropout, "use_privileged_critic": use_privileged_critic}, "epoch": epoch, "metrics": metrics}
        torch.save(ckpt, ckpt_dir / "dsac_asym_last.pt")
        score = float(val_metrics.get("point_edge_accuracy", 0.0))
        if score > best_val:
            best_val = score
            torch.save(ckpt, ckpt_dir / "dsac_asym_best.pt")
            print(f"[checkpoint] Updated best D-SAC checkpoint at epoch {epoch}", flush=True)
    with (report_dir / "dsac_training_report.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(history, f, indent=2)
    return model, history
