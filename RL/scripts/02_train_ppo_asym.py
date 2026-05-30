from __future__ import annotations
import argparse
from pathlib import Path
from _bootstrap import add_rl_src_to_path
add_rl_src_to_path()
from onedir_ppo.config import apply_overrides, deep_get, load_config, resolve_device
from onedir_ppo.data import load_rl_dataset
from onedir_ppo.ppo import train_ppo_asym
from onedir_ppo.reward import RewardConfig

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=Path('RL/configs/ppo_default.yaml')); p.add_argument('--override',nargs='*',default=[]); a=p.parse_args(); cfg=apply_overrides(load_config(a.config),a.override); device=resolve_device(deep_get(cfg,'project.device','auto'))
    train=load_rl_dataset(deep_get(cfg,'paths.train_dataset')); val=load_rl_dataset(deep_get(cfg,'paths.val_dataset'))
    rcfg=RewardConfig(correct_action_reward=float(deep_get(cfg,'reward.correct_action_reward',1.0)),wrong_action_penalty=float(deep_get(cfg,'reward.wrong_action_penalty',-0.2)),invalid_action_penalty=float(deep_get(cfg,'reward.invalid_action_penalty',-2.0)),legal_transition_bonus=float(deep_get(cfg,'reward.legal_transition_bonus',0.2)),illegal_transition_penalty=float(deep_get(cfg,'reward.illegal_transition_penalty',-1.0)),projection_error_weight=float(deep_get(cfg,'reward.projection_error_weight',0.02)),projection_error_cap_m=float(deep_get(cfg,'reward.projection_error_cap_m',25.0)),terminal_success_accuracy=float(deep_get(cfg,'reward.terminal_success_accuracy',0.90)),terminal_success_bonus=float(deep_get(cfg,'reward.terminal_success_bonus',2.0)),terminal_accuracy_weight=float(deep_get(cfg,'reward.terminal_accuracy_weight',1.0)),terminal_illegal_transition_weight=float(deep_get(cfg,'reward.terminal_illegal_transition_weight',0.2)))
    train_ppo_asym(train,val,deep_get(cfg,'paths.output_dir','RL/outputs'),device=device,seed=int(deep_get(cfg,'project.seed',42)),hidden_dim=int(deep_get(cfg,'model.hidden_dim',256)),num_layers=int(deep_get(cfg,'model.num_layers',2)),dropout=float(deep_get(cfg,'model.dropout',0.1)),use_privileged_critic=bool(deep_get(cfg,'features.use_privileged_critic',True)),epochs=int(deep_get(cfg,'ppo.epochs',40)),episodes_per_epoch=int(deep_get(cfg,'ppo.episodes_per_epoch',200)),update_epochs=int(deep_get(cfg,'ppo.update_epochs',4)),minibatch_size=int(deep_get(cfg,'ppo.minibatch_size',512)),gamma=float(deep_get(cfg,'ppo.gamma',0.98)),gae_lambda=float(deep_get(cfg,'ppo.gae_lambda',0.95)),clip_ratio=float(deep_get(cfg,'ppo.clip_ratio',0.2)),actor_lr=float(deep_get(cfg,'ppo.actor_lr',3e-4)),critic_lr=float(deep_get(cfg,'ppo.critic_lr',3e-4)),entropy_coef=float(deep_get(cfg,'ppo.entropy_coef',0.01)),value_coef=float(deep_get(cfg,'ppo.value_coef',0.5)),grad_clip_norm=float(deep_get(cfg,'ppo.grad_clip_norm',5.0)),reward_cfg=rcfg,load_bc_checkpoint=bool(deep_get(cfg,'ppo.load_bc_checkpoint',True)),bc_checkpoint=deep_get(cfg,'bc.checkpoint','RL/outputs/checkpoints/bc_actor.pt'),k_max=deep_get(cfg,'features.max_candidates',None))
if __name__=='__main__': main()
