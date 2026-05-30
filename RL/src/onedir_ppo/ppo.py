from __future__ import annotations
import json, random
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from .buffer import RolloutBuffer
from .data import RLDataset
from .evaluate import evaluate_policy
from .features import action_mask, actor_observation, observation_dims, privileged_observation
from .models import PPOActorCritic
from .reward import RewardConfig, step_reward, terminal_reward


def collect_rollouts(model,dataset,device,episodes_per_epoch,reward_cfg,k_max=None):
    buf=RolloutBuffer(); model.eval(); rewards=[]; accs=[]; legals=[]
    for _ in range(episodes_per_epoch):
        sample=random.choice(dataset.episodes); prev=None; correct=[]; legal_flags=[]; total=0.0
        for t in range(sample.length):
            ao=actor_observation(sample,t,prev,k_max).to(device); co=privileged_observation(sample,t,prev,k_max).to(device); m=action_mask(sample,t,k_max).to(device)
            with torch.no_grad(): action,logp,_,value=model.act(ao,co,m,greedy=False)
            a=int(action.item()); rew,info=step_reward(sample,t,a,prev,reward_cfg); done=t==sample.length-1
            if done: rew += terminal_reward(correct+[bool(info['correct'])], legal_flags+[bool(info['legal_transition'])], reward_cfg)
            buf.add(ao,co,m,action.squeeze(0),logp.squeeze(0),value.squeeze(0),rew,done)
            correct.append(bool(info['correct'])); legal_flags.append(bool(info['legal_transition'])); total+=rew; prev=a
        rewards.append(total); accs.append(sum(correct)/max(len(correct),1)); legals.append(sum(legal_flags)/max(len(legal_flags),1))
    return buf, {'rollout_mean_reward':sum(rewards)/max(len(rewards),1),'rollout_mean_accuracy':sum(accs)/max(len(accs),1),'rollout_legal_transition_rate':sum(legals)/max(len(legals),1),'rollout_steps':len(buf)}


def update(model,batch,actor_opt,critic_opt,device,update_epochs,minibatch_size,clip_ratio,entropy_coef,value_coef,grad_clip_norm):
    ds=TensorDataset(batch.actor_obs,batch.critic_obs,batch.masks,batch.actions,batch.old_log_probs,batch.returns,batch.advantages)
    loader=DataLoader(ds,batch_size=minibatch_size,shuffle=True); stats={'policy_loss':[],'value_loss':[],'entropy':[],'approx_kl':[]}; model.train()
    for _ in range(update_epochs):
        for ao,co,m,a,oldlp,ret,adv in loader:
            ao,co,m,a,oldlp,ret,adv=ao.to(device),co.to(device),m.to(device),a.to(device),oldlp.to(device),ret.to(device),adv.to(device)
            lp,ent,val=model.evaluate_actions(ao,co,m,a); ratio=torch.exp(lp-oldlp)
            pol=-torch.min(ratio*adv, torch.clamp(ratio,1-clip_ratio,1+clip_ratio)*adv).mean()
            vloss=(ret-val).pow(2).mean(); loss=pol+value_coef*vloss-entropy_coef*ent.mean()
            actor_opt.zero_grad(set_to_none=True); critic_opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip_norm); actor_opt.step(); critic_opt.step()
            stats['policy_loss'].append(float(pol.detach().cpu())); stats['value_loss'].append(float(vloss.detach().cpu())); stats['entropy'].append(float(ent.mean().detach().cpu())); stats['approx_kl'].append(float((oldlp-lp).mean().detach().cpu()))
    return {k:sum(v)/max(len(v),1) for k,v in stats.items()}


def load_bc_if_requested(model,path,device):
    path=Path(path)
    if not path.exists(): return False
    ckpt=torch.load(path,map_location=device,weights_only=False); state=ckpt['model_state_dict']; ms=model.state_dict(); ok={k:v for k,v in state.items() if k in ms and ms[k].shape==v.shape}; ms.update(ok); model.load_state_dict(ms); return True


def train_ppo_asym(train_dataset:RLDataset,val_dataset:RLDataset,output_dir,device='cpu',seed=42,hidden_dim=256,num_layers=2,dropout=0.1,use_privileged_critic=True,epochs=40,episodes_per_epoch=200,update_epochs=4,minibatch_size=512,gamma=0.98,gae_lambda=0.95,clip_ratio=0.2,actor_lr=3e-4,critic_lr=3e-4,entropy_coef=0.01,value_coef=0.5,grad_clip_norm=5.0,reward_cfg=RewardConfig(),load_bc_checkpoint=True,bc_checkpoint='RL/outputs/checkpoints/bc_actor.pt',k_max=None):
    random.seed(seed); torch.manual_seed(seed); ad,cd,actdim=observation_dims(train_dataset[0],k_max)
    model=PPOActorCritic(ad,cd,actdim,hidden_dim,num_layers,dropout,use_privileged_critic).to(device)
    if load_bc_checkpoint: print(f"[INFO] Loaded BC checkpoint: {load_bc_if_requested(model,bc_checkpoint,device)}")
    actor_opt=torch.optim.AdamW(model.actor.parameters(),lr=actor_lr); critic_opt=torch.optim.AdamW(model.critic.parameters(),lr=critic_lr)
    out=Path(output_dir); ckpt_dir=out/'checkpoints'; rep_dir=out/'reports'; ckpt_dir.mkdir(parents=True,exist_ok=True); rep_dir.mkdir(parents=True,exist_ok=True)
    hist=[]; best=-1.0
    for epoch in range(1,epochs+1):
        buf,roll=collect_rollouts(model,train_dataset,device,episodes_per_epoch,reward_cfg,k_max); batch=buf.compute(gamma,gae_lambda)
        upd=update(model,batch,actor_opt,critic_opt,device,update_epochs,minibatch_size,clip_ratio,entropy_coef,value_coef,grad_clip_norm)
        _,val=evaluate_policy(model,val_dataset,device=device,greedy=True,k_max=k_max)
        row={'epoch':epoch,**roll,**upd,**{f'val_{k}':v for k,v in val.items()}}; hist.append(row); print(json.dumps(row,indent=2))
        ckpt={'model_state_dict':model.state_dict(),'actor_obs_dim':ad,'critic_obs_dim':cd,'action_dim':actdim,'model_config':{'hidden_dim':hidden_dim,'num_layers':num_layers,'dropout':dropout,'use_privileged_critic':use_privileged_critic},'epoch':epoch,'metrics':row}
        torch.save(ckpt,ckpt_dir/'ppo_asym_last.pt')
        score=float(val.get('point_edge_accuracy',val.get('point_action_accuracy',0.0)))
        if score>best: best=score; torch.save(ckpt,ckpt_dir/'ppo_asym_best.pt')
    with (rep_dir/'ppo_training_report.json').open('w',encoding='utf-8',newline='\n') as f: json.dump(hist,f,indent=2)
    return model,hist
