from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import torch
from .data import RLDataset
from .features import action_mask, actor_observation, privileged_observation
from .models import PPOActorCritic
from .reward import legal_transition, projection_error_m


def build_model_from_checkpoint(path: str | Path, device='cpu') -> PPOActorCritic:
    ckpt=torch.load(path,map_location=device,weights_only=False); cfg=ckpt.get('model_config',{})
    model=PPOActorCritic(int(ckpt['actor_obs_dim']),int(ckpt['critic_obs_dim']),int(ckpt['action_dim']),int(cfg.get('hidden_dim',256)),int(cfg.get('num_layers',2)),float(cfg.get('dropout',0.1)),bool(cfg.get('use_privileged_critic',True)))
    model.load_state_dict(ckpt['model_state_dict']); return model


@torch.no_grad()
def evaluate_policy(model:PPOActorCritic,dataset:RLDataset,device='cpu',greedy=True,k_max=None):
    model.to(device); model.eval(); rows=[]
    for sample in dataset.episodes:
        prev=None
        for t in range(sample.length):
            ao=actor_observation(sample,t,prev,k_max).to(device); co=privileged_observation(sample,t,prev,k_max).to(device); m=action_mask(sample,t,k_max).to(device)
            action,_,_,value=model.act(ao,co,m,greedy=greedy); a=int(action.item()); gt=int(sample.gt_candidate_pos[t].item())
            pred_edge=int(sample.candidate_edge_idx[t,a].item()) if a<sample.num_candidates else -1
            gt_edge=int(sample.candidate_edge_idx[t,gt].item()) if 0<=gt<sample.num_candidates else -1
            dist=model.distribution(ao,m); probs=torch.softmax(dist.logits,dim=-1).squeeze(0); conf=float(probs[a].detach().cpu()) if a<probs.numel() else 0.0
            err=projection_error_m(sample,t,a); legal=legal_transition(sample,t,prev,a)
            row={'trajectory_id':int(sample.trajectory_id),'t':int(t),'pred_candidate_pos':a,'gt_candidate_pos':gt,'pred_edge_idx':pred_edge,'gt_edge_idx':gt_edge,'correct':int(a==gt),'legal_transition':int(legal),'confidence':conf,'value':float(value.detach().cpu()),'projection_error_m':err}
            if sample.candidate_proj_xy is not None and a<sample.num_candidates:
                row['pred_proj_x']=float(sample.candidate_proj_xy[t,a,0]); row['pred_proj_y']=float(sample.candidate_proj_xy[t,a,1])
            if sample.gt_proj_xy is not None:
                row['gt_proj_x']=float(sample.gt_proj_xy[t,0]); row['gt_proj_y']=float(sample.gt_proj_xy[t,1])
            rows.append(row); prev=a
    df=pd.DataFrame(rows)
    metrics={'num_points':int(len(df)),'num_trajectories':int(df['trajectory_id'].nunique()) if len(df) else 0,'point_action_accuracy':float(df['correct'].mean()) if len(df) else 0.0,'point_edge_accuracy':float((df['pred_edge_idx']==df['gt_edge_idx']).mean()) if len(df) else 0.0,'legal_transition_rate':float(df[df['t']>0]['legal_transition'].mean()) if len(df[df['t']>0]) else 1.0,'mean_projection_error_m':float(df['projection_error_m'].mean()) if len(df) else 0.0,'median_projection_error_m':float(df['projection_error_m'].median()) if len(df) else 0.0,'p90_projection_error_m':float(df['projection_error_m'].quantile(0.90)) if len(df) else 0.0,'within_5m_rate':float((df['projection_error_m']<=5.0).mean()) if len(df) else 0.0,'mean_confidence':float(df['confidence'].mean()) if len(df) else 0.0}
    return df,metrics


def save_eval_outputs(matches,metrics,match_path,metric_path):
    match_path=Path(match_path); metric_path=Path(metric_path); match_path.parent.mkdir(parents=True,exist_ok=True); metric_path.parent.mkdir(parents=True,exist_ok=True)
    matches.to_parquet(match_path,index=False)
    with metric_path.open('w',encoding='utf-8',newline='\n') as f: json.dump(metrics,f,indent=2)
