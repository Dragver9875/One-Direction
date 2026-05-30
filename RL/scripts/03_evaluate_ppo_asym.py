from __future__ import annotations
import argparse
from pathlib import Path
from _bootstrap import add_rl_src_to_path
add_rl_src_to_path()
from onedir_ppo.config import deep_get, load_config, resolve_device
from onedir_ppo.data import load_rl_dataset
from onedir_ppo.evaluate import build_model_from_checkpoint, evaluate_policy, save_eval_outputs

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=Path('RL/configs/ppo_default.yaml')); p.add_argument('--checkpoint',type=Path,default=Path('RL/outputs/checkpoints/ppo_asym_best.pt')); p.add_argument('--split',choices=['train','val','test'],default=None); a=p.parse_args(); cfg=load_config(a.config); device=resolve_device(deep_get(cfg,'project.device','auto'))
    split=a.split or deep_get(cfg,'evaluation.split','test'); ds=load_rl_dataset(deep_get(cfg,f'paths.{split}_dataset')); model=build_model_from_checkpoint(a.checkpoint,device=device)
    matches,metrics=evaluate_policy(model,ds,device=device,greedy=bool(deep_get(cfg,'evaluation.greedy',True)),k_max=deep_get(cfg,'features.max_candidates',None))
    save_eval_outputs(matches,metrics,Path(deep_get(cfg,'paths.match_dir','RL/outputs/matches'))/'ppo_asym_matches.parquet',Path(deep_get(cfg,'paths.metric_dir','RL/outputs/metrics'))/'ppo_asym_metrics.json')
    print('[OK] PPO asymmetric evaluation complete')
    for k,v in metrics.items(): print(f'{k}: {v}')
if __name__=='__main__': main()
