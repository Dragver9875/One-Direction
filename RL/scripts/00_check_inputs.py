from __future__ import annotations
import argparse
from pathlib import Path
from _bootstrap import add_rl_src_to_path
add_rl_src_to_path()
from onedir_ppo.config import deep_get, load_config
from onedir_ppo.data import load_rl_dataset

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=Path('RL/configs/ppo_default.yaml')); args=p.parse_args(); cfg=load_config(args.config)
    for split in ['train','val','test']:
        ds=load_rl_dataset(deep_get(cfg,f'paths.{split}_dataset')); print(f'[OK] {split}: {ds.summary()}')
if __name__=='__main__': main()
