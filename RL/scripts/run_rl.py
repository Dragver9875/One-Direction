from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
STAGES={'check':['RL/scripts/00_check_inputs.py'],'bc':['RL/scripts/01_train_bc_actor.py'],'ppo':['RL/scripts/02_train_ppo_asym.py'],'eval':['RL/scripts/03_evaluate_ppo_asym.py']}
ALIASES={'smoke':['check','bc','ppo','eval'],'all':['check','bc','ppo','eval'],'train':['bc','ppo']}

def resolve(items):
    out=[]
    for item in items:
        if item in ALIASES: out.extend(ALIASES[item])
        elif item in STAGES: out.append(item)
        else: raise ValueError(f'Unknown RL stage: {item}')
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument('stages',nargs='*',default=['all']); p.add_argument('--config',default='RL/configs/ppo_default.yaml'); p.add_argument('--dry-run',action='store_true'); p.add_argument('--list',action='store_true'); args=p.parse_args()
    if args.list:
        print('Stages:'); [print(f'  {k}') for k in STAGES]; print('Aliases:'); [print(f'  {k}: {", ".join(v)}') for k,v in ALIASES.items()]; return 0
    root=Path(__file__).resolve().parents[2]
    for stage in resolve(args.stages):
        cmd=[sys.executable,*STAGES[stage],'--config',args.config]; print(' '.join(cmd))
        if not args.dry_run:
            r=subprocess.run(cmd,cwd=root)
            if r.returncode!=0: return r.returncode
    return 0
if __name__=='__main__': raise SystemExit(main())
