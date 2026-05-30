# One-Direction RL: PPO + Asymmetric Privileged Learning

This folder implements a PPO-based reinforcement-learning branch for One-Direction.

The actor is deployment-safe: it sees only GPS/candidate/previous-action information. The critic is privileged during training: it sees actor observations plus GT-derived hints. At inference, only the actor is used.

```text
episode = one vehicle trajectory
step    = one GPS point
action  = choose candidate road segment index
```

## Required parent artifacts

Run the parent GNN-HMM pipeline through tensor generation first:

```powershell
python scripts\01_prepare_trajectories.py
python scripts\02_prepare_gt_routes.py
python scripts\03_build_osm_graph.py
python scripts\04_build_line_graph.py
python scripts\05_generate_candidates.py
python scripts\06_build_training_tensors.py
```

Required tensors:

```text
data/processed/tensors/train_dataset.pt
data/processed/tensors/val_dataset.pt
data/processed/tensors/test_dataset.pt
```

## Run

```powershell
pip install -r RL\requirements-rl.txt
python RL\scripts\00_check_inputs.py --config RL\configs\ppo_default.yaml
python RL\scripts\01_train_bc_actor.py --config RL\configs\ppo_default.yaml
python RL\scripts\02_train_ppo_asym.py --config RL\configs\ppo_default.yaml
python RL\scripts\03_evaluate_ppo_asym.py --config RL\configs\ppo_default.yaml
```

Unified runner:

```powershell
python RL\scripts\run_rl.py --list
python RL\scripts\run_rl.py all
```

Debug run:

```powershell
python RL\scripts\run_rl.py all --config RL\configs\ppo_debug.yaml
```
