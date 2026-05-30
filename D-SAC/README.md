# One-Direction D-SAC

Discrete Soft Actor-Critic with Asymmetric Privileged Learning for One-Direction map matching.

The branch treats map matching as a discrete sequential decision problem.

```text
episode = one vehicle trajectory
step    = one GPS point
action  = choose one candidate OSM road segment
```

Actor input is deployment-safe: candidate features, candidate mask, previous selected candidate, and timestep fraction.

Critic input is asymmetric: the critics receive the actor observation plus privileged training-only hints derived from GT candidate position, projection-to-GT distance, previous/next GT transition hints, and candidate validity. At inference, only the actor is used.

## Required V1 artifacts

Run the GNN-HMM data pipeline through tensor generation first:

```powershell
python scripts\run_project.py data
```

Required:

```text
data/processed/tensors/train_dataset.pt
data/processed/tensors/val_dataset.pt
data/processed/tensors/test_dataset.pt
```

## Install

```powershell
pip install -r D-SAC\requirements-dsac.txt
```

## Debug run

```powershell
python D-SAC\scripts\run_dsac.py all --config D-SAC\configs\dsac_debug.yaml
```

## Full run

```powershell
python D-SAC\scripts\run_dsac.py all --config D-SAC\configs\dsac_default.yaml
```

## Manual stages

```powershell
python D-SAC\scripts\00_check_inputs.py --config D-SAC\configs\dsac_default.yaml
python D-SAC\scripts\01_train_bc_actor.py --config D-SAC\configs\dsac_default.yaml
python D-SAC\scripts\02_train_discrete_sac_asym.py --config D-SAC\configs\dsac_default.yaml
python D-SAC\scripts\03_evaluate_discrete_sac.py --config D-SAC\configs\dsac_default.yaml
```

## Outputs

```text
D-SAC/outputs/checkpoints/bc_actor.pt
D-SAC/outputs/checkpoints/dsac_asym_best.pt
D-SAC/outputs/checkpoints/dsac_asym_last.pt
D-SAC/outputs/matches/dsac_asym_matches.parquet
D-SAC/outputs/metrics/dsac_asym_metrics.json
D-SAC/outputs/reports/dsac_training_report.json
```
