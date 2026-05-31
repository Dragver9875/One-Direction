# One-Direction HMM

Classical HMM/Viterbi workflow for One-Direction.

This branch is a deterministic, non-neural baseline that consumes the same candidate tensors used by the GNN-HMM pipeline.

```text
episode = one vehicle trajectory
state   = one candidate OSM road segment at timestep t
emission score = local GPS/yaw/candidate compatibility
transition score = road-to-road movement plausibility
decoder = Viterbi
```

The goal is to measure how much performance can be obtained from a carefully shaped HMM before adding GNN road embeddings.

## Required inputs

Run the main One-Direction data pipeline first:

```powershell
python scripts\run_project.py data
```

Required files:

```text
data/processed/tensors/train_dataset.pt
data/processed/tensors/val_dataset.pt
data/processed/tensors/test_dataset.pt
```

## Install

```powershell
pip install -r HMM\requirements-hmm.txt
```

## Run

Debug validation run:

```powershell
python HMM\scripts\run_hmm.py all --config HMM\configs\hmm_debug.yaml
```

Full test run:

```powershell
python HMM\scripts\run_hmm.py all --config HMM\configs\hmm_default.yaml
```

Manual stages:

```powershell
python HMM\scripts\00_check_inputs.py --config HMM\configs\hmm_default.yaml

python HMM\scripts\01_decode_hmm.py `
  --config HMM\configs\hmm_default.yaml `
  --split test

python HMM\scripts\02_evaluate_hmm.py `
  --config HMM\configs\hmm_default.yaml `
  --split test
```

Optional grid tuning on validation:

```powershell
python HMM\scripts\03_tune_hmm_grid.py --config HMM\configs\hmm_default.yaml
```

## Outputs

```text
HMM/outputs/matches/hmm_matches_val.parquet
HMM/outputs/matches/hmm_matches_test.parquet
HMM/outputs/metrics/hmm_metrics_val.json
HMM/outputs/metrics/hmm_metrics_test.json
HMM/outputs/metrics/hmm_error_cases_test.csv
HMM/outputs/metrics/hmm_trajectory_metrics_test.csv
HMM/outputs/metrics/hmm_grid_search.csv
HMM/outputs/metrics/hmm_best_params.json
```

## HMM scoring

Emission score rewards:

```text
small GPS-to-road distance
small yaw mismatch
good candidate rank
reasonable speed consistency
one-way/direction compatibility
```

Transition score rewards:

```text
legal graph transition
same-edge continuation
same OSM-way continuity
same road-class continuity
time feasibility
small route-vs-GPS mismatch
small turn/yaw-change penalty
```

Viterbi then recovers the globally best route over the candidate lattice.

## Recommended comparison

Compare this HMM branch against:

```text
raw candidate top-1
HMM baseline
GNN-HMM + Viterbi
PPO + Asym PL
D-SAC + Asym PL
```

The HMM branch is useful as an interpretable baseline. If GNN-HMM only slightly improves over HMM, then handcrafted geometry/transition features are doing most of the work. If GNN-HMM strongly improves over HMM, then learned graph embeddings are adding meaningful road-network context.
