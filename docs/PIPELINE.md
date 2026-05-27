# One-Direction Pipeline

## Input files

```text
data/raw/trajectories/points.csv
data/raw/trajectories/ground_truth.csv
data/raw/osm/oberfranken-latest.osm.pbf
```

## Pipeline order

```bash
python scripts/01_prepare_trajectories.py
python scripts/02_prepare_gt_routes.py
python scripts/03_build_osm_graph.py
python scripts/04_build_line_graph.py
python scripts/05_generate_candidates.py
python scripts/06_build_training_tensors.py
python scripts/07_train_gnn_hmm.py
python scripts/08_decode_gnn_hmm.py
python scripts/09_evaluate.py
python scripts/10_visualize_errors.py
```

## Critical checkpoint

Candidate generation must be checked before training.

```text
Top-5 candidate recall  > 95%
Top-10 candidate recall > 98%
```

If the true road segment is absent from the candidate set, the decoder cannot
recover it later.
