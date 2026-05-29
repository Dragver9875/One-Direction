# One-Direction

**One-Direction** is a GNN-enhanced HMM map-matching system that maps vehicle-acquired pose observations onto corresponding directed road segments in an OpenStreetMap road network.

The project focuses on the following problem:

```text
Input:
    Vehicle pose sequence: (x, y, yaw)
    OpenStreetMap road network

Output:
    Matched OSM road segment per timestep
    Projected point on the matched road segment
    Confidence score for each match
    Globally consistent matched road-segment sequence
```

Unlike a nearest-road matcher, One-Direction treats map matching as a **graph-structured sequence-decoding problem**. The OSM road network is converted into a directed road graph, then into an edge-centric line graph. A Graph Neural Network learns contextual road-segment embeddings, and a custom HMM/Viterbi decoder uses learned emission and transition scores to recover the most likely route.

---

## 1. Core idea

For each vehicle observation:

$$
z_t = (x_t, y_t, \psi_t)
$$

where:

* $x_t, y_t$ are projected vehicle coordinates in meters.
* $\psi_t$ is the vehicle yaw or heading.
* $t$ is the trajectory timestep.

The goal is to predict:

$$
\hat{y}_t = (e_t, p_t, c_t)
$$

where:

* $e_t$ is the matched directed OSM road segment.
* $p_t$ is the projected point on that segment.
* $c_t$ is the confidence score.

The final decoded road-segment sequence is obtained by solving:

$$
\hat{e}_{1:T}
=
\arg\max_{e_{1:T}}
\left[
\sum_{t=1}^{T} E_\theta(z_t, e_t)
+
\sum_{t=2}^{T} T_\phi(e_{t-1}, e_t, z_{t-1}, z_t)
\right]
$$

where:

* $E_\theta$ is the learned emission score.
* $T_\phi$ is the learned transition score.
* $\hat{e}_{1:T}$ is the globally decoded route over candidate OSM road segments.

---

## 2. Why GNN-HMM?

Map matching is not just nearest-road search. A road can be spatially close to a GPS point but still be incorrect due to:

* parallel roads,
* one-way constraints,
* flyovers and underpasses,
* highway/service-road ambiguity,
* ramps and slip roads,
* roundabouts,
* sparse GPS sampling,
* yaw noise,
* GPS drift,
* dense intersections,
* incomplete or inconsistent OSM metadata.

One-Direction combines three ideas:

```text
1. Graph Neural Network
   Learns road-segment embeddings from OSM topology.

2. Learned emission scoring
   Scores how well each GPS/yaw observation matches each candidate road segment.

3. Learned transition scoring + HMM decoding
   Scores road-to-road movement plausibility and uses Viterbi decoding to recover the best global path.
```

This makes the system both **learned** and **topology-aware**.

---

## 3. System overview

```text
Vehicle pose trajectory
    ↓
Trajectory preprocessing
    ↓
Coordinate projection
    ↓
Yaw/speed/step feature derivation
    ↓
Candidate road-segment generation
    ↓
Candidate feature computation

OSM PBF
    ↓
Directed road graph construction
    ↓
Edge table and node table generation
    ↓
Edge-centric line graph construction
    ↓
Road-segment feature extraction
    ↓
GNN road encoder
    ↓
Road-segment embeddings

Candidate features + road embeddings
    ↓
Emission head
    ↓
Emission scores

Transition features + road embeddings
    ↓
Transition head
    ↓
Transition scores

Emission scores + transition scores
    ↓
Custom HMM/Viterbi decoder
    ↓
Matched road-segment sequence
    ↓
Projected points + confidence scores
    ↓
Evaluation + visual debugging
```

---

## 4. Repository structure

```text
One-Direction/
│
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── default.yaml
│   ├── data.yaml
│   ├── model.yaml
│   ├── train.yaml
│   ├── eval.yaml
│   └── local.yaml
│
├── data/
│   ├── raw/
│   │   ├── osm/
│   │   │   └── oberfranken-latest.osm.pbf
│   │   └── trajectories/
│   │       ├── points.csv
│   │       └── ground_truth.csv
│   │
│   ├── interim/
│   │   ├── trajectory_clean.parquet
│   │   └── gt_routes_projected.parquet
│   │
│   ├── processed/
│   │   ├── road_graph/
│   │   │   ├── road_graph.pkl
│   │   │   ├── edge_table.parquet
│   │   │   ├── node_table.parquet
│   │   │   ├── edges.geojson
│   │   │   └── nodes.geojson
│   │   │
│   │   ├── line_graph/
│   │   │   ├── line_graph.pt
│   │   │   ├── line_edge_index.pt
│   │   │   ├── segment_features.pt
│   │   │   ├── edge_id_to_idx.json
│   │   │   ├── idx_to_edge_id.json
│   │   │   └── transition_table.parquet
│   │   │
│   │   ├── candidates/
│   │   │   ├── candidates_train.parquet
│   │   │   ├── candidates_val.parquet
│   │   │   ├── candidates_test.parquet
│   │   │   └── candidate_recall_report.json
│   │   │
│   │   ├── tensors/
│   │   │   ├── train_dataset.pt
│   │   │   ├── val_dataset.pt
│   │   │   └── test_dataset.pt
│   │   │
│   │   └── splits/
│   │       ├── train_ids.txt
│   │       ├── val_ids.txt
│   │       └── test_ids.txt
│   │
│   └── reports/
│       ├── preprocessing_report.json
│       ├── gt_route_report.json
│       ├── candidate_report.json
│       ├── tensor_report.json
│       ├── gnn_hmm_data_debug_report.json
│       └── training_report.json
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   │
│   ├── data/
│   ├── graph/
│   ├── geometry/
│   ├── models/
│   ├── decoder/
│   ├── training/
│   ├── evaluation/
│   └── utils/
│
├── scripts/
│   ├── 01_prepare_trajectories.py
│   ├── 02_prepare_gt_routes.py
│   ├── 03_build_osm_graph.py
│   ├── 04_build_line_graph.py
│   ├── 05_generate_candidates.py
│   ├── 06_build_training_tensors.py
│   ├── 07_train_gnn_hmm.py
│   ├── 08_decode_gnn_hmm.py
│   ├── 09_evaluate.py
│   ├── 10_visualize_errors.py
│   ├── 11_visualize_osm_overlay.py
│   ├── 12_debug_gnn_hmm_data.py
│   ├── 13_run_gnn_hmm_experiments.py
│   └── run_project.py
│
├── notebooks/
│   ├── 01_inspect_trajectories.ipynb
│   ├── 02_inspect_osm_graph.ipynb
│   ├── 03_debug_candidates.ipynb
│   ├── 04_debug_decoder.ipynb
│   └── 05_error_analysis.ipynb
│
├── outputs/
│   ├── checkpoints/
│   │   ├── gnn_hmm_best.pt
│   │   └── gnn_hmm_last.pt
│   │
│   ├── matches/
│   │   └── gnn_hmm_matches.parquet
│   │
│   ├── metrics/
│   │   ├── gnn_hmm_metrics.json
│   │   ├── trajectory_metrics.csv
│   │   ├── error_cases.csv
│   │   └── gnn_hmm_experiment_summary.csv
│   │
│   ├── figures/
│   │   ├── matched_paths.png
│   │   ├── error_cases.png
│   │   └── osm_overlay.html
│   │
│   └── run_logs/
│
├── RL/
│   └── future PPO + asymmetric privileged learning extension
│
└── tests/
    ├── test_geometry.py
    ├── test_candidate_search.py
    ├── test_line_graph.py
    ├── test_viterbi.py
    ├── test_emission_head.py
    ├── test_transition_head.py
    ├── test_training_tensors.py
    ├── test_gnn_hmm.py
    └── test_config.py
```

---

## 5. Data format

### 5.1 Vehicle trajectory input

Expected file:

```text
data/raw/trajectories/points.csv
```

Expected content:

```text
trajectory_id
timestamp
x / y or latitude / longitude depending on dataset format
yaw
```

The preprocessing script projects coordinates into a local metric coordinate system, sorts the points by trajectory and time, computes motion features, and writes a clean trajectory file.

Output:

```text
data/interim/trajectory_clean.parquet
```

---

### 5.2 Ground-truth route input

Expected file:

```text
data/raw/trajectories/ground_truth.csv
```

The ground-truth file is parsed into trajectory-level route geometry and projected into the same metric CRS as the vehicle observations.

Output:

```text
data/interim/gt_routes_projected.parquet
```

The GT routes are then used to mark which generated candidate road segment is correct for each timestep.

---

### 5.3 OSM input

Expected file:

```text
data/raw/osm/oberfranken-latest.osm.pbf
```

The OSM PBF is parsed into a directed road network. Each legal travel direction is represented as a separate directed edge.

Example:

```text
Two-way road:
    A → B
    B → A

One-way road:
    only the legal direction is retained
```

---

## 6. Directed road graph

The directed road graph represents the physical road network.

```text
node = OSM intersection / endpoint / geometry node
edge = directed road segment
```

Each edge stores information such as:

```text
edge_id
edge_idx
osm_way_id
u
v
geometry
length_m
bearing_rad
road_class
oneway
maxspeed
lanes
bridge
tunnel
direction
```

The directed representation is important because yaw-sensitive map matching requires direction-aware road segments.

A vehicle moving north should not be treated identically to a vehicle moving south on the same physical road, especially on one-way roads, ramps, divided roads, and junctions.

---

## 7. Edge-centric line graph

One-Direction converts the road graph into a line graph.

```text
Original road graph:
    node = intersection
    edge = road segment

Line graph:
    node = directed road segment
    edge = legal transition between road segments
```

Example:

```text
Road graph:
    A --e1--> B --e2--> C

Line graph:
    e1 --> e2
```

This representation is central to the model because the hidden HMM state is a road segment:

$$
s_t = e_t
$$

Therefore, every line-graph node corresponds directly to one possible hidden state.

---

## 8. Road-segment features

Each road segment receives a feature vector.

Typical road features include:

```text
length_m
log_length_m
sin_bearing
cos_bearing
road_class_id
oneway_flag
maxspeed_norm
lanes_norm
curvature
in_degree
out_degree
bridge_flag
tunnel_flag
```

Angles are represented using sine and cosine instead of raw degrees or radians.

Reason:

```text
359° and 1° are physically close directions,
but as raw numbers they look far apart.
```

Using sine and cosine avoids angle wraparound errors.

---

## 9. Candidate generation

For every vehicle observation:

$$
z_t = (x_t, y_t, \psi_t)
$$

One-Direction finds nearby road-segment candidates:

$$
C_t = \{e_{t,1}, e_{t,2}, \ldots, e_{t,k}\}
$$

Each candidate stores:

```text
trajectory_id
t
timestamp
edge_idx
edge_id
distance_m
yaw_diff_rad
proj_x
proj_y
offset_m
offset_ratio
candidate_rank
is_gt
```

The candidate generator is a critical stage. If the true road segment is not present in the candidate set, the model cannot select it later.

Candidate recall is therefore used as a preprocessing diagnostic.

---

## 10. Training tensor construction

The tensor-building stage converts candidate tables into model-ready trajectory tensors.

Output files:

```text
data/processed/tensors/train_dataset.pt
data/processed/tensors/val_dataset.pt
data/processed/tensors/test_dataset.pt
```

Each trajectory tensor contains:

```text
trajectory_id
timesteps
timestamps
xy
yaw
speed_mps

candidate_edge_idx
candidate_mask
candidate_proj_xy

emission_features
transition_features
transition_mask

gt_candidate_pos
gt_edge_idx
gt_proj_xy

emission_feature_names
transition_feature_names
```

The improved tensor builder includes richer emission and transition features.

Emission features describe how well one GPS/yaw observation matches one candidate segment.

Transition features describe how plausible it is to move from one candidate segment to another between consecutive timesteps.

---

## 11. Model architecture

One-Direction contains four main model components:

```text
1. Road GNN encoder
2. Emission head
3. Transition head
4. HMM/Viterbi decoder
```

---

### 11.1 Road GNN encoder

The road GNN operates on the edge-centric line graph.

Input:

```text
line_graph.pt
segment_features.pt
```

The encoder computes an embedding for every directed road segment:

$$
h_e = \mathrm{GNN}(x_e, G_{\text{line}})
$$

where:

* $x_e$ is the road-segment feature vector.
* $G_{\text{line}}$ is the line graph.
* $h_e$ is the learned road-segment embedding.

Default encoder:

```text
GraphSAGE
```

GraphSAGE is used because it is stable, scalable, and suitable as a first road-network encoder.

---

### 11.2 Emission head

The emission head answers:

```text
How well does GPS/yaw observation z_t match candidate road segment e?
```

For each candidate pair $(z_t, e)$, the model combines:

```text
road-segment embedding
distance to road
yaw difference
offset along road
vehicle speed
road bearing
candidate rank
road metadata
```

The emission score is:

$$
E_\theta(z_t, e) = \mathrm{MLP}_{\text{emission}}(r_{t,e})
$$

where $r_{t,e}$ is the candidate-specific feature vector.

---

### 11.3 Transition head

The transition head answers:

```text
How plausible is it to move from road segment e_i at timestep t-1 to road segment e_j at timestep t?
```

For each candidate transition $(e_i, e_j)$, the model combines:

```text
previous road embedding
current road embedding
GPS displacement
time gap
observed speed
route-distance proxy
route-vs-GPS distance mismatch
turn angle
yaw change
same-edge flag
same-OSM-way flag
same-road-class flag
legal-transition flag
time-feasibility flag
```

The transition score is:

$$
T_\phi(e_i, e_j, z_{t-1}, z_t)
=
\mathrm{MLP}_{\text{transition}}(q_{i,j,t})
$$

The transition model is learned, but impossible or highly implausible transitions can still be penalized during decoding.

---

### 11.4 HMM/Viterbi decoder

For each trajectory, the decoder receives:

```text
candidate sets:
    C_1, C_2, ..., C_T

emission scores:
    E_t(e), for e ∈ C_t

transition scores:
    T_t(e_prev, e_curr), for e_prev ∈ C_{t-1}, e_curr ∈ C_t
```

The decoder solves:

$$
\hat{e}_{1:T}
=
\arg\max_{e_t \in C_t}
\left[
\sum_{t=1}^{T} E_t(e_t)
+
\sum_{t=2}^{T} T_t(e_{t-1}, e_t)
\right]
$$

Viterbi recurrence:

$$
V_1(e) = E_1(e)
$$

$$
V_t(e) =
E_t(e)
+
\max_{e' \in C_{t-1}}
\left[
V_{t-1}(e') + T_t(e', e)
\right]
$$

Backpointer:

$$
B_t(e) =
\arg\max_{e' \in C_{t-1}}
\left[
V_{t-1}(e') + T_t(e', e)
\right]
$$

The final road segment is:

$$
\hat{e}_T = \arg\max_e V_T(e)
$$

The full sequence is recovered by backtracking through the stored backpointers.

---

## 12. Training objective

The GNN-HMM model trains emission and transition scores jointly.

Given a ground-truth candidate sequence:

$$
e_1^{GT}, e_2^{GT}, \ldots, e_T^{GT}
$$

the training objective contains:

```text
1. Emission loss
2. Transition loss
3. Optional hard-negative margin loss
4. Optional label smoothing
```

---

### 12.1 Emission loss

For each GPS observation, the model should rank the GT candidate above the other candidates:

$$
\mathcal{L}_{\text{emission}}
=
-\sum_t
\log
\frac{
\exp(E_t(e_t^{GT}))
}{
\sum_{e \in C_t} \exp(E_t(e))
}
$$

---

### 12.2 Transition loss

For each consecutive GT transition, the model should rank the correct next candidate above alternative next candidates:

$$
\mathcal{L}_{\text{transition}}
=
-\sum_{t=2}^{T}
\log
\frac{
\exp(T_t(e_{t-1}^{GT}, e_t^{GT}))
}{
\sum_{e \in C_t} \exp(T_t(e_{t-1}^{GT}, e))
}
$$

---

### 12.3 Hard-negative margin loss

The emission loss can include a margin term that forces the GT candidate to score higher than the hardest wrong candidate:

$$
\mathcal{L}_{\text{margin}}
=
\max(0, m - s_{GT} + s_{\text{hard negative}})
$$

This helps the model learn difficult cases such as nearby parallel roads, service roads, ramps, and reverse-direction candidates.

---

### 12.4 Total loss

$$
\mathcal{L}
=
\lambda_E \mathcal{L}_{\text{emission}}
+
\lambda_T \mathcal{L}_{\text{transition}}
+
\lambda_M \mathcal{L}_{\text{margin}}
$$

The current training script exposes these weights as command-line arguments.

---

## 13. Evaluation metrics

The evaluation script reports both strict edge-ID metrics and geometry-aware diagnostics.

Important metric categories:

```text
Exact edge identity:
    point_edge_accuracy

Spatial quality:
    mean_projection_error_m
    median_projection_error_m
    p90_projection_error_m
    within_2m_rate
    within_5m_rate
    within_10m_rate

Road-level semantic consistency:
    same_osm_way_accuracy
    same_road_class_accuracy
    same_undirected_uv_accuracy

Sequence quality:
    path_edit_distance_mean
    path_edit_distance_median
    trajectory_success_rate

Transition quality:
    pred_transition_legal_rate
    gt_transition_legal_rate

Error taxonomy:
    error_near_but_wrong_edge_rate
    error_same_way_rate
    error_reverse_edge_rate
    error_severe_rate
```

Exact edge-ID accuracy is strict. It can count a prediction as wrong even if the predicted segment lies on the same physical road but has a different OSM edge ID.

For this reason, geometry-aware and same-way diagnostics are included to distinguish:

```text
true wrong-road errors
same-road split-segment errors
reverse-direction errors
nearby parallel-road errors
illegal-transition errors
GT/OSM alignment issues
```

---

## 14. Visualization

One-Direction includes two visualization modes.

### 14.1 Static error visualization

Script:

```text
scripts/10_visualize_errors.py
```

Purpose:

```text
Plot predicted path, GT-derived path, OSM graph, and error points.
```

Output:

```text
outputs/figures/
```

### 14.2 Interactive OSM overlay

Script:

```text
scripts/11_visualize_osm_overlay.py
```

Purpose:

```text
Overlay extracted road graph, predicted path, GT-derived path, and GPS points over an OpenStreetMap basemap.
```

This is used to verify:

```text
extracted graph alignment with actual OSM tiles
CRS correctness
prediction plausibility
GT/OSM map-version mismatch
same-road vs wrong-road errors
```

Example:

```powershell
python scripts\11_visualize_osm_overlay.py --trajectory-id 44
```

Generate a combined overlay:

```powershell
python scripts\11_visualize_osm_overlay.py
```

---

## 15. Unified pipeline runner

The project includes a unified runner:

```text
scripts/run_project.py
```

List available stages:

```powershell
python scripts\run_project.py --list
```

Dry-run full GNN-HMM pipeline:

```powershell
python scripts\run_project.py gpu_e2e --dry-run
```

Run full GNN-HMM pipeline:

```powershell
python scripts\run_project.py gpu_e2e
```

Run from a specific stage:

```powershell
python scripts\run_project.py gpu_e2e --from-stage train
```

Run only data preparation:

```powershell
python scripts\run_project.py data
```

Run only evaluation and visualization:

```powershell
python scripts\run_project.py post
```

---

## 16. Manual pipeline commands

Run commands from the repository root:

```powershell
cd E:\PANDA\One-Direction
```

### 16.1 Prepare trajectories

```powershell
python scripts\01_prepare_trajectories.py
```

### 16.2 Prepare GT routes

```powershell
python scripts\02_prepare_gt_routes.py
```

### 16.3 Build directed OSM graph

```powershell
python scripts\03_build_osm_graph.py
```

### 16.4 Build line graph

```powershell
python scripts\04_build_line_graph.py
```

### 16.5 Generate candidates

```powershell
python scripts\05_generate_candidates.py
```

### 16.6 Build improved training tensors

```powershell
python scripts\06_build_training_tensors.py --transition-mask-mode all
```

Available transition mask modes:

```text
all      Keep all candidate-to-candidate transitions available.
legal    Keep only graph-legal or same-edge transitions.
speed    Keep graph-legal transitions that also satisfy a speed-feasibility check.
```

The recommended first mode is:

```text
all
```

After verifying GT transition validity, stricter modes can be tested.

### 16.7 Run data-debug checks

```powershell
python scripts\12_debug_gnn_hmm_data.py
```

Output:

```text
data/reports/gnn_hmm_data_debug_report.json
```

This report verifies:

```text
tensor shapes
candidate mask density
transition mask density
GT candidate-position validity
GT transition validity
edge ID uniqueness
road-class distribution
candidate recall by trajectory
```

### 16.8 Train GNN-HMM

```powershell
python scripts\07_train_gnn_hmm.py `
  --output outputs\checkpoints `
  --epochs 100 `
  --batch-size 2 `
  --lr 0.001 `
  --emission-weight 1.0 `
  --transition-weight 2.0 `
  --label-smoothing 0.02 `
  --margin-weight 0.1 `
  --margin 1.0 `
  --device auto
```

Output:

```text
outputs/checkpoints/gnn_hmm_best.pt
outputs/checkpoints/gnn_hmm_last.pt
data/reports/training_report.json
```

### 16.9 Decode

```powershell
python scripts\08_decode_gnn_hmm.py `
  --checkpoint outputs\checkpoints\gnn_hmm_best.pt `
  --output outputs\matches\gnn_hmm_matches.parquet `
  --illegal-transition-mode soft `
  --illegal-penalty 5.0 `
  --device auto
```

Available illegal-transition modes:

```text
none    Do not adjust illegal transitions.
soft    Apply a finite penalty.
hard    Mask illegal transitions with a very large negative score.
```

Recommended first mode:

```text
soft
```

### 16.10 Evaluate

```powershell
python scripts\09_evaluate.py
```

Output:

```text
outputs/metrics/gnn_hmm_metrics.json
outputs/metrics/trajectory_metrics.csv
outputs/metrics/error_cases.csv
```

### 16.11 Visualize static errors

```powershell
python scripts\10_visualize_errors.py
```

### 16.12 Generate OSM overlay

```powershell
python scripts\11_visualize_osm_overlay.py
```

---

## 17. Local GPU workflow

### 17.1 Activate environment

If Conda is available directly:

```powershell
conda activate one-direction
```

If PowerShell cannot find `conda`, initialize the shell hook manually:

```powershell
& "E:\Anaconda3\shell\condabin\conda-hook.ps1"
conda activate one-direction
```

Or run with the environment Python directly:

```powershell
E:\Anaconda3\envs\one-direction\python.exe scripts\run_project.py gpu_e2e
```

### 17.2 Set geospatial environment variables

```powershell
$env:PYTHONPATH="E:\PANDA\One-Direction"
$env:GDAL_DATA="$env:CONDA_PREFIX\Library\share\gdal"
$env:PROJ_LIB="$env:CONDA_PREFIX\Library\share\proj"
```

### 17.3 Verify GPU and dependencies

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -c "import geopandas, shapely, pyproj, osmnx; print('geo stack ok')"
python -c "import src; print(src.__project__, src.__version__)"
```

### 17.4 Start end-to-end run

```powershell
python scripts\run_project.py gpu_e2e
```

---

## 18. Debugging checklist

Before judging model quality, verify the following:

```text
1. Raw trajectory points are sorted correctly by trajectory and timestamp.
2. Coordinates are projected into a metric CRS.
3. OSM graph aligns with the actual OSM basemap.
4. Road directionality is preserved.
5. Two-way roads have forward and reverse directed edges.
6. One-way roads do not incorrectly receive reverse edges.
7. Candidate generation includes the GT road segment.
8. GT candidate position is valid in tensors.
9. Transition mask does not incorrectly reject GT transitions.
10. Edge IDs and edge indices are unique and stable.
11. Prediction errors are classified into true wrong-road, same-way, reverse-edge, and near-geometry cases.
```

Important debug scripts:

```powershell
python scripts\12_debug_gnn_hmm_data.py
python scripts\11_visualize_osm_overlay.py
```

---

## 19. Recommended development order

```text
1. Prepare and inspect trajectory data.
2. Build directed OSM road graph.
3. Verify graph alignment with OSM overlay.
4. Build edge-centric line graph.
5. Generate candidate road segments.
6. Verify candidate recall.
7. Build training tensors.
8. Verify tensor and transition validity.
9. Train the GNN-HMM.
10. Decode with transition-aware Viterbi.
11. Evaluate with exact, geometry-aware, same-way, and transition metrics.
12. Visualize failure cases.
13. Tune transition weight and illegal-transition penalty.
14. Improve transition features and sequence-level constraints.
15. Compare against classical and learned baselines.
```

---

## 20. Future extensions

The current focus is the **GNN-HMM map-matching pipeline**. Possible future extensions include:

```text
Graph Attention Network road encoder
Directed line-graph attention
CRF-style sequence-level training
Differentiable Viterbi / soft dynamic programming
Hard-negative mining for ambiguous candidates
Shortest-path-based transition features
Synthetic route generation from OSM
Pseudo-labelling large raw GPS datasets
Online streaming map matching
Routing-service integration
```

### Future RL extension: PPO with Asymmetric Privileged Learning

A future branch can treat map matching as a sequential decision problem:

```text
episode = one trajectory
step    = one GPS observation
action  = choose one candidate road segment
```

A PPO actor would use only deployment-safe features:

```text
GPS/yaw features
candidate features
candidate mask
previous selected candidate
```

A privileged critic could use training-only information:

```text
GT candidate position
GT route geometry
future local candidate context
same-way/reverse-edge hints
transition legality labels
```

This asymmetric setup can be used as a future sequence-level learning baseline, but it is separate from the current GNN-HMM implementation.

---

## 21. Final summary

One-Direction is a graph-based neural map-matching system for mapping vehicle pose observations to OSM road segments.

It uses:

```text
Directed OSM road graph
Edge-centric line graph
GNN road-segment encoder
Learned emission scoring
Learned transition scoring
Custom HMM/Viterbi decoder
Geometry-aware and topology-aware evaluation
```

The final output is:

```text
matched OSM road segment per timestep
projected point on matched road segment
confidence score
globally consistent matched road-segment sequence
```

The central design principle is:

```text
Use neural learning to improve road-segment and transition scoring,
while preserving graph topology and sequence consistency through HMM decoding.
```
