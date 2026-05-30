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

The current primary workflow is the **GNN-HMM pipeline**. The repository also contains experimental reinforcement-learning branches under `RL/` and `D-SAC/`, but these are treated as secondary research extensions and should be compared against the GNN-HMM baseline only after the GNN-HMM pipeline is fully trained and evaluated.

---

## 1. Core idea

For each vehicle observation:

> <i>z</i><sub>t</sub> = (<i>x</i><sub>t</sub>, <i>y</i><sub>t</sub>, ψ<sub>t</sub>)

where:

* <i>x</i><sub>t</sub>, <i>y</i><sub>t</sub> are projected vehicle coordinates in meters.
* ψ<sub>t</sub> is the vehicle yaw or heading.
* <i>t</i> is the trajectory timestep.

The goal is to predict:

> ŷ<sub>t</sub> = (<i>e</i><sub>t</sub>, <i>p</i><sub>t</sub>, <i>c</i><sub>t</sub>)

where:

* <i>e</i><sub>t</sub> is the matched directed OSM road segment.
* <i>p</i><sub>t</sub> is the projected point on that segment.
* <i>c</i><sub>t</sub> is the confidence score.

The final decoded road-segment sequence is obtained by solving:

> ê<sub>1:T</sub> = arg max<sub><i>e</i><sub>1:T</sub></sub> [ Σ<sub>t=1</sub><sup>T</sup> E<sub>θ</sub>(<i>z</i><sub>t</sub>, <i>e</i><sub>t</sub>) + Σ<sub>t=2</sub><sup>T</sup> T<sub>φ</sub>(<i>e</i><sub>t−1</sub>, <i>e</i><sub>t</sub>, <i>z</i><sub>t−1</sub>, <i>z</i><sub>t</sub>) ]

where:

* E<sub>θ</sub> is the learned emission score.
* T<sub>φ</sub> is the learned transition score.
* ê<sub>1:T</sub> is the globally decoded route over candidate OSM road segments.

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

The intended workflow is:

```text
Primary:
    GNN-HMM + Viterbi

Experimental:
    PPO + Asymmetric Privileged Learning

Advanced experimental:
    Discrete SAC + Asymmetric Privileged Learning
```

The GNN-HMM pipeline should be treated as the main system because it performs explicit global sequence decoding over the candidate lattice.

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
│   ├── README.md
│   ├── requirements-rl.txt
│   ├── pyproject.toml
│   ├── .gitignore
│   ├── configs/
│   │   ├── ppo_default.yaml
│   │   ├── ppo_debug.yaml
│   │   └── ppo_cpu.yaml
│   ├── scripts/
│   ├── src/
│   │   └── onedir_ppo/
│   ├── tests/
│   └── outputs/
│
├── D-SAC/
│   ├── README.md
│   ├── requirements-dsac.txt
│   ├── pyproject.toml
│   ├── .gitignore
│   ├── configs/
│   │   ├── dsac_default.yaml
│   │   ├── dsac_debug.yaml
│   │   └── dsac_cpu.yaml
│   ├── scripts/
│   ├── src/
│   │   └── onedir_dsac/
│   ├── tests/
│   └── outputs/
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

> <i>s</i><sub>t</sub> = <i>e</i><sub>t</sub>

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

> <i>z</i><sub>t</sub> = (<i>x</i><sub>t</sub>, <i>y</i><sub>t</sub>, ψ<sub>t</sub>)

One-Direction finds nearby road-segment candidates:

> C<sub>t</sub> = {<i>e</i><sub>t,1</sub>, <i>e</i><sub>t,2</sub>, …, <i>e</i><sub>t,k</sub>}

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

The current tensor builder supports transition mask modes:

```text
all      Keep all candidate-to-candidate transitions available.
legal    Keep only graph-legal or same-edge transitions.
speed    Keep graph-legal transitions that also satisfy a speed-feasibility check.
```

The recommended first setting is:

```text
all
```

Stricter transition masks should be tested only after confirming that GT transitions remain valid in the debug report.

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

> <i>h</i><sub>e</sub> = GNN(<i>x</i><sub>e</sub>, G<sub>line</sub>)

where:

* <i>x</i><sub>e</sub> is the road-segment feature vector.
* G<sub>line</sub> is the line graph.
* <i>h</i><sub>e</sub> is the learned road-segment embedding.

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

For each candidate pair (<i>z</i><sub>t</sub>, <i>e</i>), the model combines:

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

> E<sub>θ</sub>(<i>z</i><sub>t</sub>, <i>e</i>) = MLP<sub>emission</sub>(<i>r</i><sub>t,e</sub>)

where <i>r</i><sub>t,e</sub> is the candidate-specific feature vector.

---

### 11.3 Transition head

The transition head answers:

```text
How plausible is it to move from road segment e_i at timestep t-1 to road segment e_j at timestep t?
```

For each candidate transition (<i>e</i><sub>i</sub>, <i>e</i><sub>j</sub>), the model combines:

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

> T<sub>φ</sub>(<i>e</i><sub>i</sub>, <i>e</i><sub>j</sub>, <i>z</i><sub>t−1</sub>, <i>z</i><sub>t</sub>) = MLP<sub>transition</sub>(<i>q</i><sub>i,j,t</sub>)

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

> ê<sub>1:T</sub> = arg max<sub><i>e</i><sub>t</sub> ∈ C<sub>t</sub></sub> [ Σ<sub>t=1</sub><sup>T</sup> E<sub>t</sub>(<i>e</i><sub>t</sub>) + Σ<sub>t=2</sub><sup>T</sup> T<sub>t</sub>(<i>e</i><sub>t−1</sub>, <i>e</i><sub>t</sub>) ]

Viterbi recurrence:

> V<sub>1</sub>(<i>e</i>) = E<sub>1</sub>(<i>e</i>)

> V<sub>t</sub>(<i>e</i>) = E<sub>t</sub>(<i>e</i>) + max<sub><i>e</i>′ ∈ C<sub>t−1</sub></sub> [ V<sub>t−1</sub>(<i>e</i>′) + T<sub>t</sub>(<i>e</i>′, <i>e</i>) ]

Backpointer:

> B<sub>t</sub>(<i>e</i>) = arg max<sub><i>e</i>′ ∈ C<sub>t−1</sub></sub> [ V<sub>t−1</sub>(<i>e</i>′) + T<sub>t</sub>(<i>e</i>′, <i>e</i>) ]

The final road segment is:

> ê<sub>T</sub> = arg max<sub>e</sub> V<sub>T</sub>(<i>e</i>)

The full sequence is recovered by backtracking through the stored backpointers.

---

## 12. Training objective

The GNN-HMM model trains emission and transition scores jointly.

Given a ground-truth candidate sequence:

> <i>e</i><sub>1</sub><sup>GT</sup>, <i>e</i><sub>2</sub><sup>GT</sup>, …, <i>e</i><sub>T</sub><sup>GT</sup>

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

> L<sub>emission</sub> = −Σ<sub>t</sub> log( exp(E<sub>t</sub>(<i>e</i><sub>t</sub><sup>GT</sup>)) / Σ<sub>e ∈ C<sub>t</sub></sub> exp(E<sub>t</sub>(<i>e</i>)) )

---

### 12.2 Transition loss

For each consecutive GT transition, the model should rank the correct next candidate above alternative next candidates:

> L<sub>transition</sub> = −Σ<sub>t=2</sub><sup>T</sup> log( exp(T<sub>t</sub>(<i>e</i><sub>t−1</sub><sup>GT</sup>, <i>e</i><sub>t</sub><sup>GT</sup>)) / Σ<sub>e ∈ C<sub>t</sub></sub> exp(T<sub>t</sub>(<i>e</i><sub>t−1</sub><sup>GT</sup>, <i>e</i>)) )

---

### 12.3 Hard-negative margin loss

The emission loss can include a margin term that forces the GT candidate to score higher than the hardest wrong candidate:

> L<sub>margin</sub> = max(0, m − s<sub>GT</sub> + s<sub>hard negative</sub>)

This helps the model learn difficult cases such as nearby parallel roads, service roads, ramps, and reverse-direction candidates.

---

### 12.4 Total loss

> L = λ<sub>E</sub>L<sub>emission</sub> + λ<sub>T</sub>L<sub>transition</sub> + λ<sub>M</sub>L<sub>margin</sub>

The current training script exposes these weights as command-line arguments.

---

## 13. Training implementation details

The current training script is:

```text
scripts/07_train_gnn_hmm.py
```

It includes several important training safeguards:

```text
1. Mask-aware cross-entropy
2. Mask-aware label smoothing
3. Valid-candidate-only hard-negative margin loss
4. Separate emission and transition loss logging
5. Viterbi candidate accuracy tracking
6. Live PowerShell logging through unbuffered execution
7. Best-checkpoint selection by validation Viterbi candidate accuracy
```

The mask-aware loss is important because invalid or padded candidates are assigned very negative scores. Standard label smoothing over those masked classes can create unstable loss values. The patched training script distributes label-smoothing probability only over valid candidates.

The logged training fields include:

```text
loss
emission_loss
emission_ce_loss
emission_margin_loss
transition_loss
weighted_emission_loss
weighted_transition_loss
emission_acc
transition_acc
viterbi_candidate_acc
emission_supervised_total
emission_supervised_used
emission_supervised_skipped
transition_supervised_total
transition_supervised_used
transition_supervised_skipped
```

This makes it possible to identify whether training issues are caused by emission scoring, transition scoring, margin loss, skipped labels, or decoder behavior.

---

## 14. Evaluation metrics

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

## 15. Visualization

One-Direction includes two visualization modes.

### 15.1 Static error visualization

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

### 15.2 Interactive OSM overlay

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

## 16. Unified pipeline runner

The project includes a unified runner:

```text
scripts/run_project.py
```

The runner uses explicit stage-level command-line defaults. Training settings such as epoch count are controlled through `scripts/run_project.py` defaults or through a direct training command, not through `configs/local.yaml` unless script-level config support is later added.

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

## 17. Manual GNN-HMM pipeline commands

Run commands from the repository root:

```powershell
cd E:\PANDA\One-Direction
```

### 17.1 Prepare trajectories

```powershell
python scripts\01_prepare_trajectories.py
```

### 17.2 Prepare GT routes

```powershell
python scripts\02_prepare_gt_routes.py
```

### 17.3 Build directed OSM graph

```powershell
python scripts\03_build_osm_graph.py
```

### 17.4 Build line graph

```powershell
python scripts\04_build_line_graph.py
```

### 17.5 Generate candidates

```powershell
python scripts\05_generate_candidates.py
```

### 17.6 Build improved training tensors

```powershell
python scripts\06_build_training_tensors.py --transition-mask-mode all
```

### 17.7 Run data-debug checks

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

### 17.8 Train GNN-HMM

```powershell
python -u scripts\07_train_gnn_hmm.py `
  --output outputs\checkpoints `
  --epochs 100 `
  --batch-size 2 `
  --lr 0.001 `
  --emission-weight 1.0 `
  --transition-weight 2.0 `
  --label-smoothing 0.02 `
  --margin-weight 0.1 `
  --margin 1.0 `
  --device auto `
  --log-every-batches 25
```

Output:

```text
outputs/checkpoints/gnn_hmm_best.pt
outputs/checkpoints/gnn_hmm_last.pt
data/reports/training_report.json
```

### 17.9 Decode

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

### 17.10 Evaluate

```powershell
python scripts\09_evaluate.py
```

Output:

```text
outputs/metrics/gnn_hmm_metrics.json
outputs/metrics/trajectory_metrics.csv
outputs/metrics/error_cases.csv
```

### 17.11 Visualize static errors

```powershell
python scripts\10_visualize_errors.py
```

### 17.12 Generate OSM overlay

```powershell
python scripts\11_visualize_osm_overlay.py
```

---

## 18. Local GPU workflow

### 18.1 Activate environment

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

### 18.2 Set geospatial environment variables

```powershell
$env:PYTHONPATH="E:\PANDA\One-Direction"
$env:GDAL_DATA="$env:CONDA_PREFIX\Library\share\gdal"
$env:PROJ_LIB="$env:CONDA_PREFIX\Library\share\proj"
$env:PYTHONUNBUFFERED="1"
```

### 18.3 Verify GPU and dependencies

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -c "import geopandas, shapely, pyproj, osmnx; print('geo stack ok')"
python -c "import src; print(src.__project__, src.__version__)"
```

### 18.4 Start end-to-end run

```powershell
python scripts\run_project.py gpu_e2e
```

### 18.5 GPU safety note

Do not run GNN-HMM training and RL/D-SAC GPU training at the same time on the same GPU.

The GNN-HMM training job repeatedly encodes the full line graph and should be given exclusive GPU access. Running an RL job in parallel can cause memory fragmentation, CUDA instability, or illegal memory access errors.

Recommended order:

```text
1. Train GNN-HMM.
2. Decode and evaluate GNN-HMM.
3. Run PPO or D-SAC separately.
4. Use CPU configs for RL experiments if the GPU is occupied.
```

---

## 19. Debugging checklist

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
12. Training loss is decomposed into emission and transition components.
13. Supervised labels are not unexpectedly skipped.
14. GNN-HMM and RL jobs are not running concurrently on the same GPU.
```

Important debug scripts:

```powershell
python scripts\12_debug_gnn_hmm_data.py
python scripts\11_visualize_osm_overlay.py
```

Additional direct checks:

```powershell
python -m py_compile scripts\07_train_gnn_hmm.py
python -m py_compile scripts\run_project.py
```

For CUDA debugging:

```powershell
$env:CUDA_LAUNCH_BLOCKING="1"
```

Use this only for short diagnostic runs because it slows training.

---

## 20. Experimental RL branch: PPO + Asymmetric Privileged Learning

The PPO branch is located at:

```text
RL/
```

This branch treats map matching as a sequential decision problem:

```text
episode = one trajectory
step    = one GPS observation
action  = choose one candidate road segment
```

The PPO actor uses deployment-safe features:

```text
candidate features
candidate mask
previous selected candidate
timestep fraction
```

The privileged critic additionally uses training-only information:

```text
GT candidate position
GT route/candidate hints
previous and next GT-transition hints
projection-to-GT hints
candidate validity hints
```

At inference time, only the actor is used.

### 20.1 PPO setup

Install dependencies:

```powershell
python -m pip install -r RL\requirements-rl.txt
```

Check inputs:

```powershell
python RL\scripts\00_check_inputs.py --config RL\configs\ppo_default.yaml
```

Run debug workflow:

```powershell
python RL\scripts\run_rl.py all --config RL\configs\ppo_debug.yaml
```

Run default workflow:

```powershell
python RL\scripts\run_rl.py all --config RL\configs\ppo_default.yaml
```

Run CPU workflow:

```powershell
python RL\scripts\run_rl.py all --config RL\configs\ppo_cpu.yaml
```

Expected outputs:

```text
RL/outputs/checkpoints/bc_actor.pt
RL/outputs/checkpoints/ppo_asym_best.pt
RL/outputs/checkpoints/ppo_asym_last.pt
RL/outputs/matches/ppo_asym_matches.parquet
RL/outputs/metrics/ppo_asym_metrics.json
RL/outputs/reports/ppo_training_report.json
```

PPO is useful as a reinforcement-learning baseline, but it should not replace the GNN-HMM unless it improves both point-level matching and route-level quality.

---

## 21. Experimental RL branch: Discrete SAC + Asymmetric Privileged Learning

The Discrete SAC branch is located at:

```text
D-SAC/
```

This branch uses:

```text
Behavior cloning warm-start
Masked categorical actor
Double Q critics
Target critics
Replay buffer
Automatic entropy temperature alpha
Discrete SAC objective
Asymmetric privileged critic input
Greedy/stochastic policy evaluation
```

The actor receives only deployment-safe observations. The two critics receive privileged training-only information in addition to actor observations.

### 21.1 D-SAC setup

Install dependencies:

```powershell
python -m pip install -r D-SAC\requirements-dsac.txt
```

Check inputs:

```powershell
python D-SAC\scripts\00_check_inputs.py --config D-SAC\configs\dsac_default.yaml
```

Run debug workflow:

```powershell
python D-SAC\scripts\run_dsac.py all --config D-SAC\configs\dsac_debug.yaml
```

Run default workflow:

```powershell
python D-SAC\scripts\run_dsac.py all --config D-SAC\configs\dsac_default.yaml
```

Run CPU workflow:

```powershell
python D-SAC\scripts\run_dsac.py all --config D-SAC\configs\dsac_cpu.yaml
```

Expected outputs:

```text
D-SAC/outputs/checkpoints/bc_actor.pt
D-SAC/outputs/checkpoints/dsac_asym_best.pt
D-SAC/outputs/checkpoints/dsac_asym_last.pt
D-SAC/outputs/matches/dsac_asym_matches.parquet
D-SAC/outputs/metrics/dsac_asym_metrics.json
D-SAC/outputs/reports/dsac_training_report.json
```

D-SAC is the advanced RL extension. It should be compared against PPO and GNN-HMM after its reward scale, replay behavior, entropy coefficient, and critic stability are validated.

---

## 22. Comparing GNN-HMM, PPO, and D-SAC

The recommended comparison hierarchy is:

```text
1. Candidate Top-1 baseline
2. GNN-HMM emission-only behavior
3. GNN-HMM + Viterbi decoding
4. PPO + Asymmetric Privileged Learning
5. Discrete SAC + Asymmetric Privileged Learning
```

The main comparison dimensions are:

```text
point_edge_accuracy
projection error
within-distance success rates
legal_transition_rate
path_edit_distance
trajectory_success_rate
mean confidence
error taxonomy
```

The GNN-HMM should remain the primary model unless an RL branch improves both:

```text
1. local candidate/edge selection
2. route-level sequence consistency
```

---

## 23. Recommended development order

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
15. Run PPO as an experimental RL baseline.
16. Run D-SAC as an advanced experimental RL baseline.
17. Compare all methods under the same evaluation script.
```

---

## 24. Future extensions

Possible future extensions include:

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
Hybrid GNN-HMM + RL reranking
```

A strong future hybrid design would use GNN-HMM scores as actor features for PPO or D-SAC:

```text
GNN-HMM emission score
GNN-HMM transition score
Viterbi baseline candidate
candidate confidence
```

The RL policy would then operate as a route-level reranker rather than learning map matching entirely from scratch.

---

## 25. Final summary

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
Experimental PPO and D-SAC reinforcement-learning branches
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

The current recommended research direction is to keep **GNN-HMM + Viterbi** as the primary system, then use PPO and D-SAC as controlled experimental baselines or future hybrid reranking modules.
