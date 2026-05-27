# One-Direction

**One-Direction** is a GNN-enhanced HMM map-matching system that maps vehicle-acquired pose observations — position and yaw — onto the corresponding road segments of an OpenStreetMap road network.

The system is designed for the following problem:

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

Unlike a nearest-road matcher, One-Direction treats map matching as a **graph-structured sequence-decoding problem**. The OSM road network is converted into a directed edge-centric graph, road segments are encoded using a Graph Neural Network, and the final trajectory is decoded using a custom HMM/Viterbi decoder with learned emission and transition scores.

---

## 1. Core idea

For each vehicle observation:

$$ z_t = (x_t, y_t, \psi_t) $$

where:

* $(x_t, y_t)$ are the vehicle coordinates in a projected metric coordinate system.
* $\psi_t$ is the vehicle yaw or heading.
* $t$ is the trajectory timestep.

The goal is to predict:

$$ \hat{y}_t = (e_t, p_t, c_t) $$

where:

* $e_t$ is the matched directed OSM road segment.
* $p_t$ is the projected point on the matched road segment.
* $c_t$ is the confidence score.

The final matched path is obtained by solving:

$$ \hat{e}*{1:T} = \arg\max*{e_{1:T}} \left[ \sum_{t=1}^{T} E_\theta(z_t, e_t) + \sum_{t=2}^{T} T_\phi(e_{t-1}, e_t, z_{t-1}, z_t) \right] $$

where:

* $E_\theta$ is the learned emission score.
* $T_\phi$ is the learned transition score.
* $\hat{e}_{1:T}$ is the globally decoded road-segment sequence.

---

## 2. Why GNN-HMM?

Map matching is not a simple nearest-neighbour problem. A road segment may be spatially close to a GPS point but still be wrong due to:

* parallel roads,
* one-way constraints,
* flyovers and underpasses,
* highway/service-road ambiguity,
* ramps,
* roundabouts,
* sparse GPS sampling,
* yaw noise,
* GPS drift,
* dense urban road networks,
* incorrect or incomplete OSM metadata.

One-Direction combines three important ideas:

```text
1. Graph Neural Network
   Learns contextual road-segment embeddings from the OSM topology.

2. Learned emission scoring
   Scores how well each GPS/yaw observation matches each candidate road segment.

3. Learned transition scoring + HMM decoding
   Scores how plausible it is to move from one road segment to another and uses Viterbi decoding to recover the best global path.
```

This makes the system both **learned** and **topology-aware**.

---

## 3. System overview

```text
GPS/yaw trajectory
    ↓
Trajectory preprocessing
    ↓
Candidate road-segment generation
    ↓
Candidate feature computation

OSM PBF
    ↓
Directed road graph construction
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

Candidate transition features + road embeddings
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
│   └── eval.yaml
│
├── data/
│   ├── raw/
│   │   ├── osm/
│   │   │   └── region.osm.pbf
│   │   └── trajectories/
│   │       ├── gps_points.csv
│   │       └── gt_matches.csv
│   │
│   ├── interim/
│   │   ├── osm_graph_raw.pkl
│   │   ├── trajectory_projected.parquet
│   │   └── trajectory_clean.parquet
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
│       ├── candidate_report.json
│       ├── training_report.json
│       └── evaluation_report.json
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_gps.py
│   │   ├── load_gt.py
│   │   ├── preprocess_trajectory.py
│   │   ├── trajectory_splits.py
│   │   ├── build_training_tensors.py
│   │   └── dataset.py
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── osm_graph_builder.py
│   │   ├── road_graph_cleaner.py
│   │   ├── line_graph_builder.py
│   │   ├── graph_features.py
│   │   ├── transition_builder.py
│   │   └── candidate_search.py
│   │
│   ├── geometry/
│   │   ├── __init__.py
│   │   ├── projection.py
│   │   ├── distances.py
│   │   ├── bearing.py
│   │   ├── polyline_ops.py
│   │   └── yaw.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── road_gnn_encoder.py
│   │   ├── emission_head.py
│   │   ├── transition_head.py
│   │   ├── gnn_hmm.py
│   │   └── confidence_head.py
│   │
│   ├── decoder/
│   │   ├── __init__.py
│   │   ├── viterbi.py
│   │   ├── hmm_decoder.py
│   │   ├── path_constraints.py
│   │   └── decode_outputs.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py
│   │   ├── negative_sampling.py
│   │   ├── train_gnn_hmm.py
│   │   ├── validation.py
│   │   ├── checkpointing.py
│   │   └── scheduler.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── sequence_metrics.py
│   │   ├── projection_metrics.py
│   │   ├── compare_with_baseline.py
│   │   ├── error_analysis.py
│   │   └── visualize_matches.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── io.py
│       ├── logging.py
│       ├── seed.py
│       └── timing.py
│
├── scripts/
│   ├── 01_prepare_trajectories.py
│   ├── 02_build_osm_graph.py
│   ├── 03_build_line_graph.py
│   ├── 04_generate_candidates.py
│   ├── 05_build_training_tensors.py
│   ├── 06_train_gnn_hmm.py
│   ├── 07_decode_gnn_hmm.py
│   ├── 08_evaluate.py
│   └── 09_visualize_errors.py
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
│   ├── emissions/
│   │   └── emission_scores.parquet
│   │
│   ├── transitions/
│   │   └── transition_scores.parquet
│   │
│   ├── matches/
│   │   ├── gnn_hmm_matches.parquet
│   │   └── gnn_hmm_matches.geojson
│   │
│   ├── metrics/
│   │   ├── gnn_hmm_metrics.json
│   │   └── comparison_report.json
│   │
│   └── figures/
│       ├── candidate_debug.png
│       ├── matched_paths.png
│       └── error_cases.png
│
└── tests/
    ├── test_geometry.py
    ├── test_candidate_search.py
    ├── test_line_graph.py
    ├── test_viterbi.py
    ├── test_emission_head.py
    ├── test_transition_head.py
    └── test_gnn_hmm.py
```

---

## 5. Data format

### 5.1 GPS input

Expected file:

```text
data/raw/trajectories/gps_points.csv
```

Expected schema:

```text
trajectory_id,timestamp,lat,lon,yaw
```

Optional fields:

```text
speed,accuracy,source,vehicle_id
```

If speed is unavailable, it is computed from consecutive points after coordinate projection.

---

### 5.2 Ground-truth match file

Expected file:

```text
data/raw/trajectories/gt_matches.csv
```

Expected schema:

```text
trajectory_id,timestamp,gt_edge_id,gt_proj_x,gt_proj_y
```

The ground-truth edge ID should correspond to the OSM-derived directed road segment ID after preprocessing. If the original GT uses a different ID system, an ID alignment/conflation step is required.

---

### 5.3 OSM input

Expected file:

```text
data/raw/osm/region.osm.pbf
```

The OSM file is parsed into a directed road graph. Each legal travel direction is represented separately.

For example:

```text
Two-way road:
    edge A → B
    edge B → A

One-way road:
    only legal direction is retained
```

---

## 6. Directed road graph

The directed road graph represents the physical road network.

```text
Road graph:
    node = OSM intersection / endpoint / geometry node
    edge = directed road segment
```

Each edge stores:

```text
edge_id
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
```

The directed representation is important because yaw-sensitive matching requires direction-aware road segments. A vehicle facing north should not be matched to the southbound version of a directed edge unless the transition model determines that such a match is plausible under the available constraints.

---

## 7. Edge-centric line graph

One-Direction converts the road graph into a line graph.

```text
Original road graph:
    node = intersection
    edge = road segment

Line graph:
    node = road segment
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

$$ s_t = e_t $$

Therefore, every line-graph node corresponds directly to one possible hidden state.

---

## 8. Road-segment features

Each road-segment node receives a feature vector:

```text
length_m
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
359° and 1° are close directions,
but as raw numbers they look far apart.
```

Using sine and cosine avoids angle wraparound errors.

---

## 9. Candidate generation

For every vehicle observation:

$$ z_t = (x_t, y_t, \psi_t) $$

One-Direction finds a set of candidate road segments:

$$ C_t = {e_{t,1}, e_{t,2}, \ldots, e_{t,k}} $$

Recommended first configuration:

```yaml
candidate_generation:
  radius_m: 50.0
  max_candidates: 10
```

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

The most important candidate-generation metric is **top-k candidate recall**:

```text
Does the true road segment appear in the candidate set?
```

If the true edge is not included in the candidate set, the model cannot recover it later.

Recommended target:

```text
Top-5 candidate recall  > 95%
Top-10 candidate recall > 98%
```

---

## 10. Model architecture

One-Direction contains four main model components:

```text
1. Road GNN encoder
2. Emission head
3. Transition head
4. HMM/Viterbi decoder
```

---

### 10.1 Road GNN encoder

The road GNN operates on the edge-centric line graph.

Input:

```text
line_graph.pt
segment_features.pt
```

The encoder computes a road-segment embedding for every directed segment:

$$ h_e = \mathrm{GNN}(x_e, G_{\text{line}}) $$

where:

* $x_e$ is the feature vector of road segment (e).
* $G_{\text{line}}$ is the line graph.
* $h_e$ is the learned segment embedding.

Recommended first encoder:

```yaml
gnn:
  type: graphsage
  hidden_dim: 128
  output_dim: 128
  num_layers: 2
  dropout: 0.1
```

GraphSAGE is a suitable starting point because it is scalable, stable, and straightforward to train. A Graph Attention Network can be added later if attention over neighbouring road segments becomes useful for dense intersections and ambiguous roads.

---

### 10.2 Emission head

The emission head answers:

```text
How well does GPS/yaw observation z_t match candidate road segment $e$?
```

For each candidate pair $(z_t, e)$, build:

$$ r_{t,e} = \begin{bmatrix} h_e \ d(z_t,e) \ \Delta\psi(z_t,e) \ \mathrm{offset}(z_t,e) \ \mathrm{speed}_t \ \sin(\psi_t) \ \cos(\psi_t) \ \sin(\theta_e) \ \cos(\theta_e) \end{bmatrix} $$

where:

* $h_e$ is the road-segment embedding.
* $d(z_t,e)$ is the perpendicular distance from the vehicle point to the segment.
* $\Delta\psi(z_t,e)$ is the yaw difference.
* $\mathrm{offset}(z_t,e)$ is the projected offset along the segment.
* $\theta_e$ is the road-segment bearing.

The emission score is:

$$ E_\theta(z_t,e) = \mathrm{MLP}*{\text{emission}}(r*{t,e}) $$

---

### 10.3 Transition head

The transition head answers:

```text
How plausible is it to move from road segment $e_i$ at timestep t-1 to road segment $e_j$ at timestep t?
```

For each candidate transition $(e_i, e_j)$, build:

$$ q_{i,j,t} = \begin{bmatrix} h_i \ h_j \ d_{\text{gps}} \ d_{\text{route}} \ \lvert d_{\text{route}} - d_{\text{gps}} \rvert \ \mathrm{turn_angle} \ \mathrm{yaw_change} \ \mathrm{speed_consistency} \ \mathrm{is_connected} \ \mathrm{is_legal} \end{bmatrix} $$

The transition score is:

$$ T_\phi(e_i,e_j,z_{t-1},z_t) = \mathrm{MLP}*{\text{transition}}(q*{i,j,t}) $$

Hard constraints are still enforced even though the transition score is learned.

Examples:

```text
illegal one-way transition:
    score = -inf

disconnected transition beyond allowed search range:
    score = large negative penalty

route distance too large:
    score = large negative penalty
```

The neural transition scorer should improve ranking among plausible transitions, not override physical and topological constraints.

---

### 10.4 HMM/Viterbi decoder

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

$$ \hat{e}*{1:T} = \arg\max*{e_t \in C_t} \left[ \sum_{t=1}^{T} E_t(e_t) + \sum_{t=2}^{T} T_t(e_{t-1}, e_t) \right] $$

Viterbi recurrence:

$$ V_1(e) = E_1(e) $$

$$ V_t(e) = E_t(e) + \max_{e' \in C_{t-1}} \left[ V_{t-1}(e') + T_t(e',e) \right] $$

Backpointer:

$$ B_t(e) = \arg\max_{e' \in C_{t-1}} \left[ V_{t-1}(e') + T_t(e',e) \right] $$

The final edge is:

$$ \hat{e}_T = \arg\max_e V_T(e) $$

Then the full matched sequence is recovered by backtracking through the backpointers.

---

## 11. Training objective

One-Direction trains the emission and transition scores jointly.

Given a ground-truth road-segment sequence:

$$ e_1^{\text{GT}}, e_2^{\text{GT}}, \ldots, e_T^{\text{GT}} $$

The training objective contains two main terms:

```text
1. Emission loss
2. Transition loss
```

---

### 11.1 Emission loss

For each GPS observation, the model should rank the GT road segment above the other candidate segments:

$$ \mathcal{L}*{\text{emission}} = -\sum_t \log \frac{ \exp(E_t(e_t^{\text{GT}})) }{ \sum*{e \in C_t} \exp(E_t(e)) } $$

---

### 11.2 Transition loss

For each consecutive GT transition, the model should rank the correct next segment above alternative next candidates:

$$ \mathcal{L}*{\text{transition}} = -\sum*{t=2}^{T} \log \frac{ \exp(T_t(e_{t-1}^{\text{GT}},e_t^{\text{GT}})) }{ \sum_{e \in C_t} \exp(T_t(e_{t-1}^{\text{GT}},e)) } $$

---

### 11.3 Total loss

$$ \mathcal{L} = \mathcal{L}*{\text{emission}} + \lambda_T \mathcal{L}*{\text{transition}} + \lambda_R \lVert \Theta \rVert_2^2 $$

Recommended first setting:

```yaml
loss:
  emission_weight: 1.0
  transition_weight: 1.0
  weight_decay: 0.0001
```

The first implementation does not need to backpropagate through Viterbi. The model can train local emission and transition scores, then use Viterbi for global decoding at inference time.

---

## 12. Running the pipeline

### 12.1 Prepare trajectories

```bash
python scripts/01_prepare_trajectories.py \
  --input data/raw/trajectories/gps_points.csv \
  --output data/interim/trajectory_clean.parquet
```

This script:

```text
loads GPS data
sorts by trajectory_id and timestamp
projects lat/lon to x/y
normalizes yaw
computes speed
removes invalid points
```

---

### 12.2 Build OSM road graph

```bash
python scripts/02_build_osm_graph.py \
  --osm data/raw/osm/region.osm.pbf \
  --output data/processed/road_graph/
```

This script:

```text
parses OSM
builds a directed road graph
extracts edge geometries
computes length and bearing
saves graph and edge table
```

---

### 12.3 Build line graph

```bash
python scripts/03_build_line_graph.py \
  --road-graph data/processed/road_graph/road_graph.pkl \
  --output data/processed/line_graph/
```

This script:

```text
converts road edges into line-graph nodes
connects legal road-segment transitions
computes segment features
saves graph tensors
```

---

### 12.4 Generate candidates

```bash
python scripts/04_generate_candidates.py \
  --trajectories data/interim/trajectory_clean.parquet \
  --edges data/processed/road_graph/edge_table.parquet \
  --gt data/raw/trajectories/gt_matches.csv \
  --output data/processed/candidates/
```

This script:

```text
finds nearest road segments
computes projection points
computes distance to segment
computes yaw difference
marks GT candidates
reports top-k candidate recall
```

---

### 12.5 Build training tensors

```bash
python scripts/05_build_training_tensors.py \
  --candidates data/processed/candidates/ \
  --line-graph data/processed/line_graph/ \
  --output data/processed/tensors/
```

This script:

```text
groups candidates by trajectory
builds candidate index tensors
builds emission feature tensors
builds transition feature tensors
builds GT labels
```

---

### 12.6 Train One-Direction

```bash
python scripts/06_train_gnn_hmm.py \
  --config configs/model.yaml
```

This script:

```text
loads the line graph
loads trajectory candidate tensors
runs the road GNN encoder
computes emission scores
computes transition scores
computes emission loss
computes transition loss
saves checkpoints
```

---

### 12.7 Decode trajectories

```bash
python scripts/07_decode_gnn_hmm.py \
  --checkpoint outputs/checkpoints/gnn_hmm_best.pt \
  --split test \
  --output outputs/matches/
```

This script:

```text
loads the trained model
computes emission scores
computes transition scores
runs Viterbi decoding
recovers matched segment sequence
projects points onto matched segments
saves matches
```

---

### 12.8 Evaluate

```bash
python scripts/08_evaluate.py \
  --pred outputs/matches/gnn_hmm_matches.parquet \
  --gt data/raw/trajectories/gt_matches.csv \
  --output outputs/metrics/
```

This script computes:

```text
point_edge_accuracy
top_k_candidate_recall
mean_projection_error
median_projection_error
path_edit_distance
wrong_road_rate
yaw_error
trajectory_success_rate
```

---

### 12.9 Visualize errors

```bash
python scripts/09_visualize_errors.py \
  --pred outputs/matches/gnn_hmm_matches.parquet \
  --edges data/processed/road_graph/edges.geojson \
  --output outputs/figures/
```

This script generates:

```text
matched path overlays
wrong-road examples
parallel road failures
candidate ambiguity cases
decoder failure cases
```

---

## 13. Configuration example

```yaml
project:
  name: One-Direction
  seed: 42

data:
  trajectory_file: data/interim/trajectory_clean.parquet
  gt_file: data/raw/trajectories/gt_matches.csv
  road_graph_dir: data/processed/road_graph
  line_graph_dir: data/processed/line_graph
  candidate_dir: data/processed/candidates
  tensor_dir: data/processed/tensors

candidate_generation:
  radius_m: 50.0
  max_candidates: 10
  require_gt_in_candidates: true

graph:
  directed: true
  use_line_graph: true
  max_transition_distance_m: 300.0

model:
  name: gnn_hmm

  gnn:
    type: graphsage
    input_dim: null
    hidden_dim: 128
    output_dim: 128
    num_layers: 2
    dropout: 0.1

  emission_head:
    hidden_dims: [128, 64]
    dropout: 0.1

  transition_head:
    hidden_dims: [128, 64]
    dropout: 0.1

decoder:
  type: viterbi
  illegal_transition_score: -1000000000.0
  disconnected_transition_penalty: -1000.0
  max_route_distance_m: 300.0
  confidence_method: margin

training:
  batch_size: 8
  epochs: 50
  lr: 0.001
  weight_decay: 0.0001
  grad_clip_norm: 5.0

loss:
  emission_weight: 1.0
  transition_weight: 1.0

evaluation:
  metrics:
    - point_edge_accuracy
    - mean_projection_error
    - median_projection_error
    - path_edit_distance
    - wrong_road_rate
    - trajectory_success_rate
```

---

## 14. Evaluation metrics

### 14.1 Point edge accuracy

Measures the fraction of timesteps where the predicted edge ID matches the GT edge ID.

```text
point_edge_accuracy = correct_edge_predictions / total_points
```

---

### 14.2 Candidate recall

Measures whether the GT edge appears in the generated candidate set.

```text
Top-k candidate recall = points where GT edge appears in top-k candidates / total points
```

This is a preprocessing metric, but it is essential. Low candidate recall caps final model accuracy.

---

### 14.3 Projection error

Measures the distance between the predicted projected point and the GT projected point.

```text
projection_error = distance(pred_proj, gt_proj)
```

Report both mean and median values.

---

### 14.4 Path edit distance

Measures the sequence-level difference between predicted and GT edge sequences.

This catches errors that point-level accuracy may hide, such as:

```text
wrong turn at an intersection
incorrect ramp choice
jump to parallel road
broken route continuity
```

---

### 14.5 Wrong-road rate

Measures cases where the selected road is close to the GPS point but semantically/topologically wrong.

Common examples:

```text
highway vs service road
flyover vs ground road
main road vs side lane
opposite direction of divided road
```

---

### 14.6 Trajectory success rate

Measures the fraction of trajectories that are matched successfully under a chosen correctness threshold.

Example:

```text
trajectory_success_rate = trajectories with edge accuracy above threshold / total trajectories
```

---

## 15. Model outputs

The final prediction file is:

```text
outputs/matches/gnn_hmm_matches.parquet
```

Expected schema:

```text
trajectory_id
t
timestamp
pred_edge_id
pred_edge_idx
pred_proj_x
pred_proj_y
confidence
emission_score
transition_score
total_path_score
```

GeoJSON output:

```text
outputs/matches/gnn_hmm_matches.geojson
```

This can be opened in GIS tools or visualized in notebooks.

---

## 16. Comparison with external baselines

This repository focuses on the GNN-enhanced HMM system. If another team member or external module provides baseline outputs, they should be exported using the common format:

```text
model_outputs/{model_name}_matches.parquet
```

Required columns:

```text
trajectory_id
t
timestamp
pred_edge_id
pred_proj_x
pred_proj_y
confidence
```

The comparison script can then evaluate all models against the same GT file.

---

## 17. Recommended development order

```text
1. Implement coordinate projection and yaw normalization.
2. Build directed OSM road graph.
3. Build edge-centric line graph.
4. Generate candidate road segments.
5. Verify top-k candidate recall.
6. Build training tensors.
7. Implement RoadGNNEncoder.
8. Implement EmissionHead.
9. Implement TransitionHead.
10. Implement Viterbi decoder.
11. Train emission and transition scorers jointly.
12. Decode trajectories.
13. Evaluate edge accuracy and projection error.
14. Visualize failure cases.
15. Improve candidate generation and transition features.
```

The first hard checkpoint is candidate recall. If candidate recall is poor, improve spatial indexing, search radius, road filtering, geometry projection, and GT-to-OSM ID alignment before training the model.

---

## 18. Important implementation notes

### 18.1 Coordinate system

Never compute metric distances directly in latitude/longitude degrees.

Use:

```text
WGS84 latitude/longitude
    ↓
local projected CRS / UTM
    ↓
x, y in meters
```

---

### 18.2 Directionality

Always preserve road direction.

Yaw-sensitive matching requires directed segments. A bidirectional OSM road should become two directed edges.

---

### 18.3 Candidate recall before model accuracy

Candidate generation is not a minor preprocessing step. It controls the maximum possible accuracy of the model.

If the GT road segment is absent from the candidate list, the decoder cannot select it.

---

### 18.4 Hard constraints should remain hard

The transition model is learned, but it should not override impossible graph constraints.

Examples of hard constraints:

```text
illegal one-way movement
physically impossible jump
route distance far beyond feasible motion
invalid disconnected transition
```

---

### 18.5 Do not rely only on point-level accuracy

Map matching is a sequence problem. A model can have acceptable point-level accuracy but still produce bad route continuity.

Always report sequence-level metrics such as:

```text
path edit distance
wrong-road rate
trajectory success rate
```

---

## 19. Expected failure cases

One-Direction should explicitly analyze the following failure cases:

```text
parallel roads
highway and service-road ambiguity
flyovers and underpasses
roundabouts
GPS drift near intersections
low sampling rate trajectories
wrong or missing OSM one-way tags
incorrect OSM road class
candidate set missing the true road
sudden yaw noise
trajectory gaps
```

These cases are important because they are where naive nearest-road matching usually fails.

---

## 20. Future extensions

Possible extensions include:

```text
Graph Attention Network road encoder
Temporal trajectory encoder
CRF-style structured training
Differentiable Viterbi / soft dynamic programming
Uncertainty-aware emission scores
Map-error detection head
Online streaming map matching
Synthetic trajectory generation from OSM/SUMO
Pseudo-labelling from large raw GPS datasets
Road closure and access reliability prediction
```

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
```

The final output is:

```text
matched OSM road segment per timestep
projected point on road segment
confidence score
globally consistent matched road-segment sequence
```

The central design principle is:

```text
Use neural learning to improve road-segment and transition scoring,
but preserve graph topology and sequence consistency through HMM decoding.
```
