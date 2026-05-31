from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from onedir_hmm.data import EpisodeSample
from onedir_hmm.scoring import HMMParams, compute_emission_scores, compute_transition_scores

def make_sample():
    return EpisodeSample(
        trajectory_id=1,
        candidate_edge_idx=torch.tensor([[0, 1], [1, 2]]),
        candidate_mask=torch.ones(2, 2, dtype=torch.bool),
        candidate_proj_xy=torch.zeros(2, 2, 2),
        emission_features=torch.zeros(2, 2, 16),
        transition_features=torch.zeros(1, 2, 2, 20),
        transition_mask=torch.ones(1, 2, 2, dtype=torch.bool),
        gt_candidate_pos=torch.tensor([0, 1]),
        gt_edge_idx=torch.tensor([0, 2]),
        gt_proj_xy=torch.zeros(2, 2),
        emission_feature_names=[],
        transition_feature_names=[],
    )

def test_scoring_shapes():
    sample = make_sample()
    params = HMMParams()
    assert compute_emission_scores(sample, params).shape == (2, 2)
    assert compute_transition_scores(sample, params).shape == (1, 2, 2)
