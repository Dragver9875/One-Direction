from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from onedir_dsac.data import EpisodeSample
from onedir_dsac.features import action_mask, actor_observation, observation_dims, privileged_observation
from onedir_dsac.models import DiscreteSACModel


def test_model_shapes():
    sample = EpisodeSample(
        trajectory_id=1,
        candidate_edge_idx=torch.tensor([[0, 1, -1], [1, 2, 3]]),
        candidate_mask=torch.tensor([[True, True, False], [True, True, True]]),
        emission_features=torch.randn(2, 3, 4),
        gt_candidate_pos=torch.tensor([0, 1]),
        transition_mask=torch.ones(1, 3, 3, dtype=torch.bool),
        candidate_proj_xy=torch.randn(2, 3, 2),
        gt_proj_xy=torch.randn(2, 2),
        gt_edge_idx=torch.tensor([0, 2]),
        timestamps=None,
    )
    actor_dim, critic_dim, action_dim = observation_dims(sample)
    model = DiscreteSACModel(actor_dim, critic_dim, action_dim)
    actor_obs = actor_observation(sample, 0, None)
    critic_obs = privileged_observation(sample, 0, None)
    mask = action_mask(sample, 0)
    probs, log_probs = model.policy(actor_obs, mask)
    q1, q2 = model.q_values(actor_obs, critic_obs)
    assert probs.shape[-1] == action_dim
    assert log_probs.shape[-1] == action_dim
    assert q1.shape[-1] == action_dim
    assert q2.shape[-1] == action_dim
