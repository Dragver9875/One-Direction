import torch

from src.training.losses import (
    GNNHMMLossConfig,
    compute_emission_loss,
    compute_total_loss,
    compute_transition_loss,
)
from src.training.negative_sampling import sample_candidate_negatives


def test_compute_emission_loss():
    scores = torch.tensor([[[5.0, 0.0], [0.0, 5.0]]])
    target = torch.tensor([[0, 1]])
    loss = compute_emission_loss(scores, target)
    assert loss.item() < 0.01


def test_compute_transition_loss():
    scores = torch.tensor([[[[0.0, 5.0], [0.0, 0.0]]]])
    target = torch.tensor([[0, 1]])
    loss = compute_transition_loss(scores, target)
    assert loss.item() < 0.01


def test_compute_total_loss():
    outputs = {
        "emission_scores": torch.tensor([[[5.0, 0.0], [0.0, 5.0]]]),
        "transition_scores": torch.tensor([[[[0.0, 5.0], [0.0, 0.0]]]]),
    }
    batch = {
        "gt_candidate_pos": torch.tensor([[0, 1]]),
    }
    loss = compute_total_loss(outputs, batch, cfg=GNNHMMLossConfig())
    assert "loss" in loss
    assert loss["loss"].item() < 0.02


def test_sample_candidate_negatives():
    mask = torch.tensor([[[True, True, True]]])
    target = torch.tensor([[1]])
    negatives = sample_candidate_negatives(mask, target)
    assert negatives.shape[-1] == 5
    assert 1 not in negatives[0, 0].tolist()
