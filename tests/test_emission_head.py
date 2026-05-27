import torch

from src.models.emission_head import EmissionHead, EmissionHeadConfig


def test_emission_head_output_shape():
    head = EmissionHead(
        EmissionHeadConfig(
            road_embedding_dim=8,
            scalar_feature_dim=4,
            hidden_dims=(16,),
        )
    )

    road_embeddings = torch.randn(5, 8)
    candidate_edge_idx = torch.tensor([[[0, 1, 2], [2, 3, -1]]])
    emission_features = torch.randn(1, 2, 3, 4)
    mask = candidate_edge_idx >= 0

    scores = head(road_embeddings, candidate_edge_idx, emission_features, mask)
    assert scores.shape == (1, 2, 3)
    assert scores[0, 1, 2].item() < -1.0e8


def test_emission_head_backward():
    head = EmissionHead(
        EmissionHeadConfig(
            road_embedding_dim=4,
            scalar_feature_dim=2,
            hidden_dims=(8,),
        )
    )

    road_embeddings = torch.randn(3, 4, requires_grad=True)
    candidate_edge_idx = torch.tensor([[[0, 1]]])
    emission_features = torch.randn(1, 1, 2, 2)

    scores = head(road_embeddings, candidate_edge_idx, emission_features)
    loss = scores.sum()
    loss.backward()

    assert road_embeddings.grad is not None
