import torch

from src.models.transition_head import TransitionHead, TransitionHeadConfig


def test_transition_head_output_shape():
    head = TransitionHead(
        TransitionHeadConfig(
            road_embedding_dim=8,
            scalar_feature_dim=5,
            hidden_dims=(16,),
        )
    )

    road_embeddings = torch.randn(6, 8)
    prev_edge_idx = torch.tensor([[[[0, 0], [1, 1]]]])
    curr_edge_idx = torch.tensor([[[[2, 3], [2, -1]]]])
    transition_features = torch.randn(1, 1, 2, 2, 5)
    mask = curr_edge_idx >= 0

    scores = head(
        road_embeddings,
        prev_edge_idx,
        curr_edge_idx,
        transition_features,
        mask,
    )

    assert scores.shape == (1, 1, 2, 2)
    assert scores[0, 0, 1, 1].item() < -1.0e8


def test_transition_head_backward():
    head = TransitionHead(
        TransitionHeadConfig(
            road_embedding_dim=4,
            scalar_feature_dim=3,
            hidden_dims=(8,),
        )
    )

    road_embeddings = torch.randn(5, 4, requires_grad=True)
    prev_edge_idx = torch.tensor([[[[0, 0], [1, 1]]]])
    curr_edge_idx = torch.tensor([[[[1, 2], [2, 3]]]])
    transition_features = torch.randn(1, 1, 2, 2, 3)

    scores = head(road_embeddings, prev_edge_idx, curr_edge_idx, transition_features)
    loss = scores.sum()
    loss.backward()

    assert road_embeddings.grad is not None
