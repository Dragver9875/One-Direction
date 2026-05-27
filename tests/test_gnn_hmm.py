import pytest
import torch

pytest.importorskip("torch_geometric")

from torch_geometric.data import Data

from src.models.gnn_hmm import GNNHMM, GNNHMMConfig
from src.models.road_gnn_encoder import RoadGNNEncoderConfig
from src.models.emission_head import EmissionHeadConfig
from src.models.transition_head import TransitionHeadConfig


def make_model():
    return GNNHMM(
        GNNHMMConfig(
            road_gnn=RoadGNNEncoderConfig(
                input_dim=4,
                hidden_dim=8,
                output_dim=8,
                num_layers=2,
                gnn_type="graphsage",
            ),
            emission_head=EmissionHeadConfig(
                road_embedding_dim=8,
                scalar_feature_dim=3,
                hidden_dims=(8,),
            ),
            transition_head=TransitionHeadConfig(
                road_embedding_dim=8,
                scalar_feature_dim=4,
                hidden_dims=(8,),
            ),
        )
    )


def test_gnn_hmm_forward_shapes():
    model = make_model()

    road_data = Data(
        x=torch.randn(5, 4),
        edge_index=torch.tensor(
            [
                [0, 1, 2, 3],
                [1, 2, 3, 4],
            ],
            dtype=torch.long,
        ),
    )

    candidate_edge_idx = torch.tensor([[[0, 1], [1, 2], [2, 3]]])
    emission_features = torch.randn(1, 3, 2, 3)
    prev_edge_idx = torch.tensor([[[[0, 0], [1, 1]], [[1, 1], [2, 2]]]])
    curr_edge_idx = torch.tensor([[[[1, 2], [1, 2]], [[2, 3], [2, 3]]]])
    transition_features = torch.randn(1, 2, 2, 2, 4)

    outputs = model(
        road_x_or_data=road_data,
        candidate_edge_idx=candidate_edge_idx,
        emission_features=emission_features,
        prev_edge_idx=prev_edge_idx,
        curr_edge_idx=curr_edge_idx,
        transition_features=transition_features,
    )

    assert outputs["road_embeddings"].shape == (5, 8)
    assert outputs["emission_scores"].shape == (1, 3, 2)
    assert outputs["transition_scores"].shape == (1, 2, 2, 2)
