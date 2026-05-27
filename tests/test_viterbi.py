import torch

from src.decoder.viterbi import viterbi_decode
from src.decoder.hmm_decoder import HMMDecoder
from src.decoder.path_constraints import build_candidate_mask


def test_viterbi_decode_prefers_best_path():
    emission = torch.tensor(
        [
            [5.0, 0.0],
            [0.0, 5.0],
            [0.0, 5.0],
        ]
    )
    transition = torch.tensor(
        [
            [[0.0, 3.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 3.0]],
        ]
    )
    candidate_edge_idx = torch.tensor(
        [
            [10, 11],
            [20, 21],
            [30, 31],
        ]
    )

    result = viterbi_decode(emission, transition, candidate_edge_idx)
    assert result.path_positions.tolist() == [0, 1, 1]
    assert result.path_edge_idx.tolist() == [10, 21, 31]


def test_viterbi_respects_candidate_mask():
    emission = torch.tensor([[10.0, 20.0]])
    transition = torch.empty((0, 2, 2))
    candidate_edge_idx = torch.tensor([[1, 2]])
    candidate_mask = torch.tensor([[True, False]])

    result = viterbi_decode(emission, transition, candidate_edge_idx, candidate_mask)
    assert result.path_edge_idx.tolist() == [1]


def test_hmm_decoder_runs():
    emission = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    transition = torch.zeros((1, 2, 2))
    candidate_edge_idx = torch.tensor([[0, 1], [2, 3]])

    decoder = HMMDecoder()
    result = decoder.decode(emission, transition, candidate_edge_idx)
    assert len(result.path_edge_idx) == 2


def test_build_candidate_mask():
    candidate_edge_idx = torch.tensor([[0, 1, -1]])
    mask = build_candidate_mask(candidate_edge_idx)
    assert mask.tolist() == [[True, True, False]]
