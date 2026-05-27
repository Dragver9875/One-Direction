from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ViterbiResult:
    path_positions: Tensor
    path_edge_idx: Tensor
    path_score: Tensor
    dp_scores: Tensor
    backpointers: Tensor


def _validate_shapes(
    emission_scores: Tensor,
    transition_scores: Tensor,
    candidate_edge_idx: Tensor,
    candidate_mask: Tensor | None,
    transition_mask: Tensor | None,
) -> None:
    if emission_scores.ndim != 2:
        raise ValueError("emission_scores must have shape [T, K].")

    if transition_scores.ndim != 3:
        raise ValueError("transition_scores must have shape [T - 1, K_prev, K_curr].")

    if candidate_edge_idx.shape != emission_scores.shape:
        raise ValueError("candidate_edge_idx must have shape [T, K].")

    t_count, k_count = emission_scores.shape

    if t_count == 0:
        raise ValueError("Cannot decode an empty trajectory.")

    if t_count == 1:
        if transition_scores.numel() != 0:
            raise ValueError("For T=1, transition_scores must be empty.")
    else:
        expected = (t_count - 1, k_count, k_count)
        if tuple(transition_scores.shape) != expected:
            raise ValueError(
                f"transition_scores must have shape {expected}, "
                f"got {tuple(transition_scores.shape)}."
            )

    if candidate_mask is not None and candidate_mask.shape != emission_scores.shape:
        raise ValueError("candidate_mask must have shape [T, K].")

    if transition_mask is not None and transition_mask.shape != transition_scores.shape:
        raise ValueError("transition_mask must have shape [T - 1, K, K].")


def viterbi_decode(
    emission_scores: Tensor,
    transition_scores: Tensor,
    candidate_edge_idx: Tensor,
    candidate_mask: Tensor | None = None,
    transition_mask: Tensor | None = None,
    invalid_score: float = -1.0e9,
) -> ViterbiResult:
    _validate_shapes(
        emission_scores=emission_scores,
        transition_scores=transition_scores,
        candidate_edge_idx=candidate_edge_idx,
        candidate_mask=candidate_mask,
        transition_mask=transition_mask,
    )

    device = emission_scores.device
    dtype = emission_scores.dtype

    emissions = emission_scores.clone()

    if candidate_mask is not None:
        emissions = emissions.masked_fill(~candidate_mask.bool(), invalid_score)

    emissions = emissions.masked_fill(candidate_edge_idx < 0, invalid_score)

    t_count, k_count = emissions.shape

    if t_count == 1:
        best_pos = torch.argmax(emissions[0])
        best_score = emissions[0, best_pos]
        path_positions = best_pos.reshape(1)
        path_edge_idx = candidate_edge_idx[0, best_pos].reshape(1)
        dp_scores = emissions.clone()
        backpointers = torch.full((1, k_count), -1, dtype=torch.long, device=device)
        return ViterbiResult(
            path_positions=path_positions,
            path_edge_idx=path_edge_idx,
            path_score=best_score,
            dp_scores=dp_scores,
            backpointers=backpointers,
        )

    transitions = transition_scores.clone()

    if transition_mask is not None:
        transitions = transitions.masked_fill(~transition_mask.bool(), invalid_score)

    dp_scores = torch.full(
        (t_count, k_count),
        invalid_score,
        dtype=dtype,
        device=device,
    )
    backpointers = torch.full(
        (t_count, k_count),
        -1,
        dtype=torch.long,
        device=device,
    )

    dp_scores[0] = emissions[0]

    for t in range(1, t_count):
        prev_scores = dp_scores[t - 1].unsqueeze(-1)
        candidate_scores = prev_scores + transitions[t - 1]
        best_prev_scores, best_prev_idx = torch.max(candidate_scores, dim=0)

        dp_scores[t] = emissions[t] + best_prev_scores
        backpointers[t] = best_prev_idx

    best_last_pos = torch.argmax(dp_scores[-1])
    best_path_score = dp_scores[-1, best_last_pos]

    path_positions = torch.full(
        (t_count,),
        -1,
        dtype=torch.long,
        device=device,
    )
    path_positions[-1] = best_last_pos

    for t in range(t_count - 1, 0, -1):
        curr_pos = path_positions[t]
        prev_pos = backpointers[t, curr_pos]
        path_positions[t - 1] = prev_pos

    path_edge_idx = candidate_edge_idx[
        torch.arange(t_count, device=device),
        path_positions,
    ]

    return ViterbiResult(
        path_positions=path_positions,
        path_edge_idx=path_edge_idx,
        path_score=best_path_score,
        dp_scores=dp_scores,
        backpointers=backpointers,
    )


def batch_viterbi_decode(
    emission_scores: Tensor,
    transition_scores: Tensor,
    candidate_edge_idx: Tensor,
    candidate_mask: Tensor | None = None,
    transition_mask: Tensor | None = None,
    invalid_score: float = -1.0e9,
) -> list[ViterbiResult]:
    if emission_scores.ndim != 3:
        raise ValueError("batched emission_scores must have shape [B, T, K].")
    if transition_scores.ndim != 4:
        raise ValueError("batched transition_scores must have shape [B, T - 1, K, K].")
    if candidate_edge_idx.shape != emission_scores.shape:
        raise ValueError("candidate_edge_idx must have shape [B, T, K].")

    batch_size = emission_scores.shape[0]
    results = []

    for b in range(batch_size):
        cm = candidate_mask[b] if candidate_mask is not None else None
        tm = transition_mask[b] if transition_mask is not None else None
        results.append(
            viterbi_decode(
                emission_scores=emission_scores[b],
                transition_scores=transition_scores[b],
                candidate_edge_idx=candidate_edge_idx[b],
                candidate_mask=cm,
                transition_mask=tm,
                invalid_score=invalid_score,
            )
        )

    return results
