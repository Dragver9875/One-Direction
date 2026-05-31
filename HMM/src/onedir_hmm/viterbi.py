from __future__ import annotations
import torch
from torch import Tensor

def viterbi_decode(emissions: Tensor, transitions: Tensor) -> tuple[list[int], Tensor]:
    if emissions.ndim != 2:
        raise ValueError("emissions must have shape [T, K].")
    t_len, _ = emissions.shape
    if t_len == 0:
        return [], torch.empty(0)
    dp = emissions[0].clone()
    back = []
    history = [dp.clone()]
    for t in range(1, t_len):
        scores = dp[:, None] if transitions.numel() == 0 else dp[:, None] + transitions[t - 1]
        best_score, best_prev = scores.max(dim=0)
        dp = emissions[t] + best_score
        back.append(best_prev)
        history.append(dp.clone())
    last = int(dp.argmax().item())
    path = [last]
    for b in reversed(back):
        last = int(b[last].item())
        path.append(last)
    path.reverse()
    return path, torch.stack(history)

def confidence_from_scores(scores: Tensor, path: list[int], temperature: float = 1.0) -> list[float]:
    if not path:
        return []
    probs = torch.softmax(scores / max(float(temperature), 1e-6), dim=-1)
    return [float(probs[t, a].item()) for t, a in enumerate(path)]
