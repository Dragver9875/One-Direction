from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from onedir_hmm.viterbi import viterbi_decode

def test_viterbi_prefers_consistent_path():
    emissions = torch.tensor([[2.0, 0.0], [0.0, 2.0], [0.0, 2.0]])
    transitions = torch.tensor([[[2.0, -3.0], [-3.0, 2.0]], [[2.0, -3.0], [-3.0, 2.0]]])
    path, _ = viterbi_decode(emissions, transitions)
    assert path == [1, 1, 1]
