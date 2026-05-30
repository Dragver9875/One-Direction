from __future__ import annotations
import sys
from pathlib import Path

def add_rl_src_to_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    rl_src = repo_root / 'RL' / 'src'
    if str(rl_src) not in sys.path:
        sys.path.insert(0, str(rl_src))
    return repo_root
