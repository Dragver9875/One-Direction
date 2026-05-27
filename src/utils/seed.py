from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(
    seed: int = 42,
    deterministic: bool = True,
    benchmark: bool = False,
) -> int:
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            set_torch_deterministic(benchmark=benchmark)
    except ImportError:
        pass

    return seed


def set_torch_deterministic(benchmark: bool = False) -> None:
    try:
        import torch

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = benchmark

        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    except ImportError:
        return


def worker_seed_init(worker_id: int) -> None:
    try:
        import torch

        worker_seed = torch.initial_seed() % 2**32
    except ImportError:
        worker_seed = worker_id

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_torch_generator(seed: int = 42):
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required to create a torch.Generator.") from exc

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
