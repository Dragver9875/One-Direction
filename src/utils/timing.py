from __future__ import annotations

from contextlib import ContextDecorator
from dataclasses import dataclass, field
from functools import wraps
import time
from typing import Any, Callable


def format_seconds(seconds: float) -> str:
    seconds = float(seconds)

    if seconds < 1.0:
        return f"{seconds * 1000.0:.2f} ms"

    minutes, sec = divmod(seconds, 60.0)
    hours, minutes = divmod(minutes, 60.0)

    if hours >= 1:
        return f"{int(hours)}h {int(minutes)}m {sec:.2f}s"
    if minutes >= 1:
        return f"{int(minutes)}m {sec:.2f}s"

    return f"{sec:.2f}s"


@dataclass
class Timer(ContextDecorator):
    name: str = "timer"
    logger: Any | None = None
    enabled: bool = True
    start_time: float | None = field(default=None, init=False)
    end_time: float | None = field(default=None, init=False)
    elapsed: float | None = field(default=None, init=False)

    def __enter__(self):
        if self.enabled:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled and self.start_time is not None:
            self.end_time = time.perf_counter()
            self.elapsed = self.end_time - self.start_time
            message = f"{self.name} finished in {format_seconds(self.elapsed)}"
            if self.logger is not None:
                self.logger.info(message)
            else:
                print(message)
        return False


def timed(name: str | None = None, logger: Any | None = None):
    def decorator(func: Callable):
        timer_name = name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            with Timer(timer_name, logger=logger):
                return func(*args, **kwargs)

        return wrapper

    return decorator


class AverageMeter:
    def __init__(self, name: str = "meter") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.total = 0.0
        self.avg = 0.0
        self.latest = 0.0

    def update(self, value: float, n: int = 1) -> None:
        self.latest = float(value)
        self.total += float(value) * int(n)
        self.count += int(n)
        self.avg = self.total / max(self.count, 1)

    def as_dict(self) -> dict[str, float]:
        return {
            f"{self.name}_latest": float(self.latest),
            f"{self.name}_avg": float(self.avg),
            f"{self.name}_count": float(self.count),
        }
