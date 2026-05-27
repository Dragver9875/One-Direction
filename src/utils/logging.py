from __future__ import annotations

import logging
import sys
from pathlib import Path


_LOGGER_CACHE: dict[str, logging.Logger] = {}


def configure_logger(
    name: str = "one_direction",
    level: str | int = "INFO",
    log_file: str | Path | None = None,
    console: bool = True,
    overwrite_handlers: bool = True,
) -> logging.Logger:
    logger = logging.getLogger(name)

    if overwrite_handlers:
        logger.handlers.clear()

    logger.setLevel(_resolve_level(level))
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(_resolve_level(level))
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(_resolve_level(level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _LOGGER_CACHE[name] = logger
    return logger


def get_logger(
    name: str = "one_direction",
    level: str | int = "INFO",
) -> logging.Logger:
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]

    logger = logging.getLogger(name)
    if not logger.handlers:
        configure_logger(name=name, level=level)

    _LOGGER_CACHE[name] = logger
    return logger


def _resolve_level(level: str | int) -> int:
    if isinstance(level, int):
        return level

    text = str(level).upper()
    if not hasattr(logging, text):
        raise ValueError(f"Unsupported logging level: {level}")

    return int(getattr(logging, text))


def log_dict(
    logger: logging.Logger,
    data: dict,
    prefix: str = "",
    level: str | int = "INFO",
) -> None:
    numeric_level = _resolve_level(level)
    for key, value in data.items():
        logger.log(numeric_level, "%s%s: %s", prefix, key, value)


def silence_external_loggers(names: list[str] | None = None, level: str | int = "WARNING") -> None:
    if names is None:
        names = ["matplotlib", "fiona", "pyogrio", "geopandas", "shapely"]

    numeric_level = _resolve_level(level)
    for name in names:
        logging.getLogger(name).setLevel(numeric_level)
