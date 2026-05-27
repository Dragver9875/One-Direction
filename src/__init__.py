from .config import (
    ConfigError,
    ProjectConfig,
    apply_cli_overrides,
    deep_get,
    deep_merge,
    deep_set,
    load_config_file,
    load_project_config,
    resolve_device,
)

__version__ = "0.1.0"
__project__ = "One-Direction"

__all__ = [
    "ConfigError",
    "ProjectConfig",
    "apply_cli_overrides",
    "deep_get",
    "deep_merge",
    "deep_set",
    "load_config_file",
    "load_project_config",
    "resolve_device",
]
