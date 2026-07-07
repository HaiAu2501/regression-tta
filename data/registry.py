from importlib import import_module
from typing import Any

from omegaconf import DictConfig


def build_data(cfg: DictConfig, corruption_type: str | None = None) -> Any:
    module = import_module(cfg.data.module)
    return module.build(cfg, corruption_type=corruption_type)
