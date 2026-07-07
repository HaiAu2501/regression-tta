from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig


@dataclass(frozen=True)
class DatasetPair:
    source_train: Any = None
    source_eval: Any = None
    target_adapt: Any = None
    target_eval: Any = None


def build(cfg: DictConfig) -> DatasetPair:
    """Build the UTKFace source -> corrupted UTKFace target setup."""
    raise NotImplementedError(
        "Port UTKFace split/corruption loading from bin/dataset/utkface.py here."
    )
