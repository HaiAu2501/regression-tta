from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, Subset


@dataclass(frozen=True)
class DatasetBundle:
    source_train: Dataset
    source_eval: Dataset
    target_adapt: Dataset
    target_eval: Dataset


def load_image(path: Path) -> Image.Image:
    with path.open("rb") as f:
        return Image.open(f).convert("RGB")


class ImageTransformDataset(Dataset):
    def __init__(self, dataset: Dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[Tensor, Any]:
        image, label = self.dataset[idx]
        return self.transform(image), label


class ImageSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        self.dataset = Subset(dataset, indices)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]


def random_index_split(n_items: int, train_ratio: float,
                       seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_items)
    split = int(n_items * train_ratio)
    return indices[:split].tolist(), indices[split:].tolist()


def load_indices(path: str | None) -> list[int] | None:
    if path is None:
        return None
    return np.load(path).astype(int).tolist()
