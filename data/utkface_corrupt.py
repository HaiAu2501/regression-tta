from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Dataset

from data.corruptions import get_corruption_func
from data.splits import (DatasetBundle, ImageSubset, ImageTransformDataset,
                         load_image, load_indices, random_index_split)
from data.transforms import utkface_transforms


@dataclass
class UTKFace(Dataset):
    root: Path
    filter_gender: int | None = None

    def __post_init__(self) -> None:
        self.path_list = sorted(self.root.glob("*.jpg"))
        self.ages = np.array(
            [float(path.name.split("_")[0]) for path in self.path_list],
            dtype=np.float32,
        )

        if self.filter_gender is None:
            return

        genders = np.array(
            [int(path.name.split("_")[1]) for path in self.path_list],
            dtype=np.int64,
        )
        mask = genders == self.filter_gender
        self.path_list = [path for path, keep in zip(self.path_list, mask) if keep]
        self.ages = self.ages[mask]

    def __len__(self) -> int:
        return len(self.path_list)

    def __getitem__(self, idx: int):
        return load_image(self.path_list[idx]), float(self.ages[idx])


def build(cfg: DictConfig, corruption_type: str | None = None) -> DatasetBundle:
    """Build the UTKFace source -> corrupted UTKFace target setup."""
    root = Path(cfg.paths.data_root) / "UTKFace" / "UTKFace"
    dataset = UTKFace(root, filter_gender=cfg.data.source.filter_gender)

    split_cfg = cfg.data.target.split
    val_indices = load_indices(split_cfg.val_indices)
    if val_indices is None:
        train_indices, val_indices = random_index_split(
            len(dataset), float(split_cfg.train_ratio), int(cfg.seed)
        )
    else:
        train_mask = np.ones(len(dataset), dtype=np.bool_)
        train_mask[val_indices] = False
        train_indices = np.arange(len(dataset))[train_mask].tolist()

    corrupt_cfg = cfg.data.target.corruption
    selected_corruption = corruption_type or corrupt_cfg.type
    corrupt_func = None
    if selected_corruption is not None and selected_corruption != "none":
        corrupt_func = get_corruption_func(
            selected_corruption,
            int(corrupt_cfg.severity),
        )

    train_tf, target_tf = utkface_transforms(
        corrupt_func=corrupt_func,
        train_aug=cfg.run.stage == "train_source",
    )
    _, source_eval_tf = utkface_transforms(corrupt_func=None, train_aug=False)

    source_train = ImageTransformDataset(ImageSubset(dataset, train_indices), train_tf)
    source_eval = ImageTransformDataset(ImageSubset(dataset, val_indices),
                                        source_eval_tf)
    target_eval = ImageTransformDataset(ImageSubset(dataset, val_indices), target_tf)
    return DatasetBundle(
        source_train=source_train,
        source_eval=source_eval,
        target_adapt=target_eval,
        target_eval=target_eval,
    )
