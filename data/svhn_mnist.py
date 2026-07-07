from pathlib import Path

from omegaconf import DictConfig
from torchvision import datasets

from data.splits import DatasetBundle
from data.transforms import mnist_transforms, svhn_transforms


def build(cfg: DictConfig, corruption_type: str | None = None) -> DatasetBundle:
    """Build the SVHN source -> MNIST target regression setup."""
    if corruption_type is not None:
        raise ValueError("svhn_mnist does not support image corruptions.")

    data_root = Path(cfg.paths.data_root)
    download = bool(cfg.data.get("download", False))
    train_aug = cfg.run.stage == "train_source"

    svhn_train_tf, svhn_eval_tf = svhn_transforms(train_aug=train_aug)
    _, mnist_eval_tf = mnist_transforms(train_aug=False)

    source_train = datasets.SVHN(
        data_root / "SVHN", split="train", transform=svhn_train_tf,
        download=download,
    )
    source_eval = datasets.SVHN(
        data_root / "SVHN", split="test", transform=svhn_eval_tf,
        download=download,
    )
    target_eval = datasets.MNIST(
        data_root / "MNIST", train=False, transform=mnist_eval_tf,
        download=download,
    )
    return DatasetBundle(
        source_train=source_train,
        source_eval=source_eval,
        target_adapt=target_eval,
        target_eval=target_eval,
    )
