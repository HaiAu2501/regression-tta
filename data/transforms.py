from collections.abc import Callable

from torchvision.transforms import v2 as transforms


def svhn_transforms(train_aug: bool = True) -> tuple[Callable, Callable]:
    train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])
    val = transforms.ToTensor()
    return (train if train_aug else val), val


def mnist_transforms(train_aug: bool = True) -> tuple[Callable, Callable]:
    train = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])
    val = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(32),
        transforms.ToTensor(),
    ])
    return (train if train_aug else val), val


def utkface_transforms(corrupt_func: Callable | None = None,
                       train_aug: bool = True) -> tuple[Callable, Callable]:
    if corrupt_func is None:
        corrupt_func = lambda x: x

    train_crop = (
        transforms.Compose([
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
        ])
        if train_aug
        else transforms.CenterCrop((224, 224))
    )

    train = transforms.Compose([
        transforms.Resize((256, 256)),
        train_crop,
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])
    val = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        corrupt_func,
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])
    return train, val
