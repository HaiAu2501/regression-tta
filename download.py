import kagglehub
from torchvision import datasets

# MNIST
# datasets.MNIST(root="./data/MNIST", train=True, download=True)
# datasets.MNIST(root="./data/MNIST", train=False, download=True)

# SVHN
# datasets.SVHN(root="./data/SVHN", split="train", download=True)
# datasets.SVHN(root="./data/SVHN", split="test", download=True)

# UTKFace
# kagglehub.dataset_download("jangedoo/utkface-new", output_dir="./data/UTKFace")