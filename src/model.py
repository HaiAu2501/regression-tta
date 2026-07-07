from collections.abc import Iterator
from typing import Any

from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.feature_extraction import create_feature_extractor
import timm


class Regressor(nn.Module):
    regressor: nn.Linear

    def feature(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def predict_from_feature(self, z: Tensor) -> Tensor:
        raise NotImplementedError

    def get_feature_extractor(self) -> nn.Module:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        return self.predict_from_feature(self.feature(x))


class CNNRegressor(Regressor):
    def __init__(self, backbone: str, pretrained: bool, in_channels: int,
                 feature_dim: int):
        super().__init__()
        match backbone:
            case "resnet26":
                base_net = timm.create_model("resnet26", pretrained=pretrained)
                if in_channels != 3:
                    base_net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7,
                                               stride=2, padding=3, bias=False)
                self.feature_extractor = create_feature_extractor(
                    base_net, {"global_pool": "feature"}
                )
            case "resnet50":
                weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
                base_net = resnet50(weights=weights)
                if in_channels != 3:
                    base_net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7,
                                               stride=2, padding=3, bias=False)
                self.feature_extractor = create_feature_extractor(
                    base_net, {"avgpool": "feature"}
                )
            case _:
                raise ValueError(f"Invalid backbone: {backbone!r}")

        self.regressor = nn.Linear(feature_dim, 1)

    def feature(self, x: Tensor) -> Tensor:
        z: Tensor = self.feature_extractor(x)["feature"]
        return z.flatten(start_dim=1)

    def predict_from_feature(self, z: Tensor) -> Tensor:
        return self.regressor(z).flatten()

    def get_feature_extractor(self) -> nn.Module:
        return self.feature_extractor


class MLPRegressor(Regressor):
    def __init__(self, in_dims: int, h_dims: int, n_rep: int):
        super().__init__()
        self.fe = nn.Sequential(
            nn.Linear(in_dims, h_dims, bias=False),
            nn.BatchNorm1d(h_dims),
            nn.ReLU(),
            *(
                nn.Sequential(
                    nn.Linear(h_dims, h_dims, bias=False),
                    nn.BatchNorm1d(h_dims),
                    nn.ReLU(),
                )
                for _ in range(n_rep)
            ),
        )
        self.regressor = nn.Linear(h_dims, 1)

    def feature(self, x: Tensor) -> Tensor:
        if x.ndim >= 3:
            x = x.flatten(start_dim=1)
        return self.fe(x)

    def predict_from_feature(self, z: Tensor) -> Tensor:
        return self.regressor(z).flatten()

    def get_feature_extractor(self) -> nn.Module:
        return self.fe


def create_regressor(cfg: DictConfig | dict[str, Any]) -> Regressor:
    model_cfg = cfg["model"]
    match model_cfg["type"]:
        case "image":
            return CNNRegressor(
                backbone=model_cfg["backbone"],
                pretrained=bool(model_cfg["pretrained"]),
                in_channels=int(model_cfg["in_channels"]),
                feature_dim=int(model_cfg["feature_dim"]),
            )
        case "table":
            return MLPRegressor(**model_cfg["config"])
        case _ as model_type:
            raise ValueError(f"Invalid model type: {model_type!r}")


def extract_bn_layers(module: nn.Module) -> Iterator[_BatchNorm]:
    for child in module.children():
        if isinstance(child, _BatchNorm):
            yield child
        else:
            yield from extract_bn_layers(child)


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(requires_grad)
