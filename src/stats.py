from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader

from data.registry import build_data
from src.model import create_regressor
from utils.io import (act_stats_path, ensure_dir, feature_stats_path,
                      load_state_dict, raw_features_path, source_model_path)
from utils.pca import compute_feature_stats


@torch.no_grad()
def extract_features(model, loader: DataLoader, device: torch.device) -> Tensor:
    model.eval()
    features = []
    for x, _ in loader:
        x = x.to(device)
        features.append(model.feature(x).cpu())
    return torch.cat(features)


def load_source_model(cfg: DictConfig, device: torch.device):
    model = create_regressor(cfg).to(device)
    path = source_model_path(cfg)
    model.load_state_dict(load_state_dict(path, map_location=device))
    print(f"loaded source model: {path}")
    return model


def run_feature_stats(cfg: DictConfig) -> None:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    bundle = build_data(cfg)
    loader = DataLoader(bundle.source_train, **cfg.data.dataloader.eval)
    model = load_source_model(cfg, device)
    features = extract_features(model, loader, device)
    stats = compute_feature_stats(features)

    out_path = feature_stats_path(cfg)
    ensure_dir(out_path.parent)
    torch.save(stats, out_path)
    print(f"saved feature stats: {out_path}")

    if cfg.stats.save_raw_features:
        raw_path = raw_features_path(cfg)
        torch.save(features, raw_path)
        print(f"saved raw features: {raw_path}")


def _bn_layer_names(model: nn.Module) -> list[str]:
    return [
        name for name, module in model.named_modules()
        if isinstance(module, _BatchNorm)
    ]


def run_act_stats(cfg: DictConfig) -> None:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    bundle = build_data(cfg)
    loader = DataLoader(bundle.source_train, **cfg.data.dataloader.eval)
    model = load_source_model(cfg, device)
    model.train()

    names = _bn_layer_names(model)
    named_modules = dict(model.named_modules())
    activations: dict[str, Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(module, inputs, output):
            activations[name] = output.detach().clone()
        return hook

    for name in names:
        handles.append(named_modules[name].register_forward_hook(make_hook(name)))

    sums: dict[str, Tensor] = {}
    sum_sqs: dict[str, Tensor] = {}
    count = 0

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            model(x)
            batch_size = x.shape[0]
            count += batch_size
            for name in names:
                act = activations[name].to(torch.float64)
                if name not in sums:
                    sums[name] = torch.zeros(
                        act.shape[1:], device=device, dtype=torch.float64
                    )
                    sum_sqs[name] = torch.zeros_like(sums[name])
                sums[name].add_(act.sum(dim=0))
                sum_sqs[name].add_(act.square().sum(dim=0))
            activations.clear()

    for handle in handles:
        handle.remove()

    stats = {}
    for name in names:
        mean = sums[name] / count
        var = sum_sqs[name] / count - mean.square()
        stats[name] = {
            "mean": mean.float().cpu(),
            "var": var.clamp(min=0.0).float().cpu(),
        }

    out_path = act_stats_path(cfg)
    ensure_dir(out_path.parent)
    torch.save({"layer_names": names, "stats": stats, "count": count}, out_path)
    print(f"saved activation stats: {out_path}")
