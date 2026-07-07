from collections.abc import Callable
import copy
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from data.corruptions import CORRUPTION_TYPES
from data.registry import build_data
from src.eval import evaluate
from src.model import Regressor, create_regressor, extract_bn_layers
from utils.io import (ensure_dir, load_state_dict, save_config, save_metrics,
                      source_model_path)
from utils.logging import flatten_metrics
from utils.metrics import MetricAccumulator

AdaptStep = Callable[[tuple[torch.Tensor, torch.Tensor]], dict[str, Any]]


def device_from_cfg(cfg: DictConfig) -> torch.device:
    if str(cfg.device).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(cfg.device)


def load_model(cfg: DictConfig, device: torch.device) -> Regressor:
    model = create_regressor(cfg).to(device)
    path = source_model_path(cfg)
    model.load_state_dict(load_state_dict(path, map_location=device))
    print(f"loaded source model: {path}")
    return model


def make_optimizer(model: Regressor, cfg: DictConfig) -> torch.optim.Optimizer:
    opt_cfg = cfg.method.optimizer
    match opt_cfg.params:
        case "all":
            params = model.parameters()
        case "fe":
            params = model.get_feature_extractor().parameters()
        case "fe_bn":
            params = []
            for layer in extract_bn_layers(model.get_feature_extractor()):
                params.extend(layer.parameters())
        case _ as param_mode:
            raise ValueError(f"Invalid optimizer params: {param_mode!r}")

    return getattr(torch.optim, opt_cfg.name)(
        params,
        lr=float(opt_cfg.lr),
        weight_decay=float(opt_cfg.weight_decay),
    )


def run_online(loader: DataLoader, step: AdaptStep,
               device: torch.device) -> dict[str, float]:
    acc = MetricAccumulator()
    for batch in loader:
        x, y = batch
        out = step((x.to(device), y.float().flatten().to(device)))
        extras = {k: v for k, v in out.items() if k not in {"y_pred", "y"}}
        acc.update(out["y_pred"], out["y"], extras)
    return acc.compute()


def _target_corruptions(cfg: DictConfig) -> list[str | None]:
    corruption = OmegaConf.select(cfg, "data.target.corruption.type")
    if corruption == "all":
        return CORRUPTION_TYPES
    return [corruption]


def run_target_experiment(
    cfg: DictConfig,
    build_step: Callable[[Regressor, torch.optim.Optimizer | None,
                          torch.device, DataLoader], AdaptStep | None],
    use_optimizer: bool = True,
    save_model_prefix: str | None = None,
) -> Any:
    device = device_from_cfg(cfg)
    rows = []
    corruptions = _target_corruptions(cfg)

    for corruption_type in corruptions:
        if corruption_type is not None:
            print(f"corruption={corruption_type}")
        bundle = build_data(cfg, corruption_type=corruption_type)
        adapt_loader = DataLoader(bundle.target_adapt,
                                  **cfg.data.dataloader.adapt)
        eval_loader = DataLoader(bundle.target_eval,
                                 **cfg.data.dataloader.eval)

        model = load_model(cfg, device)
        source_state = copy.deepcopy(model.state_dict())
        optimizer = make_optimizer(model, cfg) if use_optimizer else None
        step = build_step(model, optimizer, device, adapt_loader)

        online_metrics = {}
        if step is not None:
            online_metrics = run_online(adapt_loader, step, device)

        offline_metrics = evaluate(model, eval_loader, device, source_state)
        metrics = {
            "iteration": len(adapt_loader) if step is not None else 0,
            "online": online_metrics,
            "offline": offline_metrics,
        }

        if cfg.run.save_model and save_model_prefix is not None:
            out = ensure_dir("adapted")
            suffix = corruption_type or "target"
            torch.save(model.state_dict(), out / f"{save_model_prefix}_{suffix}.pt")

        if len(corruptions) == 1:
            save_config(cfg, ".")
            save_metrics(metrics, ".")
            return metrics

        row = {
            "corruption_type": corruption_type,
            "severity": OmegaConf.select(cfg, "data.target.corruption.severity"),
        }
        row.update(flatten_metrics("online", online_metrics))
        row.update(flatten_metrics("offline", offline_metrics))
        rows.append(row)

    save_config(cfg, ".")
    save_metrics(rows, ".")
    return rows
