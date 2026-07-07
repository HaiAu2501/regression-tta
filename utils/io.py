from pathlib import Path
from typing import Any
import csv
import json

import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_dir() -> Path:
    try:
        return Path(HydraConfig.get().runtime.output_dir)
    except ValueError:
        return Path(".")


def source_key(cfg: DictConfig) -> str:
    return f"{cfg.data.source.name}_{cfg.model.name}"


def source_model_path(cfg: DictConfig) -> Path:
    if cfg.paths.source_model is not None:
        return Path(cfg.paths.source_model)
    return Path(cfg.paths.source_dir) / f"{source_key(cfg)}.pt"


def feature_stats_path(cfg: DictConfig) -> Path:
    if cfg.paths.feature_stats is not None:
        return Path(cfg.paths.feature_stats)
    return Path(cfg.paths.source_dir) / f"{source_key(cfg)}_feature_stats.pt"


def raw_features_path(cfg: DictConfig) -> Path:
    if cfg.paths.raw_features is not None:
        return Path(cfg.paths.raw_features)
    return Path(cfg.paths.source_dir) / f"{source_key(cfg)}_raw_features.pt"


def act_stats_path(cfg: DictConfig) -> Path:
    if cfg.paths.act_stats is not None:
        return Path(cfg.paths.act_stats)
    return Path(cfg.paths.source_dir) / f"{source_key(cfg)}_act_stats.pt"


def save_config(cfg: DictConfig, out_dir: str | Path = ".") -> None:
    if Path(out_dir) == Path("."):
        out_dir = output_dir()
    ensure_dir(out_dir)
    path = Path(out_dir) / "config.yaml"
    path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")


def save_metrics(metrics: Any, out_dir: str | Path) -> None:
    if Path(out_dir) == Path("."):
        out_dir = output_dir()
    out_dir = ensure_dir(out_dir)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    if isinstance(metrics, list) and metrics:
        with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
            writer.writeheader()
            writer.writerows(metrics)


def load_state_dict(path: str | Path, map_location: str | torch.device = "cpu"):
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    if isinstance(state, dict) and "regressor" in state:
        return state["regressor"]
    return state
