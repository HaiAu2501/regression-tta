from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


@dataclass
class MetricAccumulator:
    y_true: list[np.ndarray] = field(default_factory=list)
    y_pred: list[np.ndarray] = field(default_factory=list)
    extras: dict[str, list[float]] = field(default_factory=dict)

    def update(self, pred: Tensor, target: Tensor,
               extras: dict[str, Any] | None = None) -> None:
        self.y_pred.append(pred.detach().flatten().cpu().numpy())
        self.y_true.append(target.detach().flatten().cpu().numpy())
        if extras is None:
            return
        for key, value in extras.items():
            if isinstance(value, Tensor):
                if value.numel() != 1:
                    continue
                value = value.detach().cpu().item()
            if isinstance(value, (int, float)):
                self.extras.setdefault(key, []).append(float(value))

    def compute(self) -> dict[str, float]:
        y = np.concatenate(self.y_true)
        pred = np.concatenate(self.y_pred)
        diff = pred - y
        mse = float(np.mean(diff ** 2))
        mae = float(np.mean(np.abs(diff)))
        ss_res = float(np.sum(diff ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        if len(y) > 1 and np.std(y) > 0 and np.std(pred) > 0:
            corr = float(np.corrcoef(y, pred)[0, 1])
        else:
            corr = float("nan")

        metrics = {
            "rmse_loss": math.sqrt(mse),
            "mae_loss": mae,
            "R2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "r": corr,
        }
        for key, values in self.extras.items():
            metrics[key] = float(np.mean(values))
        return metrics


@torch.no_grad()
def model_distance(model: nn.Module, source_state: dict[str, Tensor]) -> float:
    dist = torch.tensor(0.0)
    for key, param in model.state_dict().items():
        if key not in source_state or not torch.is_floating_point(param):
            continue
        dist += (param.detach().cpu() - source_state[key].cpu()).square().sum()
    return float(dist.sqrt().item())
