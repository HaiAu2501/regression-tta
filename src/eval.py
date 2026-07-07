import torch
from torch.utils.data import DataLoader

from src.model import Regressor
from utils.metrics import MetricAccumulator, model_distance


@torch.no_grad()
def evaluate(model: Regressor, loader: DataLoader, device: torch.device,
             source_state: dict | None = None) -> dict[str, float]:
    model.eval()
    acc = MetricAccumulator()
    for x, y in loader:
        x = x.to(device)
        y = y.float().flatten().to(device)
        pred = model(x)
        acc.update(pred, y)

    metrics = acc.compute()
    if source_state is not None:
        metrics["model_dist"] = model_distance(model, source_state)
    return metrics
