from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from data.registry import build_data
from src.eval import evaluate
from src.model import create_regressor
from utils.io import ensure_dir, save_config, source_model_path
from utils.metrics import MetricAccumulator


def _make_optimizer(model: torch.nn.Module, cfg: DictConfig):
    opt_cfg = cfg.train.optimizer
    return getattr(torch.optim, opt_cfg.name)(
        model.parameters(),
        lr=float(opt_cfg.lr),
        weight_decay=float(opt_cfg.weight_decay),
    )


def run(cfg: DictConfig) -> None:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    bundle = build_data(cfg)
    train_loader = DataLoader(bundle.source_train, **cfg.data.dataloader.train)
    eval_loader = DataLoader(bundle.source_eval, **cfg.data.dataloader.eval)

    model = create_regressor(cfg).to(device)
    optimizer = _make_optimizer(model, cfg)

    for epoch in range(int(cfg.train.epochs)):
        model.train()
        acc = MetricAccumulator()
        for x, y in train_loader:
            x = x.to(device)
            y = y.float().flatten().to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            acc.update(pred, y, {"mse": loss.detach()})

        train_metrics = acc.compute()
        val_metrics = evaluate(model, eval_loader, device)
        print(f"epoch={epoch + 1} train={train_metrics} val={val_metrics}")

    ckpt_path = source_model_path(cfg)
    ensure_dir(ckpt_path.parent)
    torch.save(model.state_dict(), ckpt_path)
    save_config(cfg, Path("."))
    print(f"saved source model: {ckpt_path}")
