from omegaconf import DictConfig
import torch
from torch import nn
import torch.nn.functional as F

from src.adapt import run_target_experiment


class RotationHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int | None):
        super().__init__()
        if hidden_dim is None:
            self.head = nn.Linear(feature_dim, 4)
        else:
            self.head = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 4),
            )

    def forward(self, x):
        return self.head(x)


def _rotate_batch(x):
    batch_size = x.shape[0]
    x_rot = torch.cat(
        [
            x,
            torch.rot90(x, k=1, dims=[2, 3]),
            torch.rot90(x, k=2, dims=[2, 3]),
            torch.rot90(x, k=3, dims=[2, 3]),
        ],
        dim=0,
    )
    labels = torch.cat([
        torch.full((batch_size,), i, device=x.device, dtype=torch.long)
        for i in range(4)
    ])
    return x_rot, labels


def _build_step(model, optimizer, device, loader):
    cfg = _build_step.cfg
    hidden_dim = cfg.method.rotation.hidden_dim
    if hidden_dim is not None:
        hidden_dim = int(hidden_dim)
    head = RotationHead(model.regressor.in_features, hidden_dim).to(device)
    rot_opt = torch.optim.Adam(
        head.parameters(),
        lr=float(cfg.method.rotation.lr),
        weight_decay=float(cfg.method.rotation.weight_decay),
    )

    def step(batch):
        x, y = batch
        model.train() if cfg.run.train_mode else model.eval()
        head.train()
        x_rot, labels = _rotate_batch(x)
        optimizer.zero_grad()
        rot_opt.zero_grad()
        logits = head(model.feature(x_rot))
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        rot_opt.step()

        with torch.no_grad():
            model.eval()
            feature = model.feature(x)
            y_pred = model.predict_from_feature(feature)
            acc = (logits.argmax(dim=1) == labels).float().mean()
        return {
            "y_pred": y_pred.detach(),
            "y": y,
            "rot_loss": loss.detach(),
            "rot_acc": acc.detach(),
        }

    return step


def run(cfg: DictConfig) -> None:
    _build_step.cfg = cfg
    run_target_experiment(cfg, build_step=_build_step, save_model_prefix="ttt")
