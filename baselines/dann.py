from omegaconf import DictConfig
import torch
from torch import nn

from src.adapt import run_target_experiment
from utils.io import raw_features_path


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def _grl(x, lambda_):
    return _GradientReverse.apply(x, lambda_)


class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _build_step(model, optimizer, device, loader):
    cfg = _build_step.cfg
    source_features = torch.load(raw_features_path(cfg), map_location=device)
    discriminator = DomainDiscriminator(
        source_features.shape[1],
        int(cfg.method.discriminator.hidden_dim),
    ).to(device)
    disc_opt = torch.optim.Adam(
        discriminator.parameters(),
        lr=float(cfg.method.discriminator.lr),
        weight_decay=float(cfg.method.discriminator.weight_decay),
    )
    bce = nn.BCEWithLogitsLoss()
    max_iter = max(len(loader), 1)
    iteration = 0

    def step(batch):
        nonlocal iteration
        iteration += 1
        x, y = batch
        model.train() if cfg.run.train_mode else model.eval()
        target_feat = model.feature(x)
        y_pred = model.predict_from_feature(target_feat)
        batch_size = x.shape[0]
        idx = torch.randint(0, source_features.shape[0], (batch_size,),
                            device=device)
        source_feat = source_features[idx]
        source_label = torch.zeros(batch_size, device=device)
        target_label = torch.ones(batch_size, device=device)

        disc_opt.zero_grad()
        d_source = discriminator(source_feat.detach())
        d_target = discriminator(target_feat.detach())
        disc_loss = (bce(d_source, source_label) + bce(d_target, target_label)) / 2
        disc_loss.backward()
        disc_opt.step()

        progress = min(iteration / max_iter, 1.0)
        lambda_ = float(cfg.method.lambda_init) + (
            float(cfg.method.lambda_final) - float(cfg.method.lambda_init)
        ) * progress
        optimizer.zero_grad()
        fe_loss = bce(discriminator(_grl(target_feat, lambda_)), target_label)
        fe_loss.backward()
        optimizer.step()
        return {
            "y_pred": y_pred.detach(),
            "y": y,
            "disc_loss": disc_loss.detach(),
            "fe_loss": fe_loss.detach(),
            "lambda": lambda_,
        }

    return step


def run(cfg: DictConfig) -> None:
    _build_step.cfg = cfg
    run_target_experiment(cfg, build_step=_build_step, save_model_prefix="dann")
