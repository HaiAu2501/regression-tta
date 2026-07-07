from omegaconf import DictConfig
import torch

from src.adapt import run_target_experiment
from utils.io import feature_stats_path
from utils.loss import diagonal_gaussian_kl_loss
from utils.pca import load_pca_basis


def _build_step(model, optimizer, device, loader):
    cfg = _build_step.cfg
    sub_cfg = cfg.method.subspace
    mean, basis, var = load_pca_basis(
        feature_stats_path(cfg), int(sub_cfg.top_k), device
    )
    dim_weight = torch.abs(model.regressor.weight @ basis).flatten()
    dim_weight = (dim_weight + float(sub_cfg.weight_bias)).pow(
        float(sub_cfg.weight_exp)
    )
    eps = float(sub_cfg.eps)

    def step(batch):
        x, y = batch
        model.train() if cfg.run.train_mode else model.eval()
        model.zero_grad()
        feature = model.feature(x)
        y_pred = model.predict_from_feature(feature)
        f_pc = (feature - mean) @ basis
        f_mean = f_pc.mean(dim=0)
        f_var = f_pc.var(dim=0)
        zeros = torch.zeros_like(f_mean)
        kl = diagonal_gaussian_kl_loss(
            f_mean, f_var, zeros, var, eps=eps, dim_reduction="none"
        ) + diagonal_gaussian_kl_loss(
            zeros, var, f_mean, f_var, eps=eps, dim_reduction="none"
        )
        loss = kl @ dim_weight
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return {"y_pred": y_pred.detach(), "y": y, "ssa_loss": loss.detach()}

    return step


def run(cfg: DictConfig) -> None:
    _build_step.cfg = cfg
    run_target_experiment(cfg, build_step=_build_step, save_model_prefix="ssa")
