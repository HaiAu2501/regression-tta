from omegaconf import DictConfig
import torch

from src.adapt import run_target_experiment
from utils.io import feature_stats_path
from utils.pca import load_pca_basis


def _build_step(model, optimizer, device, loader):
    cfg = _build_step.cfg
    sub_cfg = cfg.method.subspace
    mean, basis, pc_vars = load_pca_basis(
        feature_stats_path(cfg), int(sub_cfg.top_k), device
    )
    src_sv = pc_vars.sqrt()
    trade_off = float(sub_cfg.trade_off)

    def step(batch):
        x, y = batch
        model.train() if cfg.run.train_mode else model.eval()
        model.zero_grad()
        feature = model.feature(x)
        y_pred = model.predict_from_feature(feature)
        centered = feature - mean
        _, s_t, vh_t = torch.linalg.svd(centered, full_matrices=False)
        k = min(basis.shape[1], x.shape[0])
        tgt_sv = s_t[:k]
        v_t = vh_t[:k, :].T
        v_s = basis[:, :k]
        s_s = src_sv[:k]
        cross = v_s.T @ v_t
        weighted_cross = (s_s.unsqueeze(1) * cross) * tgt_sv.unsqueeze(0)
        rsd_loss = (
            s_s.square().sum()
            + tgt_sv.square().sum()
            - 2.0 * weighted_cross.trace()
        ) / k
        loss = trade_off * rsd_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return {"y_pred": y_pred.detach(), "y": y, "rsd_loss": loss.detach()}

    return step


def run(cfg: DictConfig) -> None:
    _build_step.cfg = cfg
    run_target_experiment(cfg, build_step=_build_step, save_model_prefix="rsd")
