import torch
from omegaconf import DictConfig
from torch import Tensor

from src.adapt import run_target_experiment
from utils.io import feature_stats_path
from utils.pca import load_pca_basis


def _build_probe_bank(k: int, device: torch.device) -> Tensor:
    probes = [torch.eye(k, device=device)]
    inv_sqrt2 = 1.0 / (2.0 ** 0.5)
    for i in range(k):
        for j in range(i + 1, k):
            plus = torch.zeros(k, device=device)
            plus[i] = inv_sqrt2
            plus[j] = inv_sqrt2
            probes.append(plus.unsqueeze(0))

            minus = torch.zeros(k, device=device)
            minus[i] = inv_sqrt2
            minus[j] = -inv_sqrt2
            probes.append(minus.unsqueeze(0))
    return torch.cat(probes, dim=0)


def _sym_kl_1d(mu1: Tensor, var1: Tensor,
               mu2: Tensor, var2: Tensor,
               eps: float) -> Tensor:
    var1 = var1 + eps
    var2 = var2 + eps
    diff_sq = (mu1 - mu2).square()
    return 0.5 * (
        var1 / var2
        + var2 / var1
        + diff_sq * (1.0 / var1 + 1.0 / var2)
        - 2.0
    )


def _compute_tau(stat_file, top_k: int, eps: float) -> float:
    import numpy as np

    eigvals = torch.load(stat_file, map_location="cpu")["eigvals"].float()
    top_idx = set(np.argsort(eigvals.numpy())[-top_k:].tolist())
    tail = [float(eigvals[i]) for i in range(len(eigvals)) if i not in top_idx]
    if not tail:
        return eps
    return max(sum(tail) / len(tail), eps)


def _build_step(model, optimizer, device: torch.device, loader):
    method = model
    cfg = _build_step.cfg
    sub_cfg = cfg.method.subspace
    stat_file = feature_stats_path(cfg)
    mean, basis, pc_vars = load_pca_basis(stat_file, int(sub_cfg.top_k), device)
    eps = float(sub_cfg.eps)
    slack_weight = float(sub_cfg.slack_weight)
    k = basis.shape[1]
    d = basis.shape[0]
    tau = _compute_tau(stat_file, int(sub_cfg.top_k), eps)

    q = _build_probe_bank(k, device)
    src_var_q = q.square() @ pc_vars
    head = method.regressor.weight
    a = (basis.T @ head.T).squeeze(-1)
    beta_q = (torch.abs(q @ a) + float(sub_cfg.weight_bias)).pow(
        float(sub_cfg.weight_exp)
    )

    def step(batch):
        x, y = batch
        model.train() if cfg.run.train_mode else model.eval()
        model.zero_grad()

        feature = model.feature(x)
        y_pred = model.predict_from_feature(feature)
        zc = feature - mean
        u = zc @ basis
        residual = zc - u @ basis.T

        proj = u @ q.T
        mu_q = proj.mean(dim=0)
        var_q = proj.var(dim=0, unbiased=False)
        skl_q = _sym_kl_1d(mu_q, var_q, torch.zeros_like(mu_q),
                           src_var_q, eps)
        loss_sig = (beta_q * skl_q).sum() / q.shape[0]

        mu_perp = residual.mean(dim=0)
        r_centered = residual - mu_perp.unsqueeze(0)
        d_minus_k = max(d - k, 1)
        nu_perp = r_centered.square().sum() / (x.shape[0] * d_minus_k + eps) + eps
        mu_sq_norm = mu_perp.square().sum() / (d_minus_k + eps)
        loss_slack = 0.5 * (
            mu_sq_norm * (1.0 / tau + 1.0 / nu_perp)
            + tau / nu_perp
            + nu_perp / tau
            - 2.0
        )
        loss = loss_sig + slack_weight * loss_slack
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return {
            "y_pred": y_pred.detach(),
            "y": y,
            "loss_sig": loss_sig.detach(),
            "loss_slack": loss_slack.detach(),
            "nu_perp": nu_perp.detach(),
        }

    return step


def run(cfg: DictConfig) -> None:  # type: ignore[no-redef]
    _build_step.cfg = cfg
    run_target_experiment(cfg, build_step=_build_step, save_model_prefix="psc")
