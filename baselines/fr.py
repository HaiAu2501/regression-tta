from omegaconf import DictConfig
import torch
from torch.nn.modules.batchnorm import _BatchNorm

from src.adapt import run_target_experiment


def _kl_q_p(q_mean, q_var, p_mean, p_var, eps: float):
    p_var = p_var + eps
    q_var = q_var + eps
    return (
        0.5 * torch.log(p_var / q_var)
        + (q_var + (q_mean - p_mean).square()) / (2.0 * p_var)
        - 0.5
    )


class _Capture:
    def __init__(self, layers, eps: float):
        self.eps = eps
        self.src_means = [layer.running_mean.detach().clone() for layer in layers]
        self.src_vars = [layer.running_var.detach().clone() for layer in layers]
        self.means = []
        self.vars = []
        self.capturing = False
        self.handles = [
            layer.register_forward_pre_hook(self._hook())
            for layer in layers
        ]

    def _hook(self):
        def hook(module, args):
            if not self.capturing:
                return
            x = args[0]
            dims = [0] + list(range(2, x.ndim))
            self.means.append(x.mean(dims))
            self.vars.append(x.var(dims, unbiased=False))
        return hook

    def start(self):
        self.capturing = True
        self.means = []
        self.vars = []

    def stop(self):
        self.capturing = False

    def loss(self, active_layers):
        total = torch.tensor(0.0, device=self.means[0].device)
        units = 0
        for idx in active_layers:
            kl = _kl_q_p(
                self.means[idx],
                self.vars[idx],
                self.src_means[idx].to(self.means[idx].device),
                self.src_vars[idx].to(self.means[idx].device),
                self.eps,
            )
            total = total + kl.sum()
            units += kl.numel()
        return total / max(units, 1)


def _build_step(model, optimizer, device, loader):
    cfg = _build_step.cfg
    layers = [m for m in model.modules() if isinstance(m, _BatchNorm)]
    capture = _Capture(layers, float(cfg.method.eps))
    active = cfg.method.active_layers
    active_layers = list(range(len(layers))) if active is None else list(active)

    def step(batch):
        x, y = batch
        model.train() if cfg.run.train_mode else model.eval()
        optimizer.zero_grad()
        capture.start()
        feature = model.feature(x)
        y_pred = model.predict_from_feature(feature)
        capture.stop()
        loss = capture.loss(active_layers)
        loss.backward()
        optimizer.step()
        return {"y_pred": y_pred.detach(), "y": y, "fr_loss": loss.detach()}

    return step


def run(cfg: DictConfig) -> None:
    _build_step.cfg = cfg
    run_target_experiment(cfg, build_step=_build_step, save_model_prefix="fr")
