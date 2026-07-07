from omegaconf import DictConfig
from torch.nn.modules.batchnorm import _BatchNorm

from src.adapt import run_target_experiment


def _bn_layers(model):
    return [module for module in model.modules() if isinstance(module, _BatchNorm)]


def _build_step(model, optimizer, device, loader):
    cfg = _build_step.cfg
    layers = _bn_layers(model)
    if cfg.method.reset_stats:
        for layer in layers:
            layer.reset_running_stats()
    for layer in layers:
        layer.momentum = cfg.method.bn_momentum

    def step(batch):
        x, y = batch
        model.train() if cfg.run.train_mode else model.eval()
        feature = model.feature(x)
        y_pred = model.predict_from_feature(feature)
        return {"y_pred": y_pred.detach(), "y": y}

    return step


def run(cfg: DictConfig) -> None:
    _build_step.cfg = cfg
    run_target_experiment(
        cfg,
        build_step=_build_step,
        use_optimizer=False,
        save_model_prefix="bna",
    )
