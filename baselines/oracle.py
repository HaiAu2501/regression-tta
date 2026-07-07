from omegaconf import DictConfig
import torch.nn.functional as F

from src.adapt import run_target_experiment


def _build_step(model, optimizer, device, loader):
    cfg = _build_step.cfg

    def step(batch):
        x, y = batch
        model.train() if cfg.run.train_mode else model.eval()
        model.zero_grad()
        feature = model.feature(x)
        y_pred = model.predict_from_feature(feature)
        loss = F.mse_loss(y_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return {"y_pred": y_pred.detach(), "y": y, "mse": loss.detach()}

    return step


def run(cfg: DictConfig) -> None:  # type: ignore[no-redef]
    _build_step.cfg = cfg
    run_target_experiment(cfg, build_step=_build_step, save_model_prefix="oracle")
