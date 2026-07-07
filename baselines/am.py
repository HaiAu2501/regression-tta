from omegaconf import DictConfig
import torch

from src.adapt import run_target_experiment
from utils.io import act_stats_path


def _select_layers(names: list[str], max_layers: int | None) -> list[str]:
    if max_layers is None or max_layers >= len(names):
        return names
    if max_layers <= 1:
        return [names[-1]]
    step = (len(names) - 1) / (max_layers - 1)
    selected = []
    seen = set()
    for i in range(max_layers):
        idx = round(i * step)
        if idx not in seen:
            seen.add(idx)
            selected.append(names[idx])
    return selected


def _build_step(model, optimizer, device, loader):
    cfg = _build_step.cfg
    stat_dict = torch.load(act_stats_path(cfg), map_location="cpu")
    max_layers = cfg.method.max_layers
    if max_layers is not None:
        max_layers = int(max_layers)
    layer_names = _select_layers(stat_dict["layer_names"], max_layers)
    src_means = {
        name: stat_dict["stats"][name]["mean"].to(device)
        for name in layer_names
    }
    src_vars = {
        name: stat_dict["stats"][name]["var"].to(device)
        for name in layer_names
    }
    activations = {}
    handles = []
    modules = dict(model.named_modules())

    def make_hook(name):
        def hook(module, inputs, output):
            activations[name] = output.clone()
        return hook

    for name in layer_names:
        handles.append(modules[name].register_forward_hook(make_hook(name)))

    def step(batch):
        x, y = batch
        model.train() if cfg.run.train_mode else model.eval()
        model.zero_grad()
        feature = model.feature(x)
        y_pred = model.predict_from_feature(feature)
        loss = torch.tensor(0.0, device=device)
        for name in layer_names:
            act = activations[name]
            mean = act.mean(dim=0)
            var = act.var(dim=0, unbiased=False)
            loss = loss + (mean - src_means[name]).abs().mean()
            loss = loss + (var - src_vars[name]).abs().mean()
        loss = loss / max(len(layer_names), 1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        activations.clear()
        return {"y_pred": y_pred.detach(), "y": y, "am_loss": loss.detach()}

    return step


def run(cfg: DictConfig) -> None:
    _build_step.cfg = cfg
    run_target_experiment(cfg, build_step=_build_step, save_model_prefix="am")
