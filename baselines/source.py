from omegaconf import DictConfig

from src.adapt import run_target_experiment


def run(cfg: DictConfig) -> None:
    run_target_experiment(
        cfg,
        build_step=lambda model, optimizer, device, loader: None,
        use_optimizer=False,
    )
