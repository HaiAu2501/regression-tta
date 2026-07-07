from importlib import import_module

from omegaconf import DictConfig, OmegaConf

from utils.seed import seed_everything


def run(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)

    print(OmegaConf.to_yaml(cfg, resolve=True))
    module = import_module(cfg.method.module)
    module.run(cfg)
