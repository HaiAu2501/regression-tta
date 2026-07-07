from importlib import import_module

from omegaconf import DictConfig, OmegaConf

from src import stats, train
from utils.seed import seed_everything


def run(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)

    print(OmegaConf.to_yaml(cfg, resolve=True))
    match cfg.run.stage:
        case "train_source":
            train.run(cfg)
            return
        case "feature_stats":
            stats.run_feature_stats(cfg)
            return
        case "act_stats":
            stats.run_act_stats(cfg)
            return
        case "adapt":
            pass
        case _ as stage:
            raise ValueError(f"Invalid run stage: {stage!r}")

    module = import_module(cfg.method.module)
    module.run(cfg)
