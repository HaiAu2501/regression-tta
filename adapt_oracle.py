"""
Oracle upper-bound: finetune the source model on the corrupted target data
**with labels**, using the **same optimizer, parameters, and single-pass
budget** as ``adapt_ssa.py``.

This is NOT a valid TTA method (it uses ground-truth labels at test time).
Its sole purpose is to provide an **upper-bound** on what any label-free
TTA method can hope to achieve under the same optimization budget.

Usage::

    python adapt_oracle.py -c configs/tta/utkface.yaml -o result/oracle/utkface
    python adapt_oracle.py -c configs/tta/svhn.yaml    -o result/oracle/svhn
"""
from typing import Any
from pprint import pprint
import json
import csv
import copy
import itertools
from pathlib import Path

import yaml

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from ignite.engine import Engine, Events
from ignite.metrics import RootMeanSquaredError, MeanAbsoluteError
from ignite.contrib.metrics.regression.r2_score import R2Score

from utils.seed import fix_seed
from model import create_regressor, Regressor, extract_bn_layers
from dataset import get_datasets
from dataset.corruptions import CORRUPTION_TYPES
from evaluation.evaluator import RegressionEvaluator
from evaluation.metrics import PearsonCorrelation


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Oracle upper-bound: finetune with labels on target data.")
    parser.add_argument("-c", required=True, help="config")
    parser.add_argument("-o", required=True, help="output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", action="store_true", help="save model")

    args = parser.parse_args()
    pprint(vars(args))
    main(args)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main(args):
    fix_seed(args.seed)

    with open(args.c, "r", encoding="utf-8") as f:
        if args.c.endswith(".json"):
            config = json.load(f)
        else:
            config = yaml.safe_load(f)
    pprint(config)

    Path(args.o).mkdir(parents=True, exist_ok=True)
    with Path(args.o, "config.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(config, f)

    corruption_cfg = config["dataset"].get("val_corruption", None)
    is_all = (corruption_cfg is not None
              and corruption_cfg.get("corruption_type") == "all")

    if is_all:
        severity = corruption_cfg["severity"]
        all_rows: list[dict[str, Any]] = []

        for ctype in CORRUPTION_TYPES:
            print(f"\n{'='*60}")
            print(f"  [Oracle] Corruption: {ctype}  (severity={severity})")
            print(f"{'='*60}")

            cfg = copy.deepcopy(config)
            cfg["dataset"]["val_corruption"]["corruption_type"] = ctype

            metrics = run_single(cfg, args)
            row = {"corruption_type": ctype, "severity": severity}
            for phase in ("online", "offline"):
                for k, v in metrics[phase].items():
                    row[f"{phase}/{k}"] = v
            all_rows.append(row)

        # --- write CSV ---
        csv_path = Path(args.o, "metrics.csv")
        fieldnames = list(all_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nSaved all-corruption results to {csv_path}")

        with Path(args.o, "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=4, ensure_ascii=False)
    else:
        metrics = run_single(config, args)
        with Path(args.o, "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────
# Oracle finetune engine
# ──────────────────────────────────────────────────────────────────────
class OracleSupervisedEngine(Engine):
    """Engine that mirrors the TTA loop but uses full supervised MSE loss.

    This keeps the same feature extraction, train_mode behaviour and
    metrics as `TTAEngine` but replaces the KL-based unsupervised loss
    with a supervised MSE on the target labels. This ensures the oracle
    uses the same optimization budget / loop as SSA while using labels.
    """

    def __init__(self, net: Regressor, opt: torch.optim.Optimizer,
                 train_mode: bool = True, compile_model: dict | None = None):
        self.net = net
        self.opt = opt
        self.train_mode = train_mode
        super().__init__(self.update)

        y_ot = lambda d: (d["y_pred"], d["y"])
        RootMeanSquaredError(y_ot).attach(self, "rmse_loss")
        MeanAbsoluteError(y_ot).attach(self, "mae_loss")
        R2Score(y_ot).attach(self, "R2")
        PearsonCorrelation(y_ot).attach(self, "r")

        # mimic TTAEngine behaviour w.r.t. feature compilation
        self.feature_extractor = self.net.feature
        if compile_model is not None:
            try:
                self.feature_extractor = torch.compile(
                    self.net.feature, **compile_model)
            except RuntimeError as e:
                print(f"torch.compile failed: {e}")

    def update(self, engine: Engine,
               batch: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
        if self.train_mode:
            self.net.train()
        else:
            self.net.eval()
        self.net.zero_grad()

        x, y = batch
        x = x.cuda()
        y = y.float().flatten().cuda()

        feature = self.feature_extractor(x)
        y_pred = self.net.predict_from_feature(feature)

        loss = F.mse_loss(y_pred, y)
        loss.backward()
        self.opt.step()

        return {
            "y_pred": y_pred,
            "y": y,
        }


# ──────────────────────────────────────────────────────────────────────
# Single-corruption run
# ──────────────────────────────────────────────────────────────────────
def run_single(config: dict[str, Any], args) -> dict[str, Any]:
    """Finetune the source model on corrupted target data WITH labels,
    then run offline evaluation."""
    fix_seed(args.seed)

    # --- load source model ------------------------------------------------
    regressor = create_regressor(config).cuda()
    regressor.load_state_dict(torch.load(p := config["regressor"]["source"]))
    print(f"load {p}")

    # --- target dataset (corrupted val data) ------------------------------
    _, val_ds = get_datasets(config)
    adapt_dl = DataLoader(val_ds, **config["adapt_dataloader"])

    # --- build optimizer using same policy as adapt_ssa -------------------
    def create_optimizer(net: Regressor, config: dict[str, Any]) -> torch.optim.Optimizer:
        match config["optimizer"]["param"]:
            case "all":
                params = net.parameters()
            case "fe":
                params = net.get_feature_extractor().parameters()
            case "fe_bn":
                bn_layers = extract_bn_layers(net.get_feature_extractor())
                params = itertools.chain.from_iterable(
                    l.parameters() for l in bn_layers
                )
            case _ as p:
                raise ValueError(f"Invalid param: {p!r}")

        opt = eval(f"torch.optim.{config['optimizer']['name']}")(
            params, **config["optimizer"]["config"])
        return opt

    opt = create_optimizer(regressor, config)

    # --- supervised oracle engine (mirror TTA loop/budget) ---------------
    tta_cfg = config["tta"]["config"]
    engine = OracleSupervisedEngine(
        net=regressor,
        opt=opt,
        train_mode=tta_cfg.get("train_mode", True),
        compile_model=tta_cfg.get("compile_model", None),
    )

    if args.save:
        from ignite.handlers import ModelCheckpoint
        engine.add_event_handler(
            Events.COMPLETED,
            ModelCheckpoint(args.o, "oracle", require_empty=False),
            {"regressor": regressor})

    # run with the SAME budget as SSA: a single run over adapt_dataloader
    engine.run(adapt_dl)

    finetune_metrics = dict(engine.state.metrics)

    # --- offline evaluation (eval mode, no grad) --------------------------
    eval_dl = DataLoader(val_ds, **config["val_dataloader"])
    reg_evaluator = RegressionEvaluator(regressor, **config["evaluator"])
    reg_evaluator.run(eval_dl)

    metrics = {
        "iteration": engine.state.iteration,
        "online": engine.state.metrics,
        "offline": dict(reg_evaluator.state.metrics),
    }
    pprint(metrics)
    return metrics


if __name__ == "__main__":
    parse_args()
