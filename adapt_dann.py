"""
DANN-based Test-Time Adaptation.

Usage::

    python adapt_dann.py -c configs/tta/utkface_dann.yaml -o result/tta_dann/utkface

When ``corruption_type`` is ``"all"``, every corruption in
:data:`CORRUPTION_TYPES` is evaluated and the results are saved to a CSV.
"""
from typing import Any
from pprint import pprint
import json
import csv
import copy
from pathlib import Path
import itertools

import yaml

import torch
from torch.utils.data import DataLoader
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint

from utils.seed import fix_seed
from model import create_regressor, Regressor, extract_bn_layers
from dataset import get_datasets
from dataset.corruptions import CORRUPTION_TYPES
from evaluation.evaluator import RegressionEvaluator
from tta.dann_engine import DANNEngine


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="DANN-based test-time adaptation.")
    parser.add_argument("-c", required=True, help="config")
    parser.add_argument("-o", required=True, help="output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", action="store_true", help="save model")

    args = parser.parse_args()
    pprint(vars(args))
    main(args)


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
            print(f"  Corruption: {ctype}  (severity={severity})")
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


def run_single(config: dict[str, Any], args) -> dict[str, Any]:
    """Run DANN adaptation + evaluation for a single configuration."""
    # --- fresh model per corruption (reset seed for reproducibility) -------
    fix_seed(args.seed)

    regressor = create_regressor(config).cuda()
    regressor.load_state_dict(torch.load(p := config["regressor"]["source"]))
    print(f"load {p}")

    _, val_ds = get_datasets(config)
    val_dl = DataLoader(val_ds, **config["adapt_dataloader"])

    fe_opt = create_optimizer(regressor, config)

    # --- build DANN engine -------------------------------------------------
    dann_cfg = config["tta"]["config"]
    engine = DANNEngine(
        net=regressor,
        fe_opt=fe_opt,
        train_mode=dann_cfg["train_mode"],
        source_features_file=dann_cfg["source_features_file"],
        disc_config=dann_cfg.get("disc_config", {}),
        lambda_init=dann_cfg.get("lambda_init", 0.0),
        lambda_final=dann_cfg.get("lambda_final", 1.0),
        max_iterations=dann_cfg.get("max_iterations", 1000),
        compile_model=dann_cfg.get("compile_model", None),
    )

    if args.save:
        engine.add_event_handler(
            Events.COMPLETED,
            ModelCheckpoint(args.o, "adapted", require_empty=False),
            {"regressor": regressor})

    # --- offline evaluation after adaptation -------------------------------
    reg_evaluator = RegressionEvaluator(regressor, **config["evaluator"])

    engine.run(val_dl)
    reg_evaluator.run(val_dl)

    metrics = {
        "iteration": engine.state.iteration,
        "online": engine.state.metrics,
        "offline": reg_evaluator.state.metrics,
    }
    pprint(metrics)
    return metrics


def create_optimizer(net: Regressor,
                     config: dict[str, Any]) -> torch.optim.Optimizer:
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


if __name__ == "__main__":
    parse_args()