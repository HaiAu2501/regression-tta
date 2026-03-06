"""
Launcher for BatchNorm Adaptation (BNA) baseline.

Usage example:

    python adapt_bna.py -c configs/tta/utkface.yaml -o result/tta_bna/utkface

This script follows the same conventions as `adapt_ttt.py` but uses the
BNA strategy: run through the validation/target loader with the network in
train mode (so BN updates running stats) while keeping gradients disabled.
After adaptation we run the standard `RegressionEvaluator` to compute
metrics.
"""
from typing import Any
from pprint import pprint
import json
from pathlib import Path
import copy

import yaml
import torch
from torch.utils.data import DataLoader
from ignite.engine import Events

from utils.seed import fix_seed
from model import create_regressor
from dataset import get_datasets
from dataset.corruptions import CORRUPTION_TYPES
from evaluation.evaluator import RegressionEvaluator
from tta.bna_engine import BNAEngine


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="BatchNorm Adaptation (BNA) test-time baseline.")
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

        # --- write CSV/JSON (keep same behaviour as other adapt scripts) ---
        import csv
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
    """Run BNA adaptation + evaluation for a single configuration."""
    fix_seed(args.seed)

    regressor = create_regressor(config).cuda()
    regressor.load_state_dict(torch.load(p := config["regressor"]["source"]))
    print(f"load {p}")

    _, val_ds = get_datasets(config)
    val_dl = DataLoader(val_ds, **config["adapt_dataloader"]) 

    # --- build BNA engine -------------------------------------------------
    bna_cfg = config["tta"]["config"]
    engine = BNAEngine(
        net=regressor,
        train_mode=bna_cfg.get("train_mode", True),
        reset_stats=bna_cfg.get("reset_stats", True),
        bn_momentum=bna_cfg.get("bn_momentum", None),
        compile_model=bna_cfg.get("compile_model", None),
    )

    if args.save:
        from ignite.handlers import ModelCheckpoint
        from ignite.engine import Events
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


if __name__ == "__main__":
    parse_args()
