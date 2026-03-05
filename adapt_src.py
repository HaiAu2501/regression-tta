"""
Baseline evaluation: run the **source model** (no TTA / SSA) on corrupted
validation data.  When ``corruption_type`` is ``"all"``, every corruption in
:data:`CORRUPTION_TYPES` is evaluated and the results are saved to a CSV.
"""
from typing import Any
from pprint import pprint
import json
import csv
import copy
from pathlib import Path

import yaml

import torch
from torch.utils.data import DataLoader

from utils.seed import fix_seed
from model import create_regressor
from dataset import get_datasets
from dataset.corruptions import CORRUPTION_TYPES
from evaluation.evaluator import RegressionEvaluator


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate source model (no adaptation) on corrupted data.")
    parser.add_argument("-c", required=True, help="config")
    parser.add_argument("-o", required=True, help="output directory")
    parser.add_argument("--seed", type=int, default=42)

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
            for k, v in metrics["metrics"].items():
                row[k] = v
            all_rows.append(row)

        # --- write CSV ---
        csv_path = Path(args.o, "metrics.csv")
        fieldnames = list(all_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nSaved all-corruption results to {csv_path}")

        # also save as json for convenience
        with Path(args.o, "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=4, ensure_ascii=False)
    else:
        metrics = run_single(config, args)
        with Path(args.o, "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)


def run_single(config: dict[str, Any], args) -> dict[str, Any]:
    """Load source model and evaluate (no adaptation) on corrupted val data."""
    regressor = create_regressor(config).cuda()
    regressor.load_state_dict(torch.load(p := config["regressor"]["source"]))
    print(f"load {p}")

    _, val_ds = get_datasets(config)
    val_dl = DataLoader(val_ds, **config["val_dataloader"])

    evaluator = RegressionEvaluator(regressor, **config["evaluator"])
    evaluator.run(val_dl)

    metrics = {"metrics": dict(evaluator.state.metrics)}
    pprint(metrics)
    return metrics


if __name__ == "__main__":
    parse_args()
