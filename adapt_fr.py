"""
Feature Restoration (FR) / Bottom-Up Feature Restoration (BUFR).

Implements the bottom-up feature restoration approach from:
    Eastwood et al., "Source-Free Adaptation to Measurement Shift via
    Bottom-Up Feature Restoration", ICLR 2022 (Spotlight).

Usage::

    python adapt_fr.py -c baselines/fr/utkface.yaml -o result/tta_fr/utkface

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
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.modules.batchnorm import _BatchNorm

from utils.seed import fix_seed
from model import create_regressor, Regressor, extract_bn_layers
from dataset import get_datasets
from dataset.corruptions import CORRUPTION_TYPES
from evaluation.evaluator import RegressionEvaluator
from tta.fr_engine import FRStatsCapture, FREngine


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Feature-Restoration test-time adaptation (bottom-up).")
    parser.add_argument("-c", required=True, help="config")
    parser.add_argument("-o", required=True, help="output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", action="store_true", help="save model")

    args = parser.parse_args()
    pprint(vars(args))
    main(args)


# ══════════════════════════════════════════════════════════════════════════
#  Block grouping helpers
# ══════════════════════════════════════════════════════════════════════════

def _collect_bn_layers(model: nn.Module) -> list[_BatchNorm]:
    """Collect all BN layers in forward (module-tree) order."""
    return [m for m in model.modules() if isinstance(m, _BatchNorm)]


def _group_bn_layers_by_block(regressor: Regressor) -> list[list[int]]:
    """Group BN layer indices by ResNet/CNN blocks for bottom-up training.

    For ResNet-50 the structure is:
        feature_extractor.conv1 -> feature_extractor.bn1 -> ...
        feature_extractor.layer1 (contains multiple BN layers)
        feature_extractor.layer2
        feature_extractor.layer3
        feature_extractor.layer4

    We group BN layers per top-level child of the feature extractor.
    For simpler CNNs, each BN is its own group.
    """
    fe = regressor.get_feature_extractor()
    all_bn = _collect_bn_layers(fe)
    bn_id_set = {id(bn) for bn in all_bn}

    # Map each top-level child to the BN layers it contains
    groups: list[list[int]] = []
    bn_to_idx = {id(bn): i for i, bn in enumerate(all_bn)}

    for child in fe.children():
        child_bns = [m for m in child.modules() if id(m) in bn_id_set]
        if child_bns:
            group = [bn_to_idx[id(bn)] for bn in child_bns]
            groups.append(group)

    # Sanity: make sure all BN layers are covered
    covered = set()
    for g in groups:
        covered.update(g)
    assert covered == set(range(len(all_bn))), \
        f"Not all BN layers covered: {covered} vs {set(range(len(all_bn)))}"

    return groups


def _get_bn_params_for_indices(
        bn_layers: list[_BatchNorm],
        indices: list[int]) -> list[nn.Parameter]:
    """Get all trainable parameters from BN layers at given indices."""
    params: list[nn.Parameter] = []
    for i in indices:
        params.extend(bn_layers[i].parameters())
    return params


# ══════════════════════════════════════════════════════════════════════════
#  main / run_single
# ══════════════════════════════════════════════════════════════════════════

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
    """Run FR adaptation + evaluation for a single configuration.

    Bottom-up procedure (BUFR)
    --------------------------
    1. Collect all BN layers and group them by network block.
    2. For each block (from input to output):
       a. Add the block's BN params to the trainable param set
          ("unfreeze" strategy — all earlier blocks remain trainable).
       b. Create a fresh optimiser with LR decayed by block index.
       c. Run ``epochs_per_block`` epochs, minimising KL(Q‖P) over
          all BN layers seen so far (active_layers grows).
    3. Final evaluation with all restoration applied.

    If ``bottom_up`` is ``false``, all BN layers are optimised jointly.
    """
    fix_seed(args.seed)

    # --- load source model ------------------------------------------------
    regressor = create_regressor(config).cuda()
    regressor.load_state_dict(torch.load(p := config["regressor"]["source"]))
    print(f"load {p}")

    _, val_ds = get_datasets(config)
    adapt_dl = DataLoader(val_ds, **config["adapt_dataloader"])

    # --- collect BN layers and create stats capture -----------------------
    bn_layers = _collect_bn_layers(regressor)
    stats_capture = FRStatsCapture(bn_layers).to(torch.device("cuda"))
    print(f"[FR] {stats_capture.n_layers} BN layers found")

    fr_cfg = config["tta"]["config"]
    lr = fr_cfg.get("lr", 1e-3)
    weight_decay = fr_cfg.get("weight_decay", 0.0)
    lr_decay = fr_cfg.get("lr_decay", 5.0)
    bottom_up = fr_cfg.get("bottom_up", True)
    epochs_per_block = fr_cfg.get("epochs_per_block", 1)
    compile_model = fr_cfg.get("compile_model", None)
    train_mode = fr_cfg.get("train_mode", True)

    if bottom_up:
        # ── bottom-up: block by block ─────────────────────────────────────
        groups = _group_bn_layers_by_block(regressor)
        print(f"[FR] {len(groups)} blocks: "
              + ", ".join(f"block{i}({len(g)} BN)" for i, g in enumerate(groups)))

        trainable_params: list[nn.Parameter] = []
        active_layers: list[int] = []

        for block_idx, group in enumerate(groups):
            # Unfreeze: add this block's BN params
            block_params = _get_bn_params_for_indices(bn_layers, group)
            trainable_params.extend(block_params)
            active_layers.extend(group)

            # LR decays per block (matching paper)
            block_lr = lr / (lr_decay ** block_idx)

            opt = torch.optim.Adam(
                trainable_params, lr=block_lr, weight_decay=weight_decay)

            engine = FREngine(
                net=regressor,
                stats_capture=stats_capture,
                opt=opt,
                active_layers=list(active_layers),
                train_mode=train_mode,
                compile_model=compile_model,
            )
            engine.run(adapt_dl, max_epochs=epochs_per_block)

            print(f"[FR] block {block_idx}/{len(groups)-1} done  "
                  f"(lr={block_lr:.6f}, active_layers={len(active_layers)}, "
                  f"rmse={engine.state.metrics.get('rmse_loss', '?'):.4f})")
    else:
        # ── joint: all BN layers at once ──────────────────────────────────
        all_bn_params = list(itertools.chain.from_iterable(
            bn.parameters() for bn in bn_layers))

        opt = torch.optim.Adam(
            all_bn_params, lr=lr, weight_decay=weight_decay)

        engine = FREngine(
            net=regressor,
            stats_capture=stats_capture,
            opt=opt,
            active_layers=list(range(stats_capture.n_layers)),
            train_mode=train_mode,
            compile_model=compile_model,
        )
        engine.run(adapt_dl, max_epochs=epochs_per_block)

    # ── online metrics from the last engine run ───────────────────────────
    online_metrics = dict(engine.state.metrics)

    # ── offline evaluation ────────────────────────────────────────────────
    # Remove hooks before final eval (BN will use updated running stats)
    stats_capture.remove_hooks()

    val_dl = DataLoader(val_ds, **config["val_dataloader"])
    reg_evaluator = RegressionEvaluator(regressor, **config["evaluator"])
    reg_evaluator.run(val_dl)

    metrics = {
        "iteration": engine.state.iteration,
        "online": online_metrics,
        "offline": reg_evaluator.state.metrics,
    }
    pprint(metrics)
    return metrics


if __name__ == "__main__":
    parse_args()