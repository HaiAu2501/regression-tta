"""
Compute per-layer, location-aware activation statistics for AM.

For each BatchNorm layer in the source model, computes the mean and
variance of activations at every spatial location over the training set.

Usage::

    python compute_act_stats.py -c configs/act_stats/utkface.yaml \
        -o result/source/utkface

    python compute_act_stats.py -c configs/act_stats/svhn.yaml \
        -o result/source/svhn

Saves ``act_stats.pt`` containing::

    {
        "layer_names": [str, ...],
        "stats": {
            layer_name: {"mean": Tensor, "var": Tensor},
            ...
        },
        "count": int,
    }
"""
from pprint import pprint
import json
from pathlib import Path

import torch
from torch import nn, Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader

import yaml

from dataset import get_datasets
from model import create_regressor


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute activation statistics for AM (ActMAD).")
    parser.add_argument("-c", required=True, help="config file")
    parser.add_argument("-o", required=True, help="output directory")

    args = parser.parse_args()
    pprint(vars(args))
    main(args)


def get_bn_layer_names(model: nn.Module) -> list[str]:
    """Return the fully-qualified names of all BatchNorm layers."""
    return [
        name for name, mod in model.named_modules()
        if isinstance(mod, _BatchNorm)
    ]


def main(args):
    Path(args.o).mkdir(parents=True, exist_ok=True)

    with open(args.c, "r", encoding="utf-8") as f:
        config = json.load(f) if args.c.endswith(".json") else yaml.safe_load(f)
    config["dataset"]["train_aug"] = False
    pprint(config)

    # --- dataset (source / training split) ---------------------------------
    ds = get_datasets(config)[0]
    dl = DataLoader(ds, **config["dataloader"])

    # --- model -------------------------------------------------------------
    regressor = create_regressor(config).cuda()
    regressor.load_state_dict(torch.load(p := config["regressor"]["source"]))
    print(f"load regressor: {p}")
    # MUST use train mode: BN uses batch stats, matching online adaptation.
    # eval mode would use frozen running stats, causing mismatch when online
    # adaptation updates conv weights while running stats stay frozen.
    regressor.train()

    # --- discover BN layers & register hooks -------------------------------
    bn_names = get_bn_layer_names(regressor)
    print(f"Found {len(bn_names)} BN layers:")
    for n in bn_names:
        print(f"  {n}")

    activations: dict[str, Tensor] = {}
    handles = []

    def _make_hook(name: str):
        def hook(module, inp, out):
            # .clone() protects from inplace ReLU that follows BN in ResNet
            activations[name] = out.detach().clone()
        return hook

    named_mods = dict(regressor.named_modules())
    for name in bn_names:
        handles.append(named_mods[name].register_forward_hook(_make_hook(name)))

    # --- accumulate sum and sum-of-squares ---------------------------------
    sums: dict[str, Tensor] = {}
    sum_sqs: dict[str, Tensor] = {}
    count = 0

    print("computing activation statistics …", flush=True)

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(dl):
            x = x.cuda()
            regressor(x)                       # triggers hooks

            B = x.shape[0]
            count += B

            for name in bn_names:
                act = activations[name].to(torch.float64)  # (B, C, ...)
                if name not in sums:
                    sums[name] = torch.zeros(
                        act.shape[1:], device="cuda", dtype=torch.float64)
                    sum_sqs[name] = torch.zeros_like(sums[name])

                sums[name].add_(act.sum(dim=0))
                sum_sqs[name].add_(act.square().sum(dim=0))

            activations.clear()

            if (batch_idx + 1) % 50 == 0:
                print(f"  batch {batch_idx + 1}/{len(dl)}", flush=True)

    # --- remove hooks ------------------------------------------------------
    for h in handles:
        h.remove()

    # --- compute mean and variance -----------------------------------------
    stats: dict[str, dict[str, Tensor]] = {}
    for name in bn_names:
        mean_d = sums[name] / count                      # (C,...) float64, cuda
        var_d = sum_sqs[name] / count - mean_d.square()   # float64, cuda
        mean = mean_d.float().cpu()
        var = var_d.float().cpu()
        var.clamp_(min=0.0)                    # numerical safety
        stats[name] = {"mean": mean, "var": var}
        print(f"  {name}: shape={tuple(mean.shape)}")

    save_path = Path(args.o, "act_stats.pt")
    torch.save({
        "layer_names": bn_names,
        "stats": stats,
        "count": count,
    }, str(save_path))
    print(f"Saved activation stats ({count} samples) → {save_path}")


if __name__ == "__main__":
    parse_args()