"""
BNA engine – Batch Normalization Adaptation baseline.

Implements the BN-adapt strategy from
    Schneider et al., "Revisiting Batch Normalization for Improving
    Corruption Robustness", WACV 2021.

Core idea
---------
At test time the BatchNorm running statistics (running_mean / running_var)
accumulated during source training are *no longer representative* of the
target distribution.  BNA fixes this by:

1. **Resetting** the running statistics of every BN layer
   (running_mean ← 0, running_var ← 1, num_batches_tracked ← 0).
2. **Forwarding** the target data through the network in **train mode**
   so that BN layers recompute statistics from the target batches.
3. Optionally setting ``momentum = None`` so PyTorch uses a simple
   cumulative moving average (1/num_batches_tracked) instead of the
   default exponential moving average – giving an exact mean over the
   whole test set.

No model weights (conv / linear / γ / β) are modified; only the BN
buffers change.
"""
from dataclasses import dataclass, InitVar

import torch
from torch import nn, Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from ignite.engine import Engine
from ignite.metrics import RootMeanSquaredError, MeanAbsoluteError
from ignite.contrib.metrics.regression.r2_score import R2Score

from evaluation.metrics import ModelDistanceMetric, PearsonCorrelation
from model import Regressor


# ── helpers ───────────────────────────────────────────────────────────────

def _get_bn_layers(model: nn.Module) -> list[_BatchNorm]:
    """Collect all BatchNorm layers (1d / 2d / 3d) in *model*."""
    return [m for m in model.modules() if isinstance(m, _BatchNorm)]


def _reset_bn_stats(bn_layers: list[_BatchNorm]) -> None:
    """Reset running_mean, running_var, and num_batches_tracked to defaults."""
    for bn in bn_layers:
        bn.reset_running_stats()          # mean←0, var←1, tracked←0


def _set_bn_momentum(bn_layers: list[_BatchNorm],
                     momentum: float | None) -> list[float | None]:
    """Override momentum for every BN layer; return original values."""
    orig = [bn.momentum for bn in bn_layers]
    for bn in bn_layers:
        bn.momentum = momentum
    return orig


# ── engine ────────────────────────────────────────────────────────────────

@dataclass
class BNAEngine(Engine):
    net: Regressor
    train_mode: bool
    reset_stats: InitVar[bool] = True
    bn_momentum: InitVar[float | None] = None
    compile_model: InitVar[dict | None] = None

    def __post_init__(self,
                      reset_stats: bool,
                      bn_momentum: float | None,
                      compile_model: dict | None):
        super().__init__(self.update)

        # ── metrics (same interface as other engines) ─────────────────────
        y_ot = lambda d: (d["y_pred"], d["y"])
        RootMeanSquaredError(y_ot).attach(self, "rmse_loss")
        MeanAbsoluteError(y_ot).attach(self, "mae_loss")
        R2Score(y_ot).attach(self, "R2")
        PearsonCorrelation(y_ot).attach(self, "r")
        ModelDistanceMetric(self.net).attach(self, "model_dist")

        # ── BN preparation ────────────────────────────────────────────────
        bn_layers = _get_bn_layers(self.net)
        print(f"[BNA] found {len(bn_layers)} BatchNorm layers")

        if reset_stats:
            _reset_bn_stats(bn_layers)
            print("[BNA] reset running stats  (mean←0, var←1)")

        # momentum=None → cumulative moving average (paper default)
        self._orig_momentum = _set_bn_momentum(bn_layers, bn_momentum)
        print(f"[BNA] BN momentum set to {bn_momentum}")

        # ── optional torch.compile ────────────────────────────────────────
        if compile_model is None:
            self.feature_extractor = self.net.feature
        else:
            try:
                self.feature_extractor = torch.compile(
                    self.net.feature, **compile_model)
            except RuntimeError as e:
                print(f"torch.compile failed: {e}")
                self.feature_extractor = self.net.feature

    @torch.no_grad()
    def update(self, engine: Engine,
               batch: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
        """Forward one batch; BN layers update running stats in train mode."""
        if self.train_mode:
            self.net.train()
        else:
            self.net.eval()

        x, y = batch
        x = x.cuda()

        feature = self.feature_extractor(x)
        y_pred = self.net.predict_from_feature(feature)

        return {
            "y_pred": y_pred,
            "y": y.cuda().float().flatten(),
            "feat_raw": feature,
        }
