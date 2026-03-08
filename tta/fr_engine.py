"""
Feature Restoration (FR) Test-Time Adaptation engine.

Implements the bottom-up feature restoration approach from:
    Eastwood et al., "Source-Free Adaptation to Measurement Shift via
    Bottom-Up Feature Restoration", ICLR 2022 (Spotlight).

Core idea
---------
FR stores a lightweight Gaussian approximation of the feature
distribution at each BN layer (running_mean, running_var saved during
source training).  At test time, the feature extractor is adapted so
that the batch feature distribution at each BN layer re-aligns with
the saved source distribution.

Loss per BN layer *l* (per-unit univariate KL divergence, default KL(Q‖P)):

    KL(Q‖P)_j = ½ log(σ²_P,j / σ²_Q,j)
              + (σ²_Q,j + (μ_Q,j − μ_P,j)²) / (2 σ²_P,j)
              − ½

where P = source (BN running stats) and Q = target batch, computed
independently for each channel *j*.  The total loss is the mean of
per-unit KL divergences across all active BN layers.

Bottom-up optimisation (BUFR)
-----------------------------
Layers are optimised block-by-block from early to deep.  After each
block, its parameters remain trainable ("unfreeze" strategy) so all
earlier blocks continue to be refined.  LR is decayed per block
(``lr_decay`` parameter).

The model is kept in **train mode** so BN layers use batch statistics
for normalisation.  Trainable parameters are the BN affine weights
(γ, β) — consistent with the ``fe_bn`` optimizer strategy used by other
TTA methods in this codebase.
"""
from __future__ import annotations

from dataclasses import dataclass, InitVar
from collections.abc import Sequence

import torch
from torch import nn, Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from ignite.engine import Engine
from ignite.metrics import RootMeanSquaredError, MeanAbsoluteError
from ignite.contrib.metrics.regression.r2_score import R2Score

from evaluation.metrics import ModelDistanceMetric, PearsonCorrelation
from model import Regressor


# ══════════════════════════════════════════════════════════════════════════
#  Per-unit KL divergence helpers
# ══════════════════════════════════════════════════════════════════════════

def kl_q_p(q_mean: Tensor, q_var: Tensor,
           p_mean: Tensor, p_var: Tensor,
           eps: float = 1e-8) -> Tensor:
    """Per-unit KL(Q‖P) for univariate Gaussians.

    Parameters
    ----------
    q_mean, q_var : Tensor, shape (C,)
        Target batch statistics (mean, variance) per channel.
    p_mean, p_var : Tensor, shape (C,)
        Source statistics (mean, variance) per channel.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    Tensor, shape (C,)
        KL divergence per channel.
    """
    p_var_safe = p_var + eps
    q_var_safe = q_var + eps
    return (0.5 * torch.log(p_var_safe / q_var_safe)
            + (q_var_safe + (q_mean - p_mean).square()) / (2.0 * p_var_safe)
            - 0.5)


# ══════════════════════════════════════════════════════════════════════════
#  FRStatsCapture – captures batch stats at each BN layer via hooks
# ══════════════════════════════════════════════════════════════════════════

class FRStatsCapture:
    """Captures batch mean/var at each BN layer and computes KL(Q‖P) loss.

    Source statistics are the BN ``running_mean`` / ``running_var`` saved
    during source training (snapshot taken at init time).
    """

    def __init__(self, bn_layers: list[_BatchNorm], eps: float = 1e-8):
        self.n_layers = len(bn_layers)
        self.eps = eps

        # Snapshot source BN running stats (detached, on same device)
        self._src_means: list[Tensor] = []
        self._src_vars: list[Tensor] = []
        for bn in bn_layers:
            self._src_means.append(bn.running_mean.detach().clone())
            self._src_vars.append(bn.running_var.detach().clone())

        # Per-batch captured stats
        self._batch_means: list[Tensor] = []
        self._batch_vars: list[Tensor] = []
        self._capturing = False

        # Register forward pre-hooks on each BN layer
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        for i, bn in enumerate(bn_layers):
            h = bn.register_forward_pre_hook(self._make_hook(i))
            self._hooks.append(h)

    def _make_hook(self, idx: int):
        """Hook that captures batch mean/var of the input to BN layer."""
        def hook(module: nn.Module, args: tuple[Tensor, ...]):
            if not self._capturing:
                return
            x = args[0]
            # Reduce over batch and spatial dims, keep channel dim
            reduce_dims = [0] + list(range(2, x.ndim))
            self._batch_means.append(x.mean(reduce_dims))
            self._batch_vars.append(x.var(reduce_dims, unbiased=False))
        return hook

    def start_capture(self) -> None:
        self._capturing = True
        self._batch_means = []
        self._batch_vars = []

    def stop_capture(self) -> None:
        self._capturing = False

    def compute_loss(self,
                     active_layers: Sequence[int] | None = None) -> Tensor:
        """Compute mean of per-unit KL(Q‖P) over active BN layers.

        Must be called after a forward pass with capturing enabled.
        """
        if active_layers is None:
            active_layers = range(self.n_layers)

        total_units = 0
        loss = torch.tensor(0.0, device=self._batch_means[0].device)
        for i in active_layers:
            kl = kl_q_p(
                q_mean=self._batch_means[i],
                q_var=self._batch_vars[i],
                p_mean=self._src_means[i],
                p_var=self._src_vars[i],
                eps=self.eps,
            )
            loss = loss + kl.sum()
            total_units += kl.numel()

        # Average over all units (consistent with paper: 1/(2*n_units) scaling)
        return loss / max(total_units, 1)

    def to(self, device: torch.device) -> "FRStatsCapture":
        """Move source stats to device."""
        self._src_means = [m.to(device) for m in self._src_means]
        self._src_vars = [v.to(device) for v in self._src_vars]
        return self

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.stop_capture()


# ══════════════════════════════════════════════════════════════════════════
#  FREngine – ignite engine for one optimisation stage
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class FREngine(Engine):
    """Feature Restoration engine (one stage of bottom-up optimisation).

    Parameters
    ----------
    net : Regressor
        The source-trained model.
    stats_capture : FRStatsCapture
        Manages hooks and KL loss computation.
    opt : torch.optim.Optimizer
        Optimiser over the active trainable parameters.
    active_layers : list[int]
        Which BN layers' KL losses to include.
    train_mode : bool
        If True, model runs in train mode so BN uses batch stats.
    compile_model : dict | None
        ``torch.compile`` kwargs.
    """
    net: Regressor
    stats_capture: FRStatsCapture
    opt: torch.optim.Optimizer
    active_layers: list[int]
    train_mode: bool = True
    compile_model: InitVar[dict | None] = None

    def __post_init__(self, compile_model: dict | None):
        super().__init__(self.update)

        # --- metrics ---
        y_ot = lambda d: (d["y_pred"], d["y"])
        RootMeanSquaredError(y_ot).attach(self, "rmse_loss")
        MeanAbsoluteError(y_ot).attach(self, "mae_loss")
        R2Score(y_ot).attach(self, "R2")
        PearsonCorrelation(y_ot).attach(self, "r")
        ModelDistanceMetric(self.net).attach(self, "model_dist")

        if compile_model is None:
            self.feature_extractor = self.net.feature
        else:
            try:
                self.feature_extractor = torch.compile(
                    self.net.feature, **compile_model)
            except RuntimeError as e:
                print(f"torch.compile failed: {e}")
                self.feature_extractor = self.net.feature

    def update(self, engine: Engine,
               batch: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
        if self.train_mode:
            self.net.train()
        else:
            self.net.eval()

        x, y = batch
        x = x.cuda()

        self.opt.zero_grad()
        self.stats_capture.start_capture()

        feature = self.feature_extractor(x)
        y_pred = self.net.predict_from_feature(feature)

        self.stats_capture.stop_capture()

        loss = self.stats_capture.compute_loss(
            active_layers=self.active_layers)

        loss.backward()
        self.opt.step()

        return {
            "y_pred": y_pred,
            "y": y.cuda().float().flatten(),
            "fr_loss": float(loss.item()),
        }