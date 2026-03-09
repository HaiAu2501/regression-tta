"""
Activation Matching (AM) Test-Time Adaptation engine.

Implements ActMAD (Mirza et al., CVPR 2023) for scalar regression TTA:
align per-location activation statistics at multiple BN layers to
pre-computed source training statistics.

Loss per layer:
    L_l(B; θ) = |μ_l(B; θ) - μ̂_l| + |σ²_l(B; θ) - σ̂²_l|

Total loss:
    L(B; θ) = (1/|L|) Σ_{l∈L} L_l(B; θ)

Offline prerequisite:
    python compute_act_stats.py -c configs/act_stats/<dataset>.yaml \
        -o result/source/<dataset>
"""
from dataclasses import dataclass, InitVar

import torch
from torch import Tensor
from ignite.engine import Engine
from ignite.metrics import RootMeanSquaredError, MeanAbsoluteError
from ignite.contrib.metrics.regression.r2_score import R2Score

from evaluation.metrics import ModelDistanceMetric, PearsonCorrelation
from model import Regressor


def _select_layers(all_names: list[str],
                   max_layers: int | None) -> list[str]:
    """Select at most *max_layers* names, evenly spaced from *all_names*.

    If *max_layers* is ``None`` or >= len(all_names), return all names.
    """
    if max_layers is None or max_layers >= len(all_names):
        return all_names

    n = len(all_names)
    if max_layers <= 1:
        return [all_names[-1]]
    if max_layers == 2:
        return [all_names[0], all_names[-1]]

    step = (n - 1) / (max_layers - 1)
    indices = [round(i * step) for i in range(max_layers)]
    # deduplicate while keeping order
    seen = set()
    selected = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            selected.append(all_names[idx])
    return selected


@dataclass
class AMEngine(Engine):
    net: Regressor
    opt: torch.optim.Optimizer
    train_mode: bool
    act_stats_file: InitVar[str]
    max_layers: InitVar[int | None]
    compile_model: InitVar[dict | None]

    def __post_init__(self,
                      act_stats_file: str,
                      max_layers: int | None,
                      compile_model: dict | None):
        super().__init__(self.update)

        # --- metrics (same interface as other engines) ---------------------
        y_ot = lambda d: (d["y_pred"], d["y"])
        RootMeanSquaredError(y_ot).attach(self, "rmse_loss")
        MeanAbsoluteError(y_ot).attach(self, "mae_loss")
        R2Score(y_ot).attach(self, "R2")
        PearsonCorrelation(y_ot).attach(self, "r")
        ModelDistanceMetric(self.net).attach(self, "model_dist")

        # --- load precomputed activation statistics ------------------------
        stat_dict = torch.load(act_stats_file)
        all_layer_names: list[str] = stat_dict["layer_names"]

        # --- select subset of layers (paper: "evenly across the network") --
        self.layer_names = _select_layers(all_layer_names, max_layers)
        print(f"AM: {len(self.layer_names)}/{len(all_layer_names)} layers "
              f"selected from {act_stats_file!r}")

        self.src_means: dict[str, Tensor] = {}
        self.src_vars: dict[str, Tensor] = {}
        for name in self.layer_names:
            self.src_means[name] = stat_dict["stats"][name]["mean"].cuda()
            self.src_vars[name] = stat_dict["stats"][name]["var"].cuda()
            print(f"  {name}: shape={self.src_means[name].shape}")

        # --- register forward hooks ----------------------------------------
        # .clone() in the hook: ResNet uses ReLU(inplace=True) right after
        # BN.  Without clone the hooked tensor is silently overwritten to
        # post-ReLU values, zeroing gradient wherever activation < 0.
        # Cloning captures the true post-BN / pre-ReLU tensor.
        self._activations: dict[str, Tensor] = {}
        self._hook_handles = []

        named_modules = dict(self.net.named_modules())
        for name in self.layer_names:
            module = named_modules[name]
            h = module.register_forward_hook(self._make_hook(name))
            self._hook_handles.append(h)

        # --- optional torch.compile ----------------------------------------
        self.feature_extractor = self.net.feature
        if compile_model is not None:
            try:
                self.feature_extractor = torch.compile(
                    self.net.feature, **compile_model)
            except RuntimeError as e:
                print(f"torch.compile failed: {e}")

    def _make_hook(self, name: str):
        """Return a forward hook that clones & stores the layer output."""
        def hook(module, input, output):
            self._activations[name] = output.clone()
        return hook

    # --------------------------------------------------------------- update
    def update(self, engine: Engine,
               batch: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
        if self.train_mode:
            self.net.train()
        else:
            self.net.eval()
        self.net.zero_grad()

        x, y = batch
        x = x.cuda()

        # --- forward pass (hooks capture intermediate activations) ---------
        feature = self.feature_extractor(x)
        y_pred = self.net.predict_from_feature(feature)

        # --- activation matching loss --------------------------------------
        # Per-layer: mean L1 over all spatial locations,
        # then average across selected layers.
        loss = torch.tensor(0.0, device="cuda")
        n_layers = len(self.layer_names)

        for name in self.layer_names:
            act = self._activations[name]       # (B, C, ...) with grad
            batch_mean = act.mean(dim=0)         # (C, H, W) or (C,)
            batch_var = act.var(dim=0, unbiased=False)

            loss = loss \
                + (batch_mean - self.src_means[name]).abs().mean() \
                + (batch_var - self.src_vars[name]).abs().mean()

        loss = loss / n_layers

        loss.backward()
        self.opt.step()

        self._activations.clear()

        return {
            "y_pred": y_pred,
            "y": y.cuda().float().flatten(),
        }