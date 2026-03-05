"""
Test-Time Training (TTT) engine.

Implements the self-supervised rotation prediction approach from:
  "Test-time training with self-supervision for generalization under
   distribution shifts" (Sun et al., ICML 2020).

At test time, each input image is rotated by {0°, 90°, 180°, 270°}.
A rotation classifier head (on top of the shared feature extractor)
is trained to predict which rotation was applied.  The feature extractor
(BN layers only, typically) is updated alongside the rotation head.

Because the source model was *not* jointly trained with the rotation head,
we initialise the head from scratch and train it on-the-fly during
adaptation — a common practical variant used in many TTA benchmarks.
"""
from dataclasses import dataclass, InitVar

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from ignite.engine import Engine
from ignite.metrics import RootMeanSquaredError, MeanAbsoluteError
from ignite.contrib.metrics.regression.r2_score import R2Score

from evaluation.metrics import ModelDistanceMetric, PearsonCorrelation
from model import Regressor


class RotationHead(nn.Module):
    """Simple linear or MLP head for 4-way rotation classification."""

    def __init__(self, feature_dim: int, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            self.head = nn.Linear(feature_dim, 4)
        else:
            self.head = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 4),
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)


def rotate_batch(x: Tensor) -> tuple[Tensor, Tensor]:
    """Create 4 rotated copies of each image in the batch.

    Parameters
    ----------
    x : Tensor
        Input images of shape ``(B, C, H, W)``.

    Returns
    -------
    x_rot : Tensor
        Rotated images, shape ``(4*B, C, H, W)``.
        Order: [original batch, 90° batch, 180° batch, 270° batch].
    labels : Tensor
        Rotation labels ``{0, 1, 2, 3}`` of shape ``(4*B,)``.
    """
    B = x.shape[0]
    x0 = x                                         # 0°
    x1 = torch.rot90(x, k=1, dims=[2, 3])          # 90°
    x2 = torch.rot90(x, k=2, dims=[2, 3])          # 180°
    x3 = torch.rot90(x, k=3, dims=[2, 3])          # 270°

    x_rot = torch.cat([x0, x1, x2, x3], dim=0)     # (4B, C, H, W)
    labels = torch.cat([
        torch.full((B,), 0, device=x.device, dtype=torch.long),
        torch.full((B,), 1, device=x.device, dtype=torch.long),
        torch.full((B,), 2, device=x.device, dtype=torch.long),
        torch.full((B,), 3, device=x.device, dtype=torch.long),
    ])
    return x_rot, labels


@dataclass
class TTTEngine(Engine):
    net: Regressor
    fe_opt: torch.optim.Optimizer        # optimiser for feature extractor (BN)
    train_mode: bool
    rot_config: InitVar[dict]            # hidden_dim, lr, etc.
    compile_model: InitVar[dict | None]

    # ------------------------------------------------------------------ init
    def __post_init__(self,
                      rot_config: dict,
                      compile_model: dict | None):
        super().__init__(self.update)

        # --- metrics (same interface as other engines) ---------------------
        y_ot = lambda d: (d["y_pred"], d["y"])
        RootMeanSquaredError(y_ot).attach(self, "rmse_loss")
        MeanAbsoluteError(y_ot).attach(self, "mae_loss")
        R2Score(y_ot).attach(self, "R2")
        PearsonCorrelation(y_ot).attach(self, "r")
        ModelDistanceMetric(self.net).attach(self, "model_dist")

        # --- rotation prediction head --------------------------------------
        feat_dim = self.net.regressor.in_features
        hidden_dim = rot_config.get("hidden_dim", None)
        self.rotation_head = RotationHead(feat_dim, hidden_dim).cuda()

        rot_lr = rot_config.get("lr", 1e-3)
        rot_wd = rot_config.get("weight_decay", 0.0)
        self.rot_opt = torch.optim.Adam(
            self.rotation_head.parameters(), lr=rot_lr, weight_decay=rot_wd)

        print(f"TTT rotation head: feat_dim={feat_dim}, "
              f"hidden_dim={hidden_dim}, lr={rot_lr}")

        # --- optional torch.compile ----------------------------------------
        self.feature_extractor = self.net.feature
        if compile_model is not None:
            try:
                self.feature_extractor = torch.compile(
                    self.net.feature, **compile_model)
            except RuntimeError as e:
                print(f"torch.compile failed: {e}")

    # --------------------------------------------------------------- update
    def update(self, engine: Engine,
               batch: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
        if self.train_mode:
            self.net.train()
        else:
            self.net.eval()
        self.rotation_head.train()

        x, y = batch
        x = x.cuda()

        # --- 1) rotation prediction loss -----------------------------------
        x_rot, rot_labels = rotate_batch(x)          # (4B, C, H, W), (4B,)

        self.fe_opt.zero_grad()
        self.rot_opt.zero_grad()

        rot_features = self.feature_extractor(x_rot)  # (4B, D)
        rot_logits = self.rotation_head(rot_features)  # (4B, 4)

        rot_loss = F.cross_entropy(rot_logits, rot_labels)

        rot_loss.backward()
        self.fe_opt.step()
        self.rot_opt.step()

        # --- 2) regression prediction (no grad, just for metrics) ----------
        with torch.no_grad():
            self.net.eval()
            feature = self.feature_extractor(x)
            y_pred = self.net.predict_from_feature(feature)

        rot_acc = (rot_logits.argmax(dim=1) == rot_labels).float().mean()

        return {
            "y_pred": y_pred,
            "y": y.cuda().float().flatten(),
            "rot_loss": float(rot_loss.item()),
            "rot_acc": float(rot_acc.item()),
        }