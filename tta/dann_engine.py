"""
DANN-based Test-Time Adaptation engine.

Since source data is unavailable at test time, we load pre-extracted source
features (``raw_features.pt``) and train a domain discriminator to align
the target feature distribution to the source distribution via gradient
reversal.

The feature extractor (BN layers only, typically) is updated so that its
outputs fool the discriminator, while the discriminator is trained to
distinguish source vs. target features.
"""
from dataclasses import dataclass, InitVar

import torch
from torch import nn, Tensor
from ignite.engine import Engine
from ignite.metrics import RootMeanSquaredError, MeanAbsoluteError
from ignite.contrib.metrics.regression.r2_score import R2Score

from evaluation.metrics import ModelDistanceMetric, PearsonCorrelation
from model import Regressor
from .gradient_reversal import gradient_reversal


class DomainDiscriminator(nn.Module):
    """Simple MLP discriminator: feature_dim -> 1 (domain logit)."""

    def __init__(self, feature_dim: int, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class DANNEngine(Engine):
    net: Regressor
    fe_opt: torch.optim.Optimizer        # optimiser for feature extractor (BN)
    train_mode: bool
    source_features_file: InitVar[str]   # path to raw_features.pt
    disc_config: InitVar[dict]           # hidden_dim, lr, etc.
    lambda_init: InitVar[float]          # GRL lambda start
    lambda_final: InitVar[float]         # GRL lambda end
    max_iterations: InitVar[int]         # for lambda schedule
    compile_model: InitVar[dict | None]

    # ------------------------------------------------------------------ init
    def __post_init__(self,
                      source_features_file: str,
                      disc_config: dict,
                      lambda_init: float,
                      lambda_final: float,
                      max_iterations: int,
                      compile_model: dict | None):
        super().__init__(self.update)

        # --- metrics (same as TTAEngine for comparable logging) ------------
        y_ot = lambda d: (d["y_pred"], d["y"])
        RootMeanSquaredError(y_ot).attach(self, "rmse_loss")
        MeanAbsoluteError(y_ot).attach(self, "mae_loss")
        R2Score(y_ot).attach(self, "R2")
        PearsonCorrelation(y_ot).attach(self, "r")
        ModelDistanceMetric(self.net).attach(self, "model_dist")

        # --- source features -----------------------------------------------
        self.source_features: Tensor = torch.load(source_features_file).cuda()
        print(f"Loaded source features: {self.source_features.shape} "
              f"from {source_features_file!r}")
        self.n_source = self.source_features.shape[0]

        # --- domain discriminator ------------------------------------------
        feat_dim = self.source_features.shape[1]
        hidden_dim = disc_config.get("hidden_dim", 1024)
        self.discriminator = DomainDiscriminator(feat_dim, hidden_dim).cuda()

        disc_lr = disc_config.get("lr", 1e-3)
        disc_wd = disc_config.get("weight_decay", 0.0)
        self.disc_opt = torch.optim.Adam(
            self.discriminator.parameters(), lr=disc_lr, weight_decay=disc_wd)

        self.bce = nn.BCEWithLogitsLoss()

        # --- GRL lambda schedule -------------------------------------------
        self.lambda_init = lambda_init
        self.lambda_final = lambda_final
        self.max_iterations = max(max_iterations, 1)

        # --- optional torch.compile ----------------------------------------
        self.feature_extractor = self.net.feature
        if compile_model is not None:
            try:
                self.feature_extractor = torch.compile(
                    self.net.feature, **compile_model)
            except RuntimeError as e:
                print(f"torch.compile failed: {e}")

    # ---------------------------------------------------------------- helpers
    def _get_lambda(self) -> float:
        """Linear schedule from lambda_init to lambda_final."""
        progress = min(self.state.iteration / self.max_iterations, 1.0)
        return self.lambda_init + (self.lambda_final - self.lambda_init) * progress

    def _sample_source(self, n: int) -> Tensor:
        """Randomly sample *n* source features (with replacement if needed)."""
        idx = torch.randint(0, self.n_source, (n,))
        return self.source_features[idx]

    # --------------------------------------------------------------- update
    def update(self, engine: Engine,
               batch: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
        if self.train_mode:
            self.net.train()
        else:
            self.net.eval()

        x, y = batch
        x = x.cuda()
        batch_size = x.shape[0]
        lambda_ = self._get_lambda()

        # --- extract target features ---------------------------------------
        target_feat = self.feature_extractor(x)          # (B, D)
        y_pred = self.net.predict_from_feature(target_feat)

        # --- sample source features (detached — no grad to source) ---------
        source_feat = self._sample_source(batch_size)    # (B, D)

        # --- domain labels: source=0, target=1 ----------------------------
        source_label = torch.zeros(batch_size, device="cuda")
        target_label = torch.ones(batch_size, device="cuda")

        # ====== Step 1: update discriminator ===============================
        self.disc_opt.zero_grad()

        # discriminator sees raw features (no GRL)
        d_source = self.discriminator(source_feat.detach())
        d_target = self.discriminator(target_feat.detach())

        disc_loss = (self.bce(d_source, source_label)
                     + self.bce(d_target, target_label)) / 2
        disc_loss.backward()
        self.disc_opt.step()

        # ====== Step 2: update feature extractor (BN) via GRL ==============
        self.fe_opt.zero_grad()

        # pass target features through GRL then discriminator
        target_feat_rev = gradient_reversal(target_feat, lambda_)
        d_target_rev = self.discriminator(target_feat_rev)

        fe_loss = self.bce(d_target_rev, target_label)
        fe_loss.backward()
        self.fe_opt.step()

        return {
            "y_pred": y_pred,
            "y": y.cuda().float().flatten(),
            "disc_loss": float(disc_loss.item()),
            "fe_loss": float(fe_loss.item()),
            "lambda": lambda_,
        }