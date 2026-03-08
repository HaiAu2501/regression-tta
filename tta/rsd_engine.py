"""
Representation Subspace Distance (RSD) Test-Time Adaptation engine.

Implements the subspace alignment approach from:
    Chen & Chen, "Representation Subspace Distance for Domain Adaptation
    Regression", ICML 2021.

Core idea
---------
Instead of matching marginal statistics along each principal axis (as SSA
does), RSD aligns the *subspaces* themselves by minimising the Frobenius
distance between "representation matrices"  Σ_k V_k^T  for source and
target, where V_k are the top-k right singular vectors and Σ_k the
corresponding singular values of the centred feature matrix.

    d²_RSD = (1/k) ‖ diag(σ_s) V_s^T − diag(σ_t) V_t^T ‖²_F
           = (1/k) [ Σ σ²_{s,i} + Σ σ²_{t,i}
                      − 2 · tr(diag(σ_s) · V_s^T V_t · diag(σ_t)) ]

The cross-term  tr(diag(σ_s) V_s^T V_t diag(σ_t))  captures both
principal-angle alignment and singular-value matching simultaneously.

TTA adaptation
--------------
* Source subspace is pre-computed from ``feature_stats.pt`` (eigenvectors
  of the covariance = right singular vectors; eigenvalues ∝ σ²).
* At test time, the target batch features are centred (with source mean),
  and their SVD provides the target subspace.
* Only BN parameters are updated via backprop through the differentiable
  SVD.

Note: ``top_k`` is clamped to ``min(top_k, batch_size)`` because SVD of
a (B, D) matrix yields at most B singular components.
"""
from dataclasses import dataclass, InitVar

import torch
from torch import Tensor
from ignite.engine import Engine
from ignite.metrics import RootMeanSquaredError, MeanAbsoluteError
from ignite.contrib.metrics.regression.r2_score import R2Score

from evaluation.metrics import ModelDistanceMetric, PearsonCorrelation
from model import Regressor
from utils.pca_basis import get_pca_basis


@dataclass
class RSDEngine(Engine):
    net: Regressor
    opt: torch.optim.Optimizer
    train_mode: bool
    pc_config: InitVar[dict]
    loss_config: InitVar[dict]
    compile_model: InitVar[dict | None]

    @torch.no_grad()
    def __post_init__(self, pc_config: dict, loss_config: dict,
                      compile_model: dict | None):
        super().__init__(self.update)

        # --- metrics (same interface as other engines) ---------------------
        y_ot = lambda d: (d["y_pred"], d["y"])
        RootMeanSquaredError(y_ot).attach(self, "rmse_loss")
        MeanAbsoluteError(y_ot).attach(self, "mae_loss")
        R2Score(y_ot).attach(self, "R2")
        PearsonCorrelation(y_ot).attach(self, "r")
        ModelDistanceMetric(self.net).attach(self, "model_dist")

        # --- source PCA stats ---------------------------------------------
        mean, basis, pc_vars = get_pca_basis(**pc_config)
        self.mean = mean.cuda()            # (D,)
        self.basis = basis.cuda()          # (D, K)  eigenvectors = right SVs
        self.K = self.basis.shape[1]

        # eigenvalues of covariance ∝ σ² / (n-1)
        # We use sqrt(eigenvalue) as the singular-value proxy; the (n-1)
        # scale factor cancels in the loss normalisation.
        self.src_sv = pc_vars.sqrt().cuda()  # (K,)

        self.eps = loss_config.get("eps", 1e-8)
        self.trade_off = loss_config.get("trade_off", 1.0)

        print(f"[RSD] K={self.K}, trade_off={self.trade_off}, eps={self.eps}")
        print(f"[RSD] src singular-value range: "
              f"[{self.src_sv.min():.4f}, {self.src_sv.max():.4f}]")

        # --- optional torch.compile ----------------------------------------
        self.feature_extractor = self.net.feature
        if compile_model is not None:
            try:
                self.feature_extractor = torch.compile(
                    self.net.feature, **compile_model)
            except RuntimeError as e:
                print(f"torch.compile failed: {e}")

    def update(self, engine: Engine,
               batch: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
        if self.train_mode:
            self.net.train()
        else:
            self.net.eval()
        self.net.zero_grad()

        x, y = batch
        x = x.cuda()
        B = x.shape[0]

        # --- extract features & centre ------------------------------------
        feature = self.feature_extractor(x)          # (B, D)
        y_pred = self.net.predict_from_feature(feature)

        centred = feature - self.mean                # (B, D)

        # --- target SVD ---------------------------------------------------
        # centred: (B, D) with B ≤ D typically  →  thin SVD gives (B,) SVs
        U_t, S_t, Vh_t = torch.linalg.svd(centred, full_matrices=False)
        # U_t: (B, B), S_t: (B,), Vh_t: (B, D)

        K = min(self.K, B)

        # top-K target singular values & right singular vectors
        tgt_sv = S_t[:K]                             # (K,)
        V_t = Vh_t[:K, :].T                          # (D, K)

        # source top-K (already stored, slice in case K < self.K)
        src_sv = self.src_sv[:K]                     # (K,)
        V_s = self.basis[:, :K]                      # (D, K)

        # --- RSD loss -----------------------------------------------------
        # d² = (1/K) [ Σ σ²_s + Σ σ²_t − 2 tr(Σ_s V_s^T V_t Σ_t) ]
        cross = V_s.T @ V_t                          # (K, K)
        weighted_cross = (src_sv.unsqueeze(1) * cross) * tgt_sv.unsqueeze(0)

        rsd_loss = (src_sv.square().sum()
                    + tgt_sv.square().sum()
                    - 2.0 * weighted_cross.trace()) / K

        loss = self.trade_off * rsd_loss

        loss.backward()
        self.opt.step()

        # --- PCA projection for logging (no grad) -------------------------
        with torch.no_grad():
            f_pc = centred @ self.basis               # (B, K_full)

        return {
            "y_pred": y_pred,
            "y": y.cuda().float().flatten(),
            "feat_pc": f_pc,
        }