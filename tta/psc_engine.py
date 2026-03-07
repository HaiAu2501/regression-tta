"""
Predictive Spectral Calibration (PSC) Test-Time Adaptation engine.

Generalises SSA/CWSA from "subspace alignment" to "predictive spectral
calibration" by modelling the full source representation as:

    Σ_s^{psc} = V_s Λ_s V_s^T  +  τ P_⊥

where P_⊥ = I − V_s V_s^T is the complement projector and
τ = mean(tail eigenvalues) is the residual spectral floor.

Target features are decomposed into:
  u_i = V_s^T (z_i − μ_s)   ∈ R^K   (signal support)
  r_i = P_⊥  (z_i − μ_s)    ∈ R^D   (spectral slack)

Loss = L_sig + L_slack

  L_sig:  CWSA probe-bank tomographic alignment inside K-dim support
  L_slack: symmetric KL between isotropic Gaussians N(0,τI) and
           N(μ̂_⊥, ν̂_⊥ I)  — a single scalar divergence controlling
           residual leakage outside the predictive support.

This is NOT "main loss + auxiliary loss". It is one unified spectral
divergence with a signal part and a slack part, both derived from the
same block-diagonal Gaussian model.

When L_slack is dropped, PSC reduces exactly to CWSA.
When the probe bank uses only axis vectors, PSC reduces to SSA + slack.
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
from .cwsa_engine import build_probe_bank, sym_kl_1d


@torch.no_grad()
def _compute_tau(stat_file: str, contrib_top_k: int,
                 eps: float = 1e-8) -> float:
    """Compute residual spectral floor τ from tail eigenvalues.

    τ = mean(λ_{K+1}, ..., λ_D)

    Falls back to eps if there are no tail eigenvalues.
    """
    import numpy as np
    stat_dict = torch.load(stat_file)
    eigvals: Tensor = stat_dict["eigvals"]              # all eigenvalues

    topk_idx = set(np.argsort(eigvals.numpy())[-contrib_top_k:].tolist())
    tail_vals = [float(eigvals[i]) for i in range(len(eigvals))
                 if i not in topk_idx]

    if len(tail_vals) == 0:
        print(f"[PSC] No tail eigenvalues (D ≤ K), τ = {eps}")
        return eps

    tau = max(float(sum(tail_vals) / len(tail_vals)), eps)
    print(f"[PSC] τ = {tau:.6f}  (from {len(tail_vals)} tail eigenvalues)")
    return tau


@dataclass
class PSCEngine(Engine):
    net: Regressor
    opt: torch.optim.Optimizer
    train_mode: bool
    pc_config: InitVar[dict]
    loss_config: InitVar[dict]
    weight_bias: InitVar[float]
    weight_exp: InitVar[float]
    compile_model: InitVar[dict | None]

    @torch.no_grad()
    def __post_init__(self, pc_config: dict, loss_config: dict,
                      weight_bias: float, weight_exp: float,
                      compile_model: dict | None):
        super().__init__(self.update)

        # --- metrics -------------------------------------------------------
        y_ot = lambda d: (d["y_pred"], d["y"])
        RootMeanSquaredError(y_ot).attach(self, "rmse_loss")
        MeanAbsoluteError(y_ot).attach(self, "mae_loss")
        R2Score(y_ot).attach(self, "R2")
        PearsonCorrelation(y_ot).attach(self, "r")
        ModelDistanceMetric(self.net).attach(self, "model_dist")

        # === Source PCA stats ==============================================
        mean, basis, pc_vars = get_pca_basis(**pc_config)
        self.mean = mean.cuda()             # (D,)
        self.basis = basis.cuda()           # (D, K)  — V_s
        self.pc_vars = pc_vars.cuda()       # (K,)    — λ_1..λ_K

        self.K = self.basis.shape[1]
        self.D = self.basis.shape[0]
        self.eps = loss_config.get("eps", 1e-8)
        self.slack_weight = loss_config.get("slack_weight", 1.0)

        # === Residual spectral floor τ =====================================
        self.tau = _compute_tau(
            stat_file=pc_config["stat_file"],
            contrib_top_k=pc_config["contrib_top_k"],
            eps=self.eps,
        )
        self.D_minus_K = self.D - self.K
        print(f"[PSC] D={self.D}, K={self.K}, D-K={self.D_minus_K}")

        # === Probe bank (CWSA signal term) =================================
        self.Q = build_probe_bank(self.K, device=self.basis.device)
        P = self.Q.shape[0]
        print(f"[PSC] num_probes={P}")

        # source variance per probe: q^T Λ_s q  (Λ_s diagonal)
        self.src_var_q = (self.Q.square() @ self.pc_vars)    # (P,)

        # head-aware weight: β_q = (|a^T q| + bias)^exp
        w = self.net.regressor.weight                        # (1, D)
        self.a = (self.basis.T @ w.T).squeeze(-1)            # (K,)
        a_dot_q = self.Q @ self.a                            # (P,)
        self.beta_q = (torch.abs(a_dot_q) + weight_bias
                       ).pow(weight_exp)                     # (P,)

        # === V_s V_s^T for complement projection ===========================
        # We don't materialise P_⊥ (D×D).  Instead:
        #   r_i = zc_i − V_s (V_s^T zc_i) = zc_i − V_s u_i
        # which costs O(DK) per sample.

        # === optional torch.compile ========================================
        self.feature_extractor = self.net.feature
        if compile_model is not None:
            try:
                self.feature_extractor = torch.compile(
                    self.net.feature, **compile_model)
            except RuntimeError as e:
                print(f"torch.compile failed: {e}")

    # ------------------------------------------------------------------
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

        # --- forward -------------------------------------------------------
        feature = self.feature_extractor(x)              # (B, D)
        y_pred = self.net.predict_from_feature(feature)

        # --- signal / slack decomposition ----------------------------------
        zc = feature - self.mean                         # (B, D)
        u = zc @ self.basis                              # (B, K)  signal
        r = zc - u @ self.basis.T                        # (B, D)  slack

        # ====================== L_sig (CWSA) ===============================
        proj = u @ self.Q.T                              # (B, P)
        mu_q = proj.mean(dim=0)                          # (P,)
        var_q = proj.var(dim=0, unbiased=False)           # (P,)

        src_mu_q = torch.zeros_like(mu_q)
        skl_q = sym_kl_1d(mu_q, var_q,
                          src_mu_q, self.src_var_q,
                          eps=self.eps)                   # (P,)

        loss_sig = (self.beta_q * skl_q).sum() / self.Q.shape[0]

        # ====================== L_slack ====================================
        mu_perp = r.mean(dim=0)                          # (D,)
        r_centered = r - mu_perp.unsqueeze(0)            # (B, D)
        # ν̂_⊥ = Σ_i ||r_i − μ̂_⊥||² / (B · (D−K))
        nu_perp = (r_centered.square().sum()
                   / (B * self.D_minus_K + self.eps)
                   + self.eps)

        # ||μ̂_⊥||² / (D−K)
        mu_sq_norm = mu_perp.square().sum() / (self.D_minus_K + self.eps)

        tau = self.tau
        loss_slack = 0.5 * (
            mu_sq_norm * (1.0 / tau + 1.0 / nu_perp)
            + tau / nu_perp
            + nu_perp / tau
            - 2.0
        )

        # ====================== total ======================================
        loss = loss_sig + self.slack_weight * loss_slack

        loss.backward()
        self.opt.step()

        return {
            "y_pred": y_pred,
            "y": y.cuda().float().flatten(),
            "feat_pc": u,
            "loss_sig": float(loss_sig.item()),
            "loss_slack": float(loss_slack.item()),
            "nu_perp": float(nu_perp.item()),
        }