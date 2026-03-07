"""
Cramér–Wold Subspace Alignment (CWSA) Test-Time Adaptation engine.

Replaces SSA's coordinate-wise subspace matching with a tomographic view:
the target feature distribution is aligned to the source not only along
principal axes, but through a bank of one-dimensional projections inside
the significant subspace.

Probe bank Q consists of:
  - K axis probes:  e_1, ..., e_K
  - K(K-1) pairwise mix probes:  (e_i ± e_j) / sqrt(2)  for i < j

Total probes: K + K(K-1) = K^2.

For each probe q, we compute the 1D symmetric KL between the source
and target projected distributions, weighted by the probe's relevance
to the regression head.
"""
from dataclasses import dataclass, InitVar

import torch
from torch import Tensor
from ignite.engine import Engine
from ignite.metrics import RootMeanSquaredError, MeanAbsoluteError
from ignite.contrib.metrics.regression.r2_score import R2Score

from evaluation.metrics import ModelDistanceMetric, PearsonCorrelation
from model import Regressor
from utils.loss import diagonal_gaussian_kl_loss
from utils.pca_basis import get_pca_basis


def build_probe_bank(K: int, device: torch.device = torch.device("cpu")) -> Tensor:
    """Build the deterministic probe bank Q.

    Returns
    -------
    Q : Tensor, shape (num_probes, K)
        Each row is a unit-norm probe direction in the K-dim PCA subspace.
        Probes = axis vectors + pairwise (e_i ± e_j)/sqrt(2).
    """
    probes = []

    # --- axis probes: e_1, ..., e_K ---
    probes.append(torch.eye(K, device=device))  # (K, K)

    # --- pairwise mix probes ---
    inv_sqrt2 = 1.0 / (2.0 ** 0.5)
    for i in range(K):
        for j in range(i + 1, K):
            q_plus = torch.zeros(K, device=device)
            q_plus[i] = inv_sqrt2
            q_plus[j] = inv_sqrt2
            probes.append(q_plus.unsqueeze(0))

            q_minus = torch.zeros(K, device=device)
            q_minus[i] = inv_sqrt2
            q_minus[j] = -inv_sqrt2
            probes.append(q_minus.unsqueeze(0))

    return torch.cat(probes, dim=0)  # (K + K*(K-1), K) = (K^2, K)


def sym_kl_1d(mu1: Tensor, var1: Tensor,
              mu2: Tensor, var2: Tensor,
              eps: float = 1e-8) -> Tensor:
    """Symmetric KL divergence between two univariate Gaussians.

    All inputs are 1-D tensors of shape (num_probes,), computed in
    a vectorised fashion over all probes simultaneously.

    Returns shape (num_probes,).
    """
    # KL(p1 || p2) + KL(p2 || p1)
    # = 0.5 * [ var1/var2 + var2/var1 + (mu1-mu2)^2 * (1/var1 + 1/var2) - 2 ]
    var1_safe = var1 + eps
    var2_safe = var2 + eps
    diff_sq = (mu1 - mu2).square()

    skl = 0.5 * (var1_safe / var2_safe
                 + var2_safe / var1_safe
                 + diff_sq * (1.0 / var1_safe + 1.0 / var2_safe)
                 - 2.0)
    return skl


@dataclass
class CWSAEngine(Engine):
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
                      weight_bias: float,
                      weight_exp: float,
                      compile_model: dict | None):
        super().__init__(self.update)

        # --- metrics (same interface as SSA for comparable logging) ---------
        y_ot = lambda d: (d["y_pred"], d["y"])
        RootMeanSquaredError(y_ot).attach(self, "rmse_loss")
        MeanAbsoluteError(y_ot).attach(self, "mae_loss")
        R2Score(y_ot).attach(self, "R2")
        PearsonCorrelation(y_ot).attach(self, "r")
        ModelDistanceMetric(self.net).attach(self, "model_dist")

        # --- source PCA stats (same as SSA) --------------------------------
        mean, basis, pc_vars = get_pca_basis(**pc_config)
        self.mean = mean.cuda()            # (D,)
        self.basis = basis.cuda()          # (D, K)
        self.pc_vars = pc_vars.cuda()      # (K,)  eigenvalues λ_1..λ_K

        K = self.basis.shape[1]
        self.eps = loss_config.get("eps", 1e-8)

        # --- head vector in subspace: a = V_s^T w -------------------------
        # regressor.weight is (1, D), basis is (D, K)
        # a = V_s^T w = basis^T @ w^T  -> (K,)
        w = self.net.regressor.weight       # (1, D)
        self.a = (self.basis.T @ w.T).squeeze(-1)   # (K,)

        # --- build probe bank ----------------------------------------------
        self.Q = build_probe_bank(K, device=self.basis.device)  # (P, K)
        P = self.Q.shape[0]
        print(f"CWSA: K={K}, num_probes={P} "
              f"(expected K^2={K*K})")

        # --- precompute source variance per probe --------------------------
        # For probe q, source variance = q^T Λ q  where Λ = diag(λ)
        # Vectorised: (Q * (Q @ diag(λ))) summed over K
        #           = (Q^2 @ λ)
        self.src_var_q = (self.Q.square() @ self.pc_vars)  # (P,)

        # --- head-aware weighting per probe --------------------------------
        # β_q = (1 + |a^T q|)^weight_exp  with weight_bias as the "1"
        a_dot_q = self.Q @ self.a                          # (P,)
        self.beta_q = (torch.abs(a_dot_q) + weight_bias).pow(weight_exp)  # (P,)

        print(f"beta_q stats: min={self.beta_q.min():.4f}, "
              f"max={self.beta_q.max():.4f}, "
              f"mean={self.beta_q.mean():.4f}")

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

        # --- extract features & project into subspace ----------------------
        feature = self.feature_extractor(x)             # (B, D)
        y_pred = self.net.predict_from_feature(feature)

        # project to PCA subspace: zt = (z - μ_s) @ V_s   -> (B, K)
        f_pc = (feature - self.mean) @ self.basis       # (B, K)

        # --- project onto all probes at once -------------------------------
        # u = zt @ Q^T  -> (B, P)
        u = f_pc @ self.Q.T                             # (B, P)

        # --- target statistics per probe -----------------------------------
        mu_q = u.mean(dim=0)                            # (P,)
        var_q = u.var(dim=0, unbiased=False)             # (P,)

        # --- source statistics per probe (precomputed) ---------------------
        src_mu_q = torch.zeros_like(mu_q)               # (P,)  all zeros

        # --- symmetric KL per probe ---------------------------------------
        skl_q = sym_kl_1d(mu_q, var_q,
                          src_mu_q, self.src_var_q,
                          eps=self.eps)                  # (P,)

        # --- weighted aggregation ------------------------------------------
        loss = (self.beta_q * skl_q).sum() / self.Q.shape[0]

        loss.backward()
        self.opt.step()

        return {
            "y_pred": y_pred,
            "y": y.cuda().float().flatten(),
            "feat_pc": f_pc,
        }