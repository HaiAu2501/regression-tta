from pathlib import Path

import numpy as np
import torch
from torch import Tensor


@torch.no_grad()
def compute_feature_stats(features: Tensor, ddof: int = 1) -> dict[str, Tensor]:
    features = features.cpu()
    mean = features.mean(dim=0)
    centered = features - mean
    cov = centered.T @ centered / (features.shape[0] - ddof)
    eigvals, eigvecs = np.linalg.eigh(cov.numpy())
    return {
        "mean": mean,
        "basis": torch.from_numpy(eigvecs),
        "eigvals": torch.from_numpy(eigvals),
    }


@torch.no_grad()
def load_pca_basis(stat_file: str | Path,
                   top_k: int,
                   device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    stat_dict = torch.load(stat_file, map_location="cpu")
    mean: Tensor = stat_dict["mean"].float()
    basis: Tensor = stat_dict["basis"].float()
    eigvals: Tensor = stat_dict["eigvals"].float()

    top_idx = np.argsort(eigvals.numpy())[-top_k:]
    indices = torch.from_numpy(top_idx).long()
    return (
        mean.to(device),
        basis[:, indices].float().to(device),
        eigvals[indices].float().to(device),
    )
