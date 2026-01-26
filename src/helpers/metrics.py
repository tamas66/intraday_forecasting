# src/helpers/crps.py
from __future__ import annotations

from typing import Sequence, Optional, Literal

import numpy as np
import torch


# ======================================================
# 1) Empirical CRPS from samples
#    CRPS(F, y) = E|X - y| - 0.5 E|X - X'|
# ======================================================

def crps_ensemble_numpy(y: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """
    Empirical CRPS for an ensemble/simulation forecast.

    Parameters
    ----------
    y : np.ndarray
        Observations, shape (...,)
    samples : np.ndarray
        Ensemble samples, shape (..., M) where M is number of samples.

    Returns
    -------
    crps : np.ndarray
        CRPS per entry, shape (...) (same as y)
    """
    y = np.asarray(y, dtype=float)
    s = np.asarray(samples, dtype=float)

    if s.shape[:-1] != y.shape:
        raise ValueError(f"samples shape {s.shape} incompatible with y shape {y.shape}")

    # E|X - y|
    term1 = np.mean(np.abs(s - y[..., None]), axis=-1)

    # 0.5 E|X - X'| computed as mean pairwise abs differences
    # Efficient computation: use sorting identity:
    # mean_{i,j} |x_i - x_j| = (2 / M^2) * sum_{k=1}^M (2k - M - 1) * x_(k)
    # with x_(k) sorted ascending.
    M = s.shape[-1]
    s_sorted = np.sort(s, axis=-1)
    k = np.arange(1, M + 1, dtype=float)  # 1..M
    coeff = (2.0 * k - M - 1.0)  # shape (M,)
    term2 = (2.0 / (M * M)) * np.sum(s_sorted * coeff, axis=-1)

    return term1 - 0.5 * term2


def crps_ensemble_torch(y: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
    """
    Torch version of empirical CRPS. Keeps gradients off by default in eval.

    y: (...,)
    samples: (..., M)
    returns: (...,)
    """
    if samples.shape[:-1] != y.shape:
        raise ValueError(f"samples shape {samples.shape} incompatible with y shape {y.shape}")

    term1 = torch.mean(torch.abs(samples - y.unsqueeze(-1)), dim=-1)

    M = samples.shape[-1]
    s_sorted, _ = torch.sort(samples, dim=-1)
    k = torch.arange(1, M + 1, device=samples.device, dtype=samples.dtype)
    coeff = (2.0 * k - M - 1.0)
    term2 = (2.0 / (M * M)) * torch.sum(s_sorted * coeff, dim=-1)

    return term1 - 0.5 * term2


# ======================================================
# 2) CRPS from quantiles (discrete approximation)
#    CRPS(F,y) = 2 * ∫_0^1 rho_tau(y - q_tau) d tau
#    Approximate integral over taus with trapezoidal weights.
# ======================================================

def _trapz_weights(taus: np.ndarray) -> np.ndarray:
    """
    Trapezoidal weights over tau grid on [0,1].
    weights sum to 1 over [0,1] only if taus include 0 and 1;
    but we just need correct trapezoid integration weights.
    """
    if np.any(np.diff(taus) <= 0):
        raise ValueError("taus must be strictly increasing")

    w = np.zeros_like(taus, dtype=float)
    w[0] = 0.5 * (taus[1] - taus[0])
    w[-1] = 0.5 * (taus[-1] - taus[-2])
    if len(taus) > 2:
        w[1:-1] = 0.5 * (taus[2:] - taus[:-2])
    return w


def crps_from_quantiles_numpy(
    y: np.ndarray,
    q: np.ndarray,
    taus: Sequence[float],
) -> np.ndarray:
    """
    Approximate CRPS using a finite set of predicted quantiles.

    Parameters
    ----------
    y : np.ndarray
        Observations, shape (...,)
    q : np.ndarray
        Predicted quantiles, shape (..., Q)
    taus : list[float]
        Quantile levels, length Q, strictly increasing.

    Returns
    -------
    crps : np.ndarray
        Approximate CRPS per entry, shape (...)
    """
    y = np.asarray(y, dtype=float)
    q = np.asarray(q, dtype=float)
    taus = np.asarray(list(taus), dtype=float)

    if q.shape[:-1] != y.shape or q.shape[-1] != len(taus):
        raise ValueError(f"q shape {q.shape} incompatible with y {y.shape} and taus {len(taus)}")

    w = _trapz_weights(taus)  # (Q,)

    # Pinball loss rho_tau(u) with u = y - q_tau
    # rho_tau(u) = (tau - 1_{u<0}) * u
    u = y[..., None] - q  # (..., Q)
    indicator = (u < 0).astype(float)
    rho = (taus[None, ...] - indicator) * u  # broadcast taus to (..., Q)

    # CRPS ≈ 2 * ∑ w_i * rho_i
    crps = 2.0 * np.sum(rho * w[None, ...], axis=-1)
    return crps


def crps_from_quantiles_torch(
    y: torch.Tensor,
    q: torch.Tensor,
    taus: Sequence[float],
) -> torch.Tensor:
    """
    Torch version of quantile-approx CRPS.

    y: (...,)
    q: (..., Q)
    """
    taus_np = np.asarray(list(taus), dtype=float)
    if np.any(np.diff(taus_np) <= 0):
        raise ValueError("taus must be strictly increasing")

    # weights computed in numpy then moved to device
    w_np = _trapz_weights(taus_np)
    w = torch.tensor(w_np, device=q.device, dtype=q.dtype)  # (Q,)
    taus_t = torch.tensor(taus_np, device=q.device, dtype=q.dtype)  # (Q,)

    if q.shape[:-1] != y.shape or q.shape[-1] != len(taus_np):
        raise ValueError(f"q shape {q.shape} incompatible with y {y.shape} and taus {len(taus_np)}")

    u = y.unsqueeze(-1) - q  # (..., Q)
    indicator = (u < 0).to(q.dtype)
    rho = (taus_t - indicator) * u

    return 2.0 * torch.sum(rho * w, dim=-1)


# ======================================================
# 3) Optional: Sampling from quantiles (inverse-CDF via linear interpolation)
#    Useful if you want LSTM CRPS via sampling (or mixture sampling later).
# ======================================================

def sample_from_quantiles_numpy(
    q: np.ndarray,
    taus: Sequence[float],
    n_samples: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Draw samples from a distribution represented by quantiles (taus, q)
    using inverse-CDF linear interpolation.

    q: (..., Q)
    returns: (..., M)
    """
    q = np.asarray(q, dtype=float)
    taus = np.asarray(list(taus), dtype=float)

    if rng is None:
        rng = np.random.default_rng()

    Q = q.shape[-1]
    if Q != len(taus):
        raise ValueError("Last dim of q must match len(taus)")
    if np.any(np.diff(taus) <= 0):
        raise ValueError("taus must be strictly increasing")

    u = rng.uniform(taus[0], taus[-1], size=(*q.shape[:-1], n_samples))  # (..., M)

    # For each u, find right bin index j such that taus[j-1] <= u < taus[j]
    idx = np.searchsorted(taus, u, side="right")
    idx = np.clip(idx, 1, Q - 1)

    t0 = taus[idx - 1]
    t1 = taus[idx]
    q0 = np.take_along_axis(q, (idx - 1)[..., None], axis=-1).squeeze(-1)
    q1 = np.take_along_axis(q, idx[..., None], axis=-1).squeeze(-1)

    w = (u - t0) / (t1 - t0 + 1e-12)
    return q0 + w * (q1 - q0)


# ======================================================
# 4) Convenience wrappers for your shapes (T,H,...) / (B,H,...)
# ======================================================

def crps_lstm_quantiles(
    y_true: np.ndarray,
    q_pred: np.ndarray,
    taus: Sequence[float],
    reduce: Literal["none", "mean"] = "mean",
) -> np.ndarray:
    """
    LSTM output format:
      y_true: (T, H)
      q_pred: (T, H, Q)
    """
    crps = crps_from_quantiles_numpy(y_true, q_pred, taus)  # (T, H)
    if reduce == "mean":
        return np.mean(crps)
    return crps


def crps_garch_samples(
    y_true: np.ndarray,
    samples: np.ndarray,
    reduce: Literal["none", "mean"] = "mean",
) -> np.ndarray:
    """
    GARCH output format:
      y_true: (T, H)
      samples: (T, H, M)
    """
    crps = crps_ensemble_numpy(y_true, samples)  # (T, H)
    if reduce == "mean":
        return np.mean(crps)
    return crps
