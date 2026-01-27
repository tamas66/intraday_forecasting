# src/models/garchx.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

from arch import arch_model
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

try:
    from scipy.stats import genpareto
except Exception as e:
    genpareto = None


# ======================================================
# CONFIG
# ======================================================

@dataclass
class GarchEVTConfig:
    # Mean (Markov switching)
    n_states: int = 2
    mean_ar_order: int = 1
    # GARCH
    p: int = 1
    q: int = 1
    dist: str = "t"  # "t" in arch

    # EVT
    use_evt: bool = True
    upper_threshold: float = 3.0   # z > +u
    lower_threshold: float = -3.0  # z < -u
    separate_tails: bool = True

    # Safety / numerics
    maxiter: int = 2000
    disp: str = "off"


# ======================================================
# EVT HELPERS (standardized residual tails)
# ======================================================

@dataclass
class GPDTail:
    # scipy genpareto parameterization: c, loc, scale
    c: float
    loc: float
    scale: float

    def rvs(self, size: int, rng: np.random.Generator) -> np.ndarray:
        if genpareto is None:
            raise ImportError("scipy is required for EVT (genpareto). Install scipy.")
        # scipy's rvs uses global RNG unless random_state passed
        return genpareto.rvs(self.c, loc=self.loc, scale=self.scale, size=size, random_state=rng)


@dataclass
class EVTFit:
    u_pos: float
    u_neg: float
    tail_pos: Optional[GPDTail]
    tail_neg: Optional[GPDTail]


def _fit_gpd_on_exceedances(exceed: np.ndarray) -> Optional[GPDTail]:
    """
    Fit GPD to exceedances (values >= 0). Returns None if insufficient data.
    """
    exceed = np.asarray(exceed, dtype=float)
    exceed = exceed[np.isfinite(exceed)]
    exceed = exceed[exceed >= 0.0]

    # Need enough tail points to fit anything sensible
    if exceed.size < 50:
        return None

    if genpareto is None:
        raise ImportError("scipy is required for EVT (genpareto). Install scipy.")

    # Common POT choice: loc ~ 0 for exceedances; but we allow free fit for robustness
    c, loc, scale = genpareto.fit(exceed)
    # guard against degenerate fit
    if not np.isfinite([c, loc, scale]).all() or scale <= 0:
        return None
    return GPDTail(c=c, loc=loc, scale=scale)


def fit_evt_on_standardized_residuals(
    z: np.ndarray,
    upper_threshold: float,
    lower_threshold: float,
) -> EVTFit:
    """
    Fit separate GPD tails on standardized residuals:
      - positive tail: exceed = z - u_pos where z > u_pos
      - negative tail: exceed = (-z) - u_neg where z < -u_neg
    """
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]

    u_pos = float(upper_threshold)
    u_neg = float(abs(lower_threshold))

    pos_exceed = z[z > u_pos] - u_pos
    neg_exceed = (-z[z < -u_neg]) - u_neg

    tail_pos = _fit_gpd_on_exceedances(pos_exceed)
    tail_neg = _fit_gpd_on_exceedances(neg_exceed)

    return EVTFit(u_pos=u_pos, u_neg=u_neg, tail_pos=tail_pos, tail_neg=tail_neg)


def apply_evt_to_standardized_samples(
    z_samples: np.ndarray,
    evt: EVTFit,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Replace the extreme parts of z_samples with draws from fitted GPD tails,
    preserving sign and threshold continuity.

    z_samples: shape (...,)
    """
    z = np.asarray(z_samples, dtype=float).copy()

    # Positive tail
    if evt.tail_pos is not None:
        mask_pos = z > evt.u_pos
        n_pos = int(mask_pos.sum())
        if n_pos > 0:
            exceed = evt.tail_pos.rvs(size=n_pos, rng=rng)
            z[mask_pos] = evt.u_pos + exceed

    # Negative tail
    if evt.tail_neg is not None:
        mask_neg = z < -evt.u_neg
        n_neg = int(mask_neg.sum())
        if n_neg > 0:
            exceed = evt.tail_neg.rvs(size=n_neg, rng=rng)
            z[mask_neg] = -(evt.u_neg + exceed)

    return z


# ======================================================
# MARKOV MEAN HELPERS
# ======================================================

def _as_2d(x: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if x is None:
        return None
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


def _build_arx_design(y: np.ndarray, X_mean: np.ndarray | None, p: int):
    """
    Build ARX(p) design matrix with constant.
    Returns (Y_trimmed, X_design)
    """
    T = len(y)

    rows = []
    targets = []

    for t in range(p, T):
        row = [1.0]  # constant

        # AR terms
        for i in range(1, p + 1):
            row.append(y[t - i])

        # Exogenous
        if X_mean is not None:
            row.extend(X_mean[t])

        rows.append(row)
        targets.append(y[t])

    return np.asarray(targets), np.asarray(rows)



# ======================================================
# MAIN CLASS
# ======================================================

class MSGarchEVT:
    """
    Option A:
      - Markov switching regression for conditional mean (2 regimes)
      - Single Student-t GARCH(p,q) on residuals
      - Optional EVT (GPD) tails on standardized residuals, applied to simulations

    Rolling estimation is handled outside: call fit() each refit date on the rolling window.
    """

    def __init__(self, cfg: GarchEVTConfig, verbose: bool = False):
        self.cfg = cfg
        self.verbose = verbose

        self.mean_res = None
        self.garch_res = None

        self._ar_order: Optional[int] = None
        self._beta: Optional[np.ndarray] = None
        self._last_y: Optional[np.ndarray] = None
        self.evt_fit: Optional[EVTFit] = None

    # -----------------------
    # Fit
    # -----------------------
    def fit(
        self,
        y: np.ndarray,
        X_mean: Optional[np.ndarray] = None,
    ) -> "MSGarchEVT":
        """
        Fit Markov switching mean model and GARCH on residuals.
        y: shape (T,)
        X_mean: shape (T, k) or None
        """
        y = np.asarray(y, dtype=float).reshape(-1)
        X_mean = _as_2d(X_mean)

        # --- ARX mean ---
        p = self.cfg.mean_ar_order  # e.g. 1

        if X_mean is not None:
            X_mean = np.asarray(X_mean)

        y_arx, X_arx = _build_arx_design(y, X_mean, p)

        # OLS
        beta = np.linalg.lstsq(X_arx, y_arx, rcond=None)[0]

        # Save ARX parameters for forecasting
        self._ar_order = p
        self._beta = beta  # includes intercept + AR + exog
        self._last_y = y[-p:].copy()


        # Fitted mean aligned to original y
        mu_hat = np.full_like(y, np.nan, dtype=float)
        mu_hat[p:] = X_arx @ beta
        resid = y - mu_hat

        resid = np.asarray(resid, dtype=float)

        # Remove non-finite residuals
        mask = np.isfinite(resid)
        n_bad = (~mask).sum()

        if n_bad > 0:
            if self.verbose:
                print(f"[GARCH] Dropping {n_bad} non-finite residuals before GARCH fit")
            resid = resid[mask]
        # Minimum length required for GARCH
        min_obs = max(50, 5 * (self.cfg.p + self.cfg.q))

        if resid.shape[0] < min_obs:
            raise ValueError(
                f"Too few finite residuals for GARCH "
                f"(n={resid.shape[0]}, min_required={min_obs})"
            )

        # --- GARCH on residuals (Student-t) ---
        garch = arch_model(
            resid,
            mean="Zero",
            vol="GARCH",
            p=self.cfg.p,
            q=self.cfg.q,
            dist=self.cfg.dist,
            rescale=True,
        )
        self.garch_res = garch.fit(
            disp=self.cfg.disp,
            options={"maxiter": self.cfg.maxiter},
        )


        # --- EVT on standardized residuals (optional) ---
        if self.cfg.use_evt:
            # standardized residuals z_t = eps_t / sigma_t
            cond_vol = np.asarray(self.garch_res.conditional_volatility, dtype=float)
            z = resid / cond_vol
            self.evt_fit = fit_evt_on_standardized_residuals(
                z=z,
                upper_threshold=self.cfg.upper_threshold,
                lower_threshold=self.cfg.lower_threshold,
            )
        else:
            self.evt_fit = None

        return self

    # -----------------------
    # Forecast mean
    # -----------------------
    def forecast_mean(
        self,
        X_future: Optional[np.ndarray],
        horizon: int,
    ) -> np.ndarray:
        """
        Forecast ARX mean for H steps ahead.

        X_future: shape (H, k) or None
        Returns: mu shape (H,)
        """
        if not hasattr(self, "_beta"):
            raise RuntimeError("Model not fitted. Call fit() first.")

        beta = self._beta
        p = self._ar_order

        # Split coefficients
        idx = 0
        intercept = beta[idx]
        idx += 1

        ar_coefs = beta[idx : idx + p]
        idx += p

        exog_coefs = beta[idx:] if X_future is not None else None

        # Initialize with last observed y's
        y_hist = list(self._last_y)
        mu = np.zeros(horizon, dtype=float)

        for h in range(horizon):
            ar_part = sum(ar_coefs[i] * y_hist[-(i + 1)] for i in range(p))
            exog_part = (
                np.dot(exog_coefs, X_future[h]) if exog_coefs is not None else 0.0
            )

            mu_h = intercept + ar_part + exog_part
            mu[h] = mu_h

            y_hist.append(mu_h)  # recursive forecast

        return mu


    # -----------------------
    # Forecast samples
    # -----------------------
    def forecast_samples(
        self,
        X_future: Optional[np.ndarray],
        horizon: int,
        n_sim: int = 1000,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Return predictive samples for y_{t+1:t+H}.

        Output shape: (H, n_sim)

        Notes:
          - Residual samples from GARCH via simulation forecast.
          - EVT tail replacement is applied to standardized residual samples (optional).
          - Mean forecast from Markov switching mean.
        """
        if self.garch_res is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        H = int(horizon)
        M = int(n_sim)
        rng = np.random.default_rng(seed)

        # Mean path
        mu = self.forecast_mean(X_future=X_future, horizon=H)

        # Residual simulations from arch
        # arch returns simulations of variance/mean; since mean='Zero' it's residuals.
        f = self.garch_res.forecast(
            horizon=H,
            method="simulation",
            simulations=M,
            reindex=False,
        )

        # f.simulations.residuals: shape (T?, H, M). With reindex=False, it's (1, H, M)
        eps = np.asarray(f.simulations.residuals)
        if eps.ndim == 3:
            eps = eps[-1]  # (H, M)
        elif eps.ndim == 2:
            pass
        else:
            raise RuntimeError(f"Unexpected residual simulation shape: {eps.shape}")

        # If EVT enabled, adjust tails in standardized space
        if self.cfg.use_evt and self.evt_fit is not None:
            # Need simulated sigmas to standardize
            sig2 = np.asarray(f.simulations.variances)
            if sig2.ndim == 3:
                sig2 = sig2[-1]  # (H, M)
            sigma = np.sqrt(sig2)

            z = eps / sigma
            z_adj = apply_evt_to_standardized_samples(z, self.evt_fit, rng=rng)
            eps = z_adj * sigma

        # Compose y = mu + eps
        y_samp = eps + mu[None, :]  # (H, M)
        return y_samp

    # -----------------------
    # Diagnostics
    # -----------------------
    def get_state(self) -> Dict[str, Any]:
        """
        Return fitted objects / key diagnostics for downstream evaluation/debugging.
        """
        return {
            "ms_result": self.ms_res,
            "garch_result": self.garch_res,
            "last_regime_prob": None if self._last_prob is None else self._last_prob.copy(),
            "evt_fit": self.evt_fit,
        }

from omegaconf import DictConfig

def garch_from_hydra(cfg: DictConfig) -> MSGarchEVT:
    """
    Build MSGarchEVT from Hydra config.
    """
    gcfg = GarchEVTConfig(
        n_states=cfg.model.regimes.n_states,
        p=cfg.model.variance.p,
        q=cfg.model.variance.q,
        dist=cfg.model.distribution.core,
        use_evt=cfg.model.spikes.use_evt,
        upper_threshold=cfg.model.spikes.upper_threshold,
        lower_threshold=cfg.model.spikes.lower_threshold,             
    )
    return MSGarchEVT(gcfg)

