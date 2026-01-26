# src/models/garch.py
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


def _extract_regime_params(ms_res) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Extract per-regime intercept + exog coefficients from statsmodels MarkovRegression.

    We fit with switching intercept, non-switching exog by default for stability.
    If exog also switches, this still works because params will include regime-specific
    exog blocks; we attempt to parse by name.

    Returns:
      {regime: {"const": float, "beta": (k_exog,)}}  (beta may be shared)
    """
    names = ms_res.model.param_names
    params = np.asarray(ms_res.params)

    # Intercepts usually appear as "const[0]", "const[1]" or similar
    # Exog usually as "x1", "x2", ... (possibly with regime suffix if switching)
    out: Dict[int, Dict[str, Any]] = {}
    n_states = ms_res.model.k_regimes
    k_exog = ms_res.model.k_exog

    # defaults
    for s in range(n_states):
        out[s] = {"const": 0.0, "beta": np.zeros(k_exog, dtype=float)}

    # Map exog names in model order
    # statsmodels uses 'x1'...'xk' for exog columns if you pass ndarray
    # If you pass DataFrame, it uses column names.
    exog_names = []
    if ms_res.model.exog is not None:
        # If exog was DataFrame, statsmodels keeps names; otherwise it makes "x1"... format
        if hasattr(ms_res.model.exog, "columns"):
            exog_names = list(ms_res.model.exog.columns)  # type: ignore
        else:
            exog_names = [f"x{i+1}" for i in range(k_exog)]

    # Fill by parsing
    for name, val in zip(names, params):
        # intercept per regime
        if name.startswith("const"):
            # try const[0] pattern
            if "[" in name and "]" in name:
                s = int(name.split("[")[1].split("]")[0])
                out[s]["const"] = float(val)
            else:
                # single intercept shared
                for s in range(n_states):
                    out[s]["const"] = float(val)
        else:
            # exog coefficient (shared or switching)
            # possible patterns:
            #   "beta.x1" / "x1" / "x1[0]" etc depending on statsmodels version
            base = name
            s_reg = None
            if "[" in name and "]" in name:
                try:
                    s_reg = int(name.split("[")[1].split("]")[0])
                    base = name.split("[")[0]
                except Exception:
                    s_reg = None

            # try match to exog names
            # strip common prefixes
            for pref in ("beta.", "exog.", "b."):
                if base.startswith(pref):
                    base = base[len(pref):]

            if base in exog_names:
                j = exog_names.index(base)
                if s_reg is None:
                    # shared across regimes
                    for s in range(n_states):
                        out[s]["beta"][j] = float(val)
                else:
                    out[s_reg]["beta"][j] = float(val)

    return out


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

    def __init__(self, cfg: GarchEVTConfig):
        self.cfg = cfg

        self.ms_res = None
        self.garch_res = None

        self._regime_params = None
        self._last_prob = None

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

        # --- Markov switching mean ---
        # Switching intercept; keep exog coefficients non-switching for stability by default.
        # (If you later want switching betas, expose a flag.)
        ms_model = MarkovRegression(
            endog=y,
            exog=X_mean,
            k_regimes=self.cfg.n_states,
            trend="c",
            switching_variance=False,
        )
        self.ms_res = ms_model.fit(disp=False)

        # Extract last filtered regime probabilities
        # filtered_marginal_probabilities: (T, n_states)
        filt = np.asarray(self.ms_res.filtered_marginal_probabilities)
        self._last_prob = filt[-1, :].copy()

        # Build fitted mean series using smoothed/filtered probs (for residuals)
        # We'll use filtered probs for causality.
        reg_params = _extract_regime_params(self.ms_res)
        self._regime_params = reg_params

        if X_mean is None:
            mu_reg = np.array([reg_params[s]["const"] for s in range(self.cfg.n_states)], dtype=float)
            mu_hat = (filt * mu_reg.reshape(1, -1)).sum(axis=1)
        else:
            mu_by_state = []
            for s in range(self.cfg.n_states):
                mu_s = reg_params[s]["const"] + X_mean @ reg_params[s]["beta"]
                mu_by_state.append(mu_s)
            mu_by_state = np.stack(mu_by_state, axis=1)  # (T, n_states)
            mu_hat = (filt * mu_by_state).sum(axis=1)

        resid = y - mu_hat

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
        self.garch_res = garch.fit(disp=self.cfg.disp, maxiter=self.cfg.maxiter)

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
        regime_prob: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Forecast conditional mean for horizon steps using last regime probabilities
        and per-regime intercept+beta.

        X_future: shape (H, k) or None
        Returns: mu shape (H,)
        """
        if self._regime_params is None or self._last_prob is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        H = int(horizon)
        X_future = _as_2d(X_future)

        p = self._last_prob if regime_prob is None else np.asarray(regime_prob, dtype=float).reshape(-1)
        p = p / p.sum()

        if X_future is None:
            mu_reg = np.array([self._regime_params[s]["const"] for s in range(self.cfg.n_states)], dtype=float)
            return np.repeat((p * mu_reg).sum(), H)

        mu_by_state = []
        for s in range(self.cfg.n_states):
            mu_s = self._regime_params[s]["const"] + X_future @ self._regime_params[s]["beta"]
            mu_by_state.append(mu_s)
        mu_by_state = np.stack(mu_by_state, axis=1)  # (H, n_states)
        mu = (mu_by_state * p.reshape(1, -1)).sum(axis=1)
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
        y_samp = mu.reshape(H, 1) + eps  # (H, M)
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
        n_states=cfg.garch.regimes.n_states,
        p=cfg.garch.variance.p,
        q=cfg.garch.variance.q,
        dist=cfg.garch.distribution.core,
        use_evt=cfg.garch.spikes.use_evt,
        upper_threshold=cfg.garch.spikes.upper_threshold,
        lower_threshold=cfg.garch.spikes.lower_threshold,             
    )
    return MSGarchEVT(gcfg)

