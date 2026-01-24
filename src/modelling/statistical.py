import numpy as np
from omegaconf import DictConfig
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model


def _ensure_float64(X: np.ndarray) -> np.ndarray:
    """Ensure exogenous matrix is strictly float64."""
    X = np.asarray(X)
    if X.dtype != np.float64:
        X = X.astype(np.float64)
    return X


# ======================================================
# BASE CLASS
# ======================================================

class BaseParametricModel:
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError


# ======================================================
# SARIMAX (ARX)
# ======================================================

class SARIMAXModel(BaseParametricModel):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model = None
        self.result = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = _ensure_float64(X)
        y = np.asarray(y, dtype=np.float64)

        sarimax_cfg = self.cfg.model.sarimax

        self.model = SARIMAX(
            endog=y,
            exog=X,
            order=tuple(sarimax_cfg.order),
            seasonal_order=tuple(sarimax_cfg.seasonal_order),
            trend=sarimax_cfg.trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        self.result = self.model.fit(disp=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.result is None:
            raise RuntimeError("Model must be fitted before prediction")

        X = _ensure_float64(X)
        return self.result.forecast(steps=len(X), exog=X)


# ======================================================
# SARIMAX + GARCH-X
# ======================================================

class GARCHModel(BaseParametricModel):
    """
    SARIMAX mean equation with GARCH variance.

    Mean equation:
      - SARIMAX with exogenous regressors

    Variance equation:
      - GARCH(p, q) with variance exogenous variables
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.sarimax_res = None
        self.garch_res = None

    def fit(self, X_mean: np.ndarray, X_var: np.ndarray, y: np.ndarray):
        X_mean = _ensure_float64(X_mean)
        X_var = _ensure_float64(X_var)
        y = np.asarray(y, dtype=np.float64)

        sarimax_cfg = self.cfg.model.sarimax
        garch_cfg = self.cfg.model.garch

        sarimax = SARIMAX(
            endog=y,
            exog=X_mean,
            order=tuple(sarimax_cfg.order),
            seasonal_order=tuple(sarimax_cfg.seasonal_order),
            trend=sarimax_cfg.trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        self.sarimax_res = sarimax.fit(disp=False)

        resid = self.sarimax_res.resid

        garch = arch_model(
            resid,
            vol="GARCH",
            p=garch_cfg.p,
            q=garch_cfg.q,
            dist=garch_cfg.dist,
            x=X_var,
            rescale=False,
        )

        self.garch_res = garch.fit(disp="off")

    def predict(self, X_mean: np.ndarray) -> np.ndarray:
        if self.sarimax_res is None:
            raise RuntimeError("Model must be fitted before prediction")

        return self.sarimax_res.forecast(
            steps=len(X_mean),
            exog=_ensure_float64(X_mean),
        )
