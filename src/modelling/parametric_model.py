import numpy as np
from modelling.model_config import SARIMAXConfig, GARCHConfig

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from arch import arch_model

def _ensure_float64(X: np.ndarray) -> np.ndarray:
    """
    Ensure exogenous matrix is strictly float64.
    """
    X = np.asarray(X)
    if X.dtype != np.float64:
        X = X.astype(np.float64)
    return X


# ======================================================
# BASE CLASS
# ======================================================

class BaseParametricModel:
    def fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# ======================================================
# SARIMAX (ARX)
# ======================================================

class SARIMAXModel(BaseParametricModel):
    def __init__(self, config: SARIMAXConfig):
        self.config = config
        self.result = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = _ensure_float64(X)
        y = np.asarray(y, dtype=np.float64)
        self.model = SARIMAX(
            endog=y,
            exog=X,
            order=self.config.order,
            seasonal_order=self.config.seasonal_order,
            trend=self.config.trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.result = self.model.fit(disp=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.result is None:
            raise RuntimeError("Model must be fitted before prediction")
        return self.result.predict(start=0, end=len(X) - 1, exog=X)


# ======================================================
# SARIMAX + GARCH-X
# ======================================================

class GARCHModel(BaseParametricModel):
    def __init__(self, config: GARCHConfig):
        self.config = config
        self.sarimax_res = None
        self.garch_res = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = _ensure_float64(X)
        y = np.asarray(y, dtype=np.float64)
        # Mean equation
        sarimax = SARIMAX(
            endog=y,
            exog=X,
            order=(1, 0, 1),
            seasonal_order=(1, 0, 1, 24),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.sarimax_res = sarimax.fit(disp=False)

        resid = self.sarimax_res.resid

        # Variance equation
        garch = arch_model(
            resid,
            vol="GARCH",
            p=self.config.p,
            q=self.config.q,
            dist=self.config.dist,
            rescale=False,
        )
        self.garch_res = garch.fit(disp="off")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.sarimax_res is None:
            raise RuntimeError("Model must be fitted before prediction")

        # Mean forecast only (variance not evaluated in RMSE/MAE)
        return self.sarimax_res.predict(start=0, end=len(X) - 1, exog=X)