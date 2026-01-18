# src/models/parametric_model.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict

import pandas as pd
import numpy as np
import time

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from arch import arch_model


# ======================================================
# CONFIG OBJECTS
# ======================================================

@dataclass
class SARIMAXConfig:
    order: tuple = (1, 0, 1)
    seasonal_order: tuple = (1, 0, 1, 24)
    trend: str = "c"


@dataclass
class GARCHConfig:
    p: int = 1
    q: int = 1
    dist: str = "t"
    use_exogenous: bool = True


@dataclass
class MarkovConfig:
    k_regimes: int = 2
    order: int = 1
    switching_variance: bool = True


# ======================================================
# COLUMN REGISTRIES
# ======================================================

MEAN_EXOG_COLS = [
    # Prices / fundamentals
    "da_price",

    # Demand
    "demand_actual",
    "demand_error",
    "demand_actual_lag1",
    "demand_actual_lag24",

    # Wind
    "wind_outturn",
    "wind_error",
    "wind_outturn_lag1",
    "wind_outturn_lag24",

    # Solar
    "solar_outturn",
    "solar_error",
    "solar_outturn_lag1",

    # Calendar
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
]

VARIANCE_EXOG_COLS = [
    "abs_demand_error",
    "abs_wind_error",
    "abs_solar_error",
    "abs_id_da_spread",
    "is_peak_15_18",
]


# ======================================================
# HELPER FUNCTIONS
# ======================================================

def _prepare_exog_data(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Safely extract and prepare exogenous variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe
    columns : List[str]
        List of column names to extract
        
    Returns
    -------
    pd.DataFrame
        Cleaned exogenous variables as numeric types
    """
    # Get available columns
    available_cols = [c for c in columns if c in df.columns]
    
    if not available_cols:
        return None
    
    # Extract and ensure numeric types
    X = df[available_cols].copy()
    
    # Convert all columns to numeric, coercing errors
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Drop any rows with NaN that resulted from conversion
    X = X.dropna()
    
    # Ensure float64 dtype for all columns
    X = X.astype(np.float64)
    
    return X


def _align_data(y: pd.Series, X: Optional[pd.DataFrame]) -> tuple:
    """
    Align target and exogenous variables by index.
    
    Parameters
    ----------
    y : pd.Series
        Target variable
    X : pd.DataFrame or None
        Exogenous variables
        
    Returns
    -------
    tuple
        (aligned_y, aligned_X) both as numpy arrays
    """
    if X is None:
        return y.values, None
    
    # Find common index
    common_idx = y.index.intersection(X.index)
    
    if len(common_idx) == 0:
        raise ValueError("No common index between y and X")
    
    # Align and convert to numpy
    y_aligned = y.loc[common_idx].values.astype(np.float64)
    X_aligned = X.loc[common_idx].values.astype(np.float64)
    
    return y_aligned, X_aligned


def _time_execution(func):
    """Decorator to time function execution"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        return result, elapsed_time
    return wrapper


# ======================================================
# MODEL 1 — SARIMAX (ARX) - WITH TIMING
# ======================================================
def fit_sarimax(
    df: pd.DataFrame,
    config: SARIMAXConfig,
    return_time: bool = False,
):
    """
    Fit SARIMAX model with exogenous variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe from load_parametric_dataset()
    config : SARIMAXConfig
        Model configuration
    return_time : bool
        If True, return (model, training_time) tuple
        
    Returns
    -------
    SARIMAXResults or tuple
        Fitted model results, optionally with training time
    """
    start_time = time.time()
    
    # Prepare target
    y = df["y"].copy()
    y = pd.to_numeric(y, errors='coerce').dropna()
    
    # Prepare exogenous variables
    X = _prepare_exog_data(df, MEAN_EXOG_COLS)
    
    # Align data
    if X is not None:
        y_aligned, X_aligned = _align_data(y, X)
    else:
        y_aligned = y.values
        X_aligned = None
    
    # Fit model
    model = SARIMAX(
        endog=y_aligned,
        exog=X_aligned,
        order=config.order,
        seasonal_order=config.seasonal_order,
        trend=config.trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    result = model.fit(disp=False, maxiter=200)
    
    training_time = time.time() - start_time
    
    if return_time:
        return result, training_time
    return result


# ======================================================
# MODEL 2 — SARIMAX + GARCH / GARCH-X - WITH TIMING
# ======================================================

def fit_sarimax_garch(
    df: pd.DataFrame,
    sarimax_config: SARIMAXConfig,
    garch_config: GARCHConfig,
    return_time: bool = False,
) -> Dict[str, object]:
    """
    Fit two-stage SARIMAX + GARCH model.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe from load_parametric_dataset()
    sarimax_config : SARIMAXConfig
        Mean equation configuration
    garch_config : GARCHConfig
        Variance equation configuration
    return_time : bool
        If True, include training times in return dict
        
    Returns
    -------
    dict
        Dictionary with 'sarimax', 'garch', and optionally timing info
    """
    overall_start = time.time()
    
    # ---- Stage 1: Mean equation (SARIMAX)
    sarimax_start = time.time()
    sarimax_res = fit_sarimax(df, sarimax_config, return_time=False)
    sarimax_time = time.time() - sarimax_start
    
    resid = pd.Series(sarimax_res.resid, index=df["y"].dropna().index).dropna()

    # ---- Stage 2: Variance equation (GARCH)
    garch_start = time.time()
    
    if garch_config.use_exogenous:
        Z = _prepare_exog_data(df, VARIANCE_EXOG_COLS)
        
        if Z is not None:
            common_idx = resid.index.intersection(Z.index)
            resid_aligned = resid.loc[common_idx].values
            Z_aligned = Z.loc[common_idx].values
        else:
            resid_aligned = resid.values
            Z_aligned = None
    else:
        resid_aligned = resid.values
        Z_aligned = None

    garch = arch_model(
        resid_aligned,
        vol="GARCH",
        p=garch_config.p,
        q=garch_config.q,
        dist=garch_config.dist,
        x=Z_aligned,
        rescale=False,
    )

    garch_res = garch.fit(disp="off", options={'maxiter': 500})
    garch_time = time.time() - garch_start
    
    total_time = time.time() - overall_start

    result = {
        "sarimax": sarimax_res,
        "garch": garch_res,
    }
    
    if return_time:
        result["timing"] = {
            "sarimax_time": sarimax_time,
            "garch_time": garch_time,
            "total_time": total_time,
        }
    
    return result


# ======================================================
# MODEL 3 — MARKOV-SWITCHING ARX - WITH TIMING
# ======================================================

def fit_markov_switching(
    df: pd.DataFrame,
    config: MarkovConfig,
    return_time: bool = False,
):
    """
    Fit Markov-switching autoregressive model with exogenous variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe from load_parametric_dataset()
    config : MarkovConfig
        Model configuration
    return_time : bool
        If True, return (model, training_time) tuple
        
    Returns
    -------
    MarkovRegressionResults or tuple
        Fitted model results, optionally with training time
    """
    start_time = time.time()
    
    # Prepare target
    y = df["y"].copy()
    y = pd.to_numeric(y, errors='coerce').dropna()
    
    # Prepare exogenous variables (including AR lags)
    ar_cols = ["y_lag1", "y_lag24"]
    exog_cols = ar_cols + [c for c in MEAN_EXOG_COLS if c in df.columns]
    
    X = _prepare_exog_data(df, exog_cols)
    
    if X is None:
        raise ValueError("No valid exogenous variables found for Markov-switching model")
    
    # Align data
    y_aligned, X_aligned = _align_data(y, X)
    
    # Fit model
    model = MarkovRegression(
        endog=y_aligned,
        exog=X_aligned,
        k_regimes=config.k_regimes,
        order=config.order,
        switching_variance=config.switching_variance,
    )

    result = model.fit(disp=False, maxiter=200, em_iter=20)
    
    training_time = time.time() - start_time
    
    if return_time:
        return result, training_time
    return result