import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict
from typing import Dict, Tuple

def calculate_forecast_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    training_time: float = None,
    inference_time: float = None,
) -> Dict[str, float]:
    """
    Calculate comprehensive forecast evaluation metrics including timing.
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual values
    y_pred : np.ndarray
        Predicted values
    training_time : float, optional
        Time taken to train model (seconds)
    inference_time : float, optional
        Time taken for inference (seconds)
    
    Returns
    -------
    dict
        Dictionary of metric names and values
    """
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    pct_errors = 100 * errors / y_true
    
    metrics = {
        # Basic errors
        "MAE": np.mean(abs_errors),
        "RMSE": np.sqrt(np.mean(errors**2)),
        "MedAE": np.median(abs_errors),
        
        # Percentage errors
        "MAPE": np.mean(np.abs(pct_errors)),
        "sMAPE": 200 * np.mean(np.abs(errors) / (np.abs(y_true) + np.abs(y_pred))),
        
        # Bias metrics
        "ME": np.mean(errors),
        "MPE": np.mean(pct_errors),
        
        # R-squared
        "R2": 1 - (np.sum(errors**2) / np.sum((y_true - np.mean(y_true))**2)),
        
        # Quantile metrics
        "Q90_AE": np.quantile(abs_errors, 0.9),
        "Q95_AE": np.quantile(abs_errors, 0.95),
    }
    
    # Add timing metrics if provided
    if training_time is not None:
        metrics["training_time_seconds"] = training_time
        metrics["training_time_minutes"] = training_time / 60
        
    if inference_time is not None:
        metrics["inference_time_seconds"] = inference_time
        metrics["inference_time_per_prediction_ms"] = (inference_time / len(y_pred)) * 1000
    
    # Efficiency metric: accuracy per second of training
    if training_time is not None and training_time > 0:
        metrics["mae_per_training_second"] = metrics["MAE"] / training_time
        metrics["efficiency_score"] = (1 - metrics["MAPE"]/100) / (training_time / 60)  # Accuracy per minute
    
    return metrics


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string"""
    if seconds < 1:
        return f"{seconds*1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"

def calculate_residual_diagnostics(residuals: np.ndarray) -> Dict[str, float]:
    """
    Calculate residual diagnostic statistics.
    """
    from scipy import stats
    
    diagnostics = {
        "mean": np.mean(residuals),
        "std": np.std(residuals),
        "skewness": stats.skew(residuals),
        "kurtosis": stats.kurtosis(residuals),
        "jarque_bera_stat": stats.jarque_bera(residuals)[0],
        "jarque_bera_pval": stats.jarque_bera(residuals)[1],
    }
    
    # Ljung-Box test for autocorrelation
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=False)
    diagnostics["ljung_box_stat"] = lb_test[0][0]
    diagnostics["ljung_box_pval"] = lb_test[1][0]
    
    return diagnostics


# ======================================================
# PLOTTING
# ======================================================

def plot_mean_diagnostics(y, fitted, resid, title: str):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(y, label="Actual", alpha=0.7)
    axes[0].plot(fitted, label="Fitted", alpha=0.7)
    axes[0].set_title(f"{title} — Actual vs Fitted")
    axes[0].legend()

    axes[1].plot(resid, color="black", alpha=0.7)
    axes[1].set_title("Residuals")

    axes[2].hist(resid.dropna(), bins=50, density=True)
    axes[2].set_title("Residual Distribution")

    plt.tight_layout()
    plt.show()


def plot_conditional_volatility(garch_res):
    vol = garch_res.conditional_volatility

    plt.figure(figsize=(12, 4))
    plt.plot(vol, alpha=0.8)
    plt.title("Conditional Volatility (GARCH)")
    plt.tight_layout()
    plt.show()


def plot_regime_probabilities(markov_res):
    probs = markov_res.smoothed_marginal_probabilities

    plt.figure(figsize=(12, 5))
    for regime in probs.columns:
        plt.plot(probs[regime], label=f"Regime {regime}")

    plt.title("Smoothed Regime Probabilities")
    plt.legend()
    plt.tight_layout()
    plt.show()