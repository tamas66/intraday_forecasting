import numpy as np
import torch
from typing import Dict, Union


ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    """
    Ensure input is a NumPy array.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Root Mean Squared Error.
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Mean Absolute Error.
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> Dict[str, float]:
    """
    Compute all evaluation metrics for a model.
    """
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
    }
