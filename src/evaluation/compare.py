import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import torch

from data import load_parametric_data, load_parametric_dataset, load_lstm_data
from modelling.model_config import (
    SARIMAXConfig,
    GARCHConfig,
    LSTMConfig,
)
from modelling.parametric_model import (
    SARIMAXModel,
    GARCHModel,
)
from modelling.ann_model import LSTMModel
from evaluation.metrics import compute_metrics
from train import train_lstm
from config import MODELS_DIR, FINAL_DATA_DIR


def _run_parametric_model_fixed_split(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    timestamps,
) -> Dict:
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    return {
        "evaluation": "fixed_split",
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "train_time": train_time,
        "y_true": y_test,
        "y_pred": y_pred,
        "timestamps": timestamps,
    }


def _run_lstm_fixed_split(
    config: LSTMConfig,
    target: str,
    force_retrain: bool = False,
) -> Dict:
    model_path = MODELS_DIR / target / "lstm.pt"

    # Train if missing or forced
    if not model_path.exists() or force_retrain:
        stats = train_lstm(config=config, target=target)
        train_time = stats["training_time"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    )

    model = LSTMModel(config).to(device)
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )
    train_time = checkpoint["training_time"]
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, test_loader, test_timestamps = load_lstm_data(
        config=config,
        target=target,
        verbose=False,
    )

    preds = []
    y_true = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_hat = model(x_batch)
            preds.append(y_hat.detach().cpu().numpy())
            y_true.append(y_batch.detach().cpu().numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(y_true)

    metrics = compute_metrics(y_true, y_pred)

    return {
        "evaluation": "fixed_split",
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "train_time": train_time,
        "y_true": y_true,
        "y_pred": y_pred,
        "timestamps": test_timestamps,
    }


def run_comparison(
    target: str = "level",
    force_retrain: bool = False,
) -> Dict[str, Dict]:
    """
    Run full comparison across all models, supporting:
    - fixed train/test evaluation for SARIMAX, GARCH-X, LSTM
    - expanding-window refit evaluation for Markov-switching
    """
    results: Dict[str, Dict] = {}

    # ==================================================
    # Fixed-split models (load once)
    # ==================================================
    X_train, y_train, X_test, y_test, timestamps = load_parametric_data(
        target=target,
        verbose=False,
        exogenous_type="core",
    )

    # SARIMAX (fixed split)
    results["SARIMAX"] = _run_parametric_model_fixed_split(
        SARIMAXModel(SARIMAXConfig()),
        X_train, y_train, X_test, y_test, timestamps
    )

    X_train, y_train, X_test, y_test, timestamps = load_parametric_data(
        target=target,
        verbose=False,
        exogenous_type="variance",
    )

    # GARCH-X (fixed split)
    results["GARCH-X"] = _run_parametric_model_fixed_split(
        GARCHModel(GARCHConfig()),
        X_train, y_train, X_test, y_test, timestamps
    )

    # LSTM (fixed split)
    results["LSTM"] = _run_lstm_fixed_split(
        config=LSTMConfig(),
        target=target,
        force_retrain=force_retrain,
    )

    return results
