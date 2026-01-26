import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from omegaconf import DictConfig
import hydra

from data import load_parametric_data, load_lstm_data
from src.modelling.garchx import SARIMAXModel, GARCHModel
from src.modelling.lstm import LSTMModel
from evaluation.metrics import compute_metrics
from train import train_lstm


# ======================================================
# FIXED-SPLIT: PARAMETRIC MODELS
# ======================================================

def _run_parametric_model_fixed_split(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    timestamps,
    X_var_train=None,
) -> Dict:
    """
    Fixed train/test evaluation for parametric models.
    """
    start_time = time.time()

    if isinstance(model, GARCHModel):
        if X_var_train is None:
            raise ValueError("GARCHModel requires variance regressors")
        model.fit(X_train, X_var_train, y_train)
    else:
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


# ======================================================
# FIXED-SPLIT: LSTM
# ======================================================

def _run_lstm_fixed_split(
    cfg: DictConfig,
    target: str,
    force_retrain: bool,
) -> Dict:
    model_dir = (
        f"{cfg.data.paths.models_dir}/{target}"
    )

    # --------------------------------------------------
    # Load or train model
    # --------------------------------------------------
    model_path = None
    if not force_retrain:
        model_files = list(sorted(
            Path(model_dir).glob("lstm_*.pt"),
            key=lambda p: p.stat().st_mtime,
        ))
        if model_files:
            model_path = model_files[-1]

    if model_path is None:
        stats = train_lstm(cfg=cfg, target=target, save_model=True)
        train_time = stats["training_time"]

        model_files = list(Path(model_dir).glob("lstm_*.pt"))
        model_path = max(model_files, key=lambda p: p.stat().st_mtime)
    else:
        checkpoint = torch.load(model_path, map_location="cpu")
        train_time = checkpoint["training_time"]

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------
    device = torch.device(
        cfg.model.lstm.runtime.device
        if torch.cuda.is_available()
        else "cpu"
    )

    model = LSTMModel(cfg).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, test_loader, test_timestamps, param_length = load_lstm_data(
        cfg=cfg,
        target=target,
        verbose=False,
    )

    assert param_length == checkpoint["param_length"], (
        "Mismatch between dataset length and checkpoint metadata"
    )

    preds, trues = [], []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_hat = model(x_batch)
            preds.append(y_hat.cpu().numpy())
            trues.append(y_batch.cpu().numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)

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


# ======================================================
# PUBLIC API
# ======================================================

def run_comparison(
    *,
    cfg: DictConfig,
    target: str,
    force_retrain: bool = False,
) -> Dict[str, Dict]:
    """
    Run full comparison across:
    - SARIMAX
    - GARCH-X
    - LSTM
    - Rolling / Expanding ARX
    """
    results: Dict[str, Dict] = {}

    # --------------------------------------------------
    # SARIMAX
    # --------------------------------------------------
    X_train, y_train, X_test, y_test, timestamps = load_parametric_data(
        cfg=cfg,
        target=target,
        model="sarimax",
        verbose=False,
    )

    results["SARIMAX"] = _run_parametric_model_fixed_split(
        SARIMAXModel(cfg),
        X_train,
        y_train,
        X_test,
        y_test,
        timestamps,
    )

    # --------------------------------------------------
    # GARCH-X
    # --------------------------------------------------
    (
        X_mean_train,
        X_var_train,
        y_train,
        X_mean_test,
        X_var_test,
        y_test,
        timestamps,
    ) = load_parametric_data(
        cfg=cfg,
        target=target,
        model="garch",
        verbose=False,
    )

    results["GARCH-X"] = _run_parametric_model_fixed_split(
        GARCHModel(cfg),
        X_mean_train,
        y_train,
        X_mean_test,
        y_test,
        timestamps,
        X_var_train=X_var_train,
    )

    # --------------------------------------------------
    # LSTM
    # --------------------------------------------------
    results["LSTM"] = _run_lstm_fixed_split(
        cfg=cfg,
        target=target,
        force_retrain=force_retrain,
    )

    # --------------------------------------------------
    # Rolling / Expanding ARX
    # --------------------------------------------------
    rolling_res = run_rolling_arx(
        cfg=cfg,
        target=target,
        run_if_missing=False,
        verbose=True,
    )

    if rolling_res is not None:
        results["RollingARX"] = rolling_res

    return results


# ======================================================
# ENTRYPOINT (optional)
# ======================================================

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    target = cfg.data.dataframe.target_col
    results = run_comparison(cfg=cfg, target=target, force_retrain=False)

    for name, res in results.items():
        print(
            f"{name:12s} | "
            f"RMSE={res['rmse']:.4f} | "
            f"MAE={res['mae']:.4f} | "
            f"time={res['train_time']:.1f}s"
        )


if __name__ == "__main__":
    main()
