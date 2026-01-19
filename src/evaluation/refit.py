import json
import time
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd

from config import FINAL_DATA_DIR, RESULTS_DIR
from data import load_parametric_dataset
from evaluation.metrics import compute_metrics
from modelling.model_config import RollingARXConfig as cfg
from modelling.model_config import CORE_MEAN_EXOG_COLS, ALL_MEAN_EXOG_COLS, VARIANCE_EXOG_COLS


def _prepare_arx_xy(
    target: str = "level",
    exog_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, List[str]]:
    """
    Use the same feature-engineered parametric dataframe to build a stable
    ARX design for rolling re-estimation.
    """
    file_path = FINAL_DATA_DIR / "core_2020-01-01_2025-12-31.parquet"

    df = load_parametric_dataset(
        file_path=str(file_path),
        target=target,
        drop_na=True,
        train_end_date=None,
        test_start_date=None,
        verbose=False,
    )

    # Minimal, stable ARX regressors
    default_cols = ["y_lag1", "y_lag24"]

    use_cols = exog_cols if exog_cols is not None else default_cols
    use_cols = [c for c in use_cols if c in df.columns]

    y_s = pd.to_numeric(df["y"], errors="coerce")
    X_df = df[use_cols].apply(pd.to_numeric, errors="coerce")

    valid = pd.concat([y_s, X_df], axis=1).dropna()
    y = valid["y"].astype(float).values
    X = valid.drop(columns=["y"]).astype(float).values
    ts = valid.index

    # Add intercept explicitly (OLS)
    X = np.column_stack([np.ones(len(X)), X])
    used = ["const"] + use_cols

    return y, X, ts, used


def _ols_fit_predict_1step(X_train: np.ndarray, y_train: np.ndarray, x_next: np.ndarray) -> float:
    """
    Fit OLS y = Xb and predict for one row x_next.
    Uses np.linalg.lstsq (robust and fast).
    """
    beta, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
    return float(x_next @ beta)


def run_rolling_arx(
    target: str = "level",
    min_train_size: int = 24 * 365 * 2,     # 2-year burn-in
    refit_every: int = 1,                   # 1 = re-estimate each step
    window_size: Optional[int] = None,      # None => expanding; else rolling fixed length
    exog_cols: Optional[List[str]] = None,  # default: ["y_lag1","y_lag24"]
    output_prefix: str = "rolling_arx",
    verbose: bool = True,
) -> Dict:
    """
    Rolling / expanding ARX benchmark (constantly re-estimated).

    - Expanding window if window_size is None
    - Rolling fixed window if window_size is provided

    Forecast: 1-step-ahead.
    """
    y, X, ts, used_cols = _prepare_arx_xy(target=target, exog_cols=exog_cols)

    n = len(y)
    if n <= min_train_size + 1:
        raise ValueError(f"Not enough data: n={n}, min_train_size={min_train_size}")

    preds = []
    trues = []
    ts_out = []
    statuses = []

    total_fit_time = 0.0
    n_refits = 0

    start_wall = time.time()

    beta_cached = None
    last_fit_t = None

    for t in range(min_train_size, n):
        do_refit = (beta_cached is None) or ((t - min_train_size) % refit_every == 0)

        if do_refit:
            # determine training slice
            if window_size is None:
                start_idx = 0
            else:
                start_idx = max(0, t - window_size)

            X_train = X[start_idx:t]
            y_train = y[start_idx:t]

            fit_start = time.time()
            beta_cached, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
            total_fit_time += time.time() - fit_start

            n_refits += 1
            last_fit_t = t

            if verbose and (n_refits == 1 or n_refits % 500 == 0):
                mode = "expanding" if window_size is None else f"rolling({window_size})"
                print(
                    f"[Rolling ARX] refit #{n_refits} at t={t} ({ts[t]}) "
                    f"mode={mode} total_fit_time={total_fit_time:.1f}s"
                )

        # 1-step ahead predict
        try:
            y_hat = float(X[t] @ beta_cached)
            preds.append(y_hat)
            trues.append(float(y[t]))
            ts_out.append(ts[t])
            statuses.append("ok")
        except Exception as e:
            statuses.append(f"fail:{type(e).__name__}")
            continue

    elapsed = time.time() - start_wall

    out_df = pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(ts_out),
            "y_true": np.array(trues, dtype=float),
            "y_pred": np.array(preds, dtype=float),
            "status": statuses[: len(ts_out)],
        }
    ).set_index("timestamp")

    metrics = compute_metrics(out_df["y_true"].values, out_df["y_pred"].values) if len(out_df) else {}

    meta = {
        "model": "RollingARX",
        "target": target,
        "evaluation": "expanding_refit" if window_size is None else "rolling_refit",
        "min_train_size": int(min_train_size),
        "refit_every": int(refit_every),
        "window_size": None if window_size is None else int(window_size),
        "used_cols": used_cols,
        "n_total_obs": int(n),
        "n_forecasts": int(len(out_df)),
        "n_refits": int(n_refits),
        "total_fit_time_sec": float(total_fit_time),
        "wall_time_sec": float(elapsed),
        "rmse": float(metrics.get("rmse", np.nan)),
        "mae": float(metrics.get("mae", np.nan)),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"{output_prefix}_{target}.csv"
    meta_path = RESULTS_DIR / f"{output_prefix}_{target}_meta.json"

    out_df.to_csv(csv_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"\n[Rolling ARX] Done. Forecasts: {len(out_df)}")
        print(f"[Rolling ARX] RMSE: {meta['rmse']:.4f} | MAE: {meta['mae']:.4f}")
        print(f"[Rolling ARX] total_fit_time_sec: {total_fit_time:.2f} | wall_time_sec: {elapsed:.2f}")
        print(f"[Rolling ARX] Saved CSV: {csv_path}")
        print(f"[Rolling ARX] Saved meta: {meta_path}")

    return meta


if __name__ == "__main__":
    # Defaults: expanding window, refit every step (true “constantly re-estimated”)
    # For faster runs: refit_every=24 (daily)
    run_rolling_arx(
        target="level",
        min_train_size=cfg.min_train_size,
        refit_every=cfg.refit_every,
        window_size=cfg.window_size,
        exog_cols=cfg.exog_cols,
        verbose=True,
    )
