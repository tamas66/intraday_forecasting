# src/train.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import time
from omegaconf import DictConfig, OmegaConf

from data import load_dataset, Seq2SeqDataset
from models.garchx import garch_from_hydra
from models.lstm import lstm_from_hydra, quantile_loss, spike_bce_loss


# ======================================================
# Helpers: paths, dates, schedules
# ======================================================

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _get_model_run_dir(cfg: DictConfig, target: str, model_name: str) -> Path:
    """
    Required by you:
      cfg.data.paths.models_dir/{target}/{model_name}
    """
    run_name = cfg.get("run_name", None)
    if run_name:
        root = Path(cfg.data.paths.models_dir) / target / model_name / str(run_name) / f"horizon_{cfg.horizon}/"
    else:
        root = Path(cfg.data.paths.models_dir) / target / model_name / f"horizon_{cfg.horizon}/"
    return _ensure_dir(root)



def _get_split_dates(cfg: DictConfig) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Prefer config if present, else fall back to current hard-coded split.
    Returns UTC-aware pandas Timestamps.
    """
    splits = cfg.get("splits", {})
    train_end = splits.get("train_end_date", "2024-12-31 23:00")
    test_start = splits.get("test_start_date", "2025-01-01 00:00")

    # Localize naive strings to UTC (your data index is UTC)
    return (pd.Timestamp(train_end, tz="UTC"), pd.Timestamp(test_start, tz="UTC"))


def _month_starts_between(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    start = start.normalize()
    end = end.normalize()
    months = pd.date_range(start=start, end=end, freq="MS")
    return [pd.Timestamp(m) for m in months]


def _forecast_origins(test_index: pd.DatetimeIndex, *, horizon: int) -> pd.DatetimeIndex:
    """
    Schedule rule:
      - if H <= 4: forecast every hour
      - else: forecast daily at 12:00 UTC
    """
    if horizon <= 4:
        return test_index
    return test_index[test_index.hour == 12]


def _window_slice(df: pd.DataFrame, end: pd.Timestamp, window_days: int) -> pd.DataFrame:
    start = end - pd.Timedelta(days=window_days)
    return df[(df.index >= start) & (df.index < end)].copy()


def _resolve_target(cfg: DictConfig) -> Literal["level", "spread"]:
    """
    Tries common locations for target; defaults to 'level'.
    """
    # Prefer explicit cfg.target if you have it
    t = cfg.get("target", None)
    if t in ("level", "spread"):
        return t
    # Or cfg.dataset.target if you use that pattern
    ds = cfg.get("dataset", {})
    if isinstance(ds, dict):
        t2 = ds.get("target", None)
        if t2 in ("level", "spread"):
            return t2
    return "level"


def _resolve_window_days(cfg: DictConfig) -> int:
    """
    Rolling training window length.
    Priority:
      1) cfg.model.rolling.window_days (per-model)
      2) cfg.rolling.window_days (global)
      3) 365
    """
    if "rolling" in cfg.model and "window_days" in cfg.model.rolling:
        return int(cfg.model.rolling.window_days)
    if "rolling" in cfg and "window_days" in cfg.rolling:
        return int(cfg.rolling.window_days)
    return 365


# ======================================================
# LSTM: training loop (rolling-friendly)
# ======================================================

def _train_lstm_one_window(
    *,
    model_cfg: DictConfig,
    model: torch.nn.Module,
    train_df: pd.DataFrame,
) -> torch.nn.Module:
    device = torch.device(model_cfg.runtime.device)
    model.to(device)
    model.train()

    lookback = int(model_cfg.data.lookback_length)
    horizon = int(model_cfg.data.forecast_horizon)
    batch_size = int(model_cfg.data.batch_size)

    past_feats = list(model_cfg.data.past_features)
    fut_feats = list(model_cfg.data.known_future_features)

    ds = Seq2SeqDataset(
        df=train_df,
        past_features=past_feats,
        future_features=fut_feats,
        lookback=lookback,
        horizon=horizon,
        upper_threshold=float(model_cfg.spikes.upper_threshold),
        lower_threshold=float(model_cfg.spikes.lower_threshold),
    )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    lr = float(model_cfg.training.learning_rate)
    epochs = int(model_cfg.training.num_epochs)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    taus = list(model_cfg.architecture.outputs.quantiles)
    use_spike = bool(model_cfg.architecture.outputs.spike_probability)
    spike_w = float(model_cfg.training.get("loss", {}).get("spike_weight", 1.0))

    for _ in range(epochs):
        for enc_x, dec_x, y, spike in loader:
            enc_x = enc_x.to(device)
            dec_x = dec_x.to(device)
            y = y.to(device)
            spike = spike.to(device)

            q_pred, spike_prob = model(enc_x, dec_x)

            loss_q = quantile_loss(y=y, q=q_pred, quantiles=taus)

            if use_spike and spike_prob is not None:
                # spike: (B,H,2) => train "any spike" as union by default
                spike_any = torch.clamp(spike.sum(dim=-1), 0, 1)  # (B,H)
                loss_s = spike_bce_loss(spike_prob=spike_prob, spike_target=spike_any)
                loss = loss_q + spike_w * loss_s
            else:
                loss = loss_q

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    return model


@torch.no_grad()
def _predict_lstm_one_origin(
    *,
    model_cfg: DictConfig,
    model: torch.nn.Module,
    df: pd.DataFrame,
    origin: pd.Timestamp,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      q_pred: (H, Q)
      spike_prob: (H,) (NaN if disabled)
    """
    device = torch.device(model_cfg.runtime.device)
    model.to(device)
    model.eval()

    lookback = int(model_cfg.data.lookback_length)
    horizon = int(model_cfg.data.forecast_horizon)

    past_feats = list(model_cfg.data.past_features)
    fut_feats = list(model_cfg.data.known_future_features)

    hist = df.loc[:origin].iloc[-lookback:]
    if len(hist) < lookback:
        raise ValueError(f"Not enough history at origin {origin} for lookback={lookback}")

    fut = df.loc[origin:].iloc[:horizon]
    if len(fut) < horizon:
        raise ValueError(f"Not enough future rows at origin {origin} for horizon={horizon}")

    enc_x = torch.tensor(hist[past_feats].values, dtype=torch.float32).unsqueeze(0).to(device)
    dec_x = torch.tensor(fut[fut_feats].values, dtype=torch.float32).unsqueeze(0).to(device)

    out = model.predict(enc_x, dec_x)
    q = out["quantiles"].squeeze(0).detach().cpu().numpy()  # (H,Q)

    spike = out.get("spike_prob", None)
    if spike is None:
        spike_np = np.full((horizon,), np.nan, dtype=float)
    else:
        spike_np = spike.squeeze(0).detach().cpu().numpy()

    return q, spike_np


# ======================================================
# Main rolling runner
# ======================================================

def run_rolling_training(
    *,
    cfg: DictConfig,
    model_name: Literal["garch", "lstm"],
    target: Literal["level", "spread"] = "level",
    verbose: bool = True,
) -> Path:
    """
    Rolling training + forecasting.
    Stores EVERYTHING:
      - LSTM: quantiles + spike_prob
      - GARCH: samples
      - y_true aligned per (origin, horizon)
    """
    # -----------------------------
    # Load full dataset once
    # -----------------------------
    file_path = f"{cfg.data.paths.final_data_dir}/{cfg.dataset}.parquet"
    df = load_dataset(cfg=cfg, file_path=file_path, target=target, drop_na=True, verbose=verbose)

    train_end_date, test_start_date = _get_split_dates(cfg)
    df_test_full = df[df.index >= test_start_date].copy()

    # Active model config is ALWAYS cfg.model (Hydra composes model=garch or model=lstm)
    model_cfg = cfg.model

    # Horizon from active model config
    if model_name == "lstm":
        horizon = int(model_cfg.data.forecast_horizon)
    else:
        horizon = int(model_cfg.forecast.horizon)

    origins = _forecast_origins(df_test_full.index, horizon=horizon)

    # Monthly refit dates
    refit_dates = _month_starts_between(test_start_date, df_test_full.index.max())
    refit_dates = [d for d in refit_dates if d >= test_start_date]
    if len(refit_dates) == 0:
        refit_dates = [test_start_date.normalize()]

    window_days = _resolve_window_days(cfg)

    # Output dir
    run_dir = _get_model_run_dir(cfg, target=target, model_name=model_name)
    _ensure_dir(run_dir)

    # Save config snapshot
    OmegaConf.save(cfg, run_dir / "config_snapshot.yaml")

    rows: List[Dict] = []
    refit_rows: List[Dict] = []
    dist_dir = _ensure_dir(run_dir / "distributions")

    if verbose:
        print(f"[{model_name}] rolling window_days={window_days}, horizon={horizon}")
        print(f"[{model_name}] refits={len(refit_dates)}, origins={len(origins)}")

    lstm_model: Optional[torch.nn.Module] = None

    for i, refit_date in enumerate(refit_dates):
        next_refit = (
            refit_dates[i + 1]
            if i + 1 < len(refit_dates)
            else (df_test_full.index.max() + pd.Timedelta(hours=1))
        )

        df_hist = df[df.index < refit_date].copy()
        df_win = _window_slice(df_hist, end=refit_date, window_days=window_days)

        if len(df_win) < 24 * 30:
            if verbose:
                print(f"[{model_name}] skip refit {refit_date}: insufficient window rows={len(df_win)}")
            continue

        seg_origins = origins[(origins >= refit_date) & (origins < next_refit)]
        if len(seg_origins) == 0:
            continue

        if verbose:
            print(f"[{model_name}] refit @ {refit_date} | origins in segment: {len(seg_origins)}")

        # -----------------------------
        # Fit + forecast
        # -----------------------------
        if model_name == "garch":
            garch = garch_from_hydra(cfg)

            y_train = df_win["y"].values
            mean_cols = list(cfg.model.mean.exogenous)
            X_mean = df_win[mean_cols].values if mean_cols else None

            fitted = False
            t0 = time.perf_counter()
            try:
                garch.fit(y=y_train, X_mean=X_mean)
                fitted = True
            except Exception as e:
                print(f"[GARCH] Skipping refit at {refit_date}: {e}")
            train_time = time.perf_counter() - t0
            refit_rows.append(
                {
                    "model": "garch",
                    "target": target,
                    "refit_date": refit_date,
                    "train_time_sec": train_time,
                    "window_days": window_days,
                    "horizon": horizon,
                    "n_origins": int(len(seg_origins)),
                    "status": "fitted" if fitted else "failed",
                }
            )
            if not fitted:
                continue

            M = int(cfg.model.forecast.simulation_draws)

            for origin in seg_origins:
                fut = df.loc[origin:].iloc[:horizon]
                if len(fut) < horizon:
                    continue

                X_future = fut[mean_cols].values if mean_cols else None
                t1 = time.perf_counter()
                if fitted:
                    samples = garch.forecast_samples(X_future=X_future, horizon=horizon, n_sim=M)
                else:
                    continue
                forecast_time = time.perf_counter() - t1
                y_true = fut["y"].values

                dist_path = dist_dir / f"samples_{origin.strftime('%Y%m%d%H')}.npy"
                np.save(dist_path, samples.astype(np.float32))

                for h in range(horizon):
                    rows.append(
                        {
                            "model": "garch",
                            "target": target,
                            "origin_time": origin,
                            "horizon": h + 1,
                            "target_time": fut.index[h],
                            "y_true": float(y_true[h]),
                            "dist_path": str(dist_path),
                            "dist_kind": "samples",
                            "meta_refit": refit_date,
                            "train_time_sec": float(train_time),
                            "pred_time_sec": float(forecast_time),
                        }
                    )

        elif model_name == "lstm":
            warm_start = bool(model_cfg.training.get("warm_start", False))

            if (lstm_model is None) or (not warm_start):
                lstm_model = lstm_from_hydra(cfg)  # should read cfg.model internally
            t0 = time.perf_counter()
            lstm_model = _train_lstm_one_window(model_cfg=model_cfg, model=lstm_model, train_df=df_win)
            train_time = time.perf_counter() - t0
            refit_rows.append(
                {
                    "model": "lstm",
                    "target": target,
                    "refit_date": refit_date,
                    "train_time_sec": train_time,
                    "window_days": window_days,
                    "horizon": horizon,
                    "n_origins": int(len(seg_origins)),
                    "status": "ok",
                    "warm_start": bool(warm_start),
                }
            )
            taus = list(model_cfg.architecture.outputs.quantiles)

            for origin in seg_origins:
                fut = df.loc[origin:].iloc[:horizon]
                if len(fut) < horizon:
                    continue
                t1 = time.perf_counter()
                q_pred, spike_prob = _predict_lstm_one_origin(
                    model_cfg=model_cfg, model=lstm_model, df=df, origin=origin
                )
                pred_time = time.perf_counter() - t1
                y_true = fut["y"].values

                dist_path = dist_dir / f"quantiles_{origin.strftime('%Y%m%d%H')}.npz"
                np.savez_compressed(
                    dist_path,
                    quantiles=q_pred.astype(np.float32),
                    taus=np.array(taus, dtype=np.float32),
                    spike_prob=spike_prob.astype(np.float32),
                )

                for h in range(horizon):
                    rows.append(
                        {
                            "model": "lstm",
                            "target": target,
                            "origin_time": origin,
                            "horizon": h + 1,
                            "target_time": fut.index[h],
                            "y_true": float(y_true[h]),
                            "dist_path": str(dist_path),
                            "dist_kind": "quantiles",
                            "meta_refit": refit_date,
                            "train_time_sec": float(train_time),
                            "pred_time_sec": float(pred_time),
                        }
                    )
        else:
            raise ValueError("model_name must be 'garch' or 'lstm'")

    out_df = pd.DataFrame(rows)
    out_path = run_dir / "forecast_index.parquet"
    out_df.to_parquet(out_path, index=False)
    if refit_rows:
        pd.DataFrame(refit_rows).to_parquet(run_dir / "refit_times.parquet", index=False)
    if verbose:
        print(f"[{model_name}] wrote {len(out_df)} rows -> {out_path}")

    return run_dir


# ======================================================
# Entry: train selected model from Hydra composition
# ======================================================

def train(cfg: DictConfig) -> None:
    """
    Hydra composes one active model into cfg.model via:
      python src/train.py model=garch
      python src/train.py model=lstm

    We run ONLY the active model config (recommended for reproducibility).
    """
    target = _resolve_target(cfg)

    if "model" not in cfg or "name" not in cfg.model:
        raise ValueError(
            "Hydra did not compose an active model into cfg.model. "
            "Check configs/config.yaml defaults and run with e.g. model=garch."
        )

    model_name = str(cfg.model.name).lower()
    if model_name not in ("garch", "lstm"):
        raise ValueError(f"cfg.model.name must be 'garch' or 'lstm', got: {cfg.model.name}")

    run_rolling_training(cfg=cfg, model_name=model_name, target=target)


# ======================================================
# Script entry point (Hydra)
# ======================================================

import hydra


@hydra.main(
    version_base=None,
    config_path="../configs",   # NOTE: relative to this file (src/train.py)
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
