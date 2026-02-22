# src/train.py
from __future__ import annotations

from pathlib import Path
from typing import Literal, Tuple
import time

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
import hydra

from data import (
    load_dataset,
    load_garch_data,
    iter_rolling_windows,
    Seq2SeqJumpDataset,
    Seq2SeqQuantileDataset,
)
from models.garchx import garch_from_hydra
from models.lstm import lstm_from_hydra, jump_lstm_from_hydra, quantile_loss, spike_bce_loss


# ======================================================
# Helpers
# ======================================================

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _get_run_dir(cfg: DictConfig, target: str, model_name: str) -> Path:
    run_name = cfg.get("run_name", None)
    base = Path(cfg.data.paths.models_dir) / target / model_name
    root = base / str(run_name) / f"horizon_{cfg.horizon}" if run_name else base / f"horizon_{cfg.horizon}"
    return _ensure_dir(root)


def _get_split_dates(cfg: DictConfig) -> Tuple[str, str]:
    splits = cfg.get("splits", {})
    train_end  = splits.get("train_end_date",  "2024-12-31 23:30")
    test_start = splits.get("test_start_date", "2025-01-01 00:00")
    return train_end, test_start


def _resolve_target(cfg: DictConfig) -> Literal["level", "spread", "wavelet", "jump"]:
    t = cfg.get("target", "level")
    if t not in ("level", "spread", "wavelet", "jump"):
        raise ValueError(f"Unknown target '{t}'. Must be level | spread | wavelet | jump.")
    return t


# ======================================================
# LSTM TRAINING
# ======================================================

def _train_lstm(
    *,
    model_cfg: DictConfig,
    model: torch.nn.Module,
    train_df: pd.DataFrame,
    target: str,
) -> torch.nn.Module:

    device = torch.device(model_cfg.runtime.device)
    model.to(device).train()

    lookback   = int(model_cfg.data.lookback_length)
    horizon    = int(model_cfg.data.forecast_horizon)
    batch_size = int(model_cfg.data.batch_size)
    past_feats = list(model_cfg.data.past_features)
    fut_feats  = list(model_cfg.data.known_future_features)
    taus       = list(model_cfg.architecture.outputs.quantiles)

    DatasetCls = Seq2SeqJumpDataset if target == "jump" else Seq2SeqQuantileDataset
    ds = DatasetCls(df=train_df, past_features=past_feats, future_features=fut_feats,
                    lookback=lookback, horizon=horizon)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=float(model_cfg.training.learning_rate))

    for _ in range(int(model_cfg.training.num_epochs)):
        for batch in loader:
            enc_x, dec_x = batch[0].to(device), batch[1].to(device)

            if target == "jump":
                smooth_y, jump_flag, jump_size = batch[2].to(device), batch[3].to(device), batch[4].to(device)
                smooth_q, jump_prob, jump_size_pred = model(enc_x, dec_x)

                loss_q    = quantile_loss(smooth_y, smooth_q, taus)
                loss_prob = spike_bce_loss(jump_prob, jump_flag)
                jump_mask = jump_flag > 0.5
                loss_jump = (
                    torch.mean((jump_size_pred[jump_mask] - jump_size[jump_mask]) ** 2)
                    if jump_mask.sum() > 0
                    else torch.tensor(0.0, device=device)
                )
                loss = loss_q + loss_prob + loss_jump
            else:
                y = batch[2].to(device)
                q_pred, _ = model(enc_x, dec_x)
                loss = quantile_loss(y, q_pred, taus)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    return model


# ======================================================
# LSTM PREDICTION
# ======================================================

@torch.no_grad()
def _predict_lstm(
    *,
    model_cfg: DictConfig,
    model: torch.nn.Module,
    df: pd.DataFrame,
    origin: pd.Timestamp,
    target: str,
):
    device   = torch.device(model_cfg.runtime.device)
    lookback = int(model_cfg.data.lookback_length)
    horizon  = int(model_cfg.data.forecast_horizon)
    past_feats = list(model_cfg.data.past_features)
    fut_feats  = list(model_cfg.data.known_future_features)

    model.to(device).eval()

    hist  = df.loc[:origin].iloc[-lookback:]
    fut   = df.loc[origin:].iloc[:horizon]
    enc_x = torch.tensor(hist[past_feats].values, dtype=torch.float32).unsqueeze(0).to(device)
    dec_x = torch.tensor(fut[fut_feats].values,   dtype=torch.float32).unsqueeze(0).to(device)

    if target == "jump":
        smooth_q, jump_prob, jump_size = model(enc_x, dec_x)
        return (
            smooth_q.squeeze(0).cpu().numpy(),
            jump_prob.squeeze(0).cpu().numpy(),
            jump_size.squeeze(0).cpu().numpy(),
        )
    else:
        q, _ = model(enc_x, dec_x)
        return q.squeeze(0).cpu().numpy()


# ======================================================
# MAIN ROLLING RUNNER
# ======================================================

def run_rolling_training(
    *,
    cfg: DictConfig,
    model_name: Literal["garch", "lstm"],
    target: Literal["level", "spread", "wavelet", "jump"],
) -> Path:

    if model_name == "garch" and target == "jump":
        raise ValueError("GARCH is not used for the jump target.")

    run_dir  = _get_run_dir(cfg, target, model_name)
    dist_dir = _ensure_dir(run_dir / "distributions")
    OmegaConf.save(cfg, run_dir / "config_snapshot.yaml")

    train_end, test_start = _get_split_dates(cfg)
    rows: list[dict] = []

    # --------------------------------------------------
    # GARCH
    # --------------------------------------------------
    if model_name == "garch":
        bundle  = load_garch_data(cfg=cfg, target=target)
        model   = garch_from_hydra(cfg)
        horizon = int(cfg.model.variance.p)  # reuse horizon from cfg
        horizon = int(cfg.horizon)
        refit_every = int(cfg.get("refit_every", 48))  # default: daily
        for y_fit, X_fit, X_fut, timestamps in iter_rolling_windows(
            bundle, horizon=horizon, refit_every=refit_every
        ):
            t0 = time.perf_counter()
            model.fit(y_fit, X_mean=X_fit)
            train_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            samples = model.forecast_samples(
                X_future=X_fut,
                horizon=horizon,
                n_sim=int(cfg.get("n_sim", 1000)),
            )
            pred_time = time.perf_counter() - t0

            origin = timestamps[0]
            dist_path = dist_dir / f"samples_{origin.strftime('%Y%m%d%H%M')}.npz"
            np.savez_compressed(dist_path, samples=samples)

            y_true_window = bundle.y_test[
                np.where(bundle.test_timestamps == origin)[0][0] :
                np.where(bundle.test_timestamps == origin)[0][0] + horizon
            ]

            for h in range(horizon):
                rows.append(dict(
                    model=model_name,
                    target=target,
                    origin_time=origin,
                    horizon=h + 1,
                    target_time=timestamps[h],
                    y_true=float(y_true_window[h]),
                    dist_path=str(dist_path),
                    dist_kind="samples",
                    da_price=float(X_fut["da_price"].values[h]),
                    wav_detail=float(X_fut["wav_detail"].values[h]),  # or whatever the column is named
                    train_time=train_time,
                    pred_time=pred_time / horizon,       # normalise to per-step
                    refit_frequency=refit_every,
                ))

    # --------------------------------------------------
    # LSTM
    # --------------------------------------------------
    elif model_name == "lstm":
        file_path = f"{cfg.data.paths.final_data_dir}/{cfg.dataset}.parquet"
        train_df, test_df = load_dataset(cfg=cfg, file_path=file_path, target=target, train_end_date=train_end, test_start_date=test_start, drop_na=True)
        print(f"[DEBUG] df.index.tz={test_df.index.tz}, test_start='{test_start}'")
        print(f"[DEBUG] df after test_start: {(test_df.index >= pd.Timestamp(test_start, tz='UTC')).sum()} rows")
        horizon = int(cfg.model.data.forecast_horizon)
        model_cfg = cfg.model
        taus = np.array(model_cfg.architecture.outputs.quantiles)
        
        lstm_model = jump_lstm_from_hydra(cfg) if target == "jump" else lstm_from_hydra(cfg)

        for origin in test_df.index:
            df_hist = train_df[train_df.index < origin]
            if len(df_hist) < 24 * 30:
                continue

            fut = test_df.loc[origin:].iloc[:horizon]
            if len(fut) < horizon:
                continue

            t0 = time.perf_counter()
            lstm_model = _train_lstm(
                model_cfg=model_cfg, model=lstm_model, train_df=df_hist, target=target
            )
            train_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            preds = _predict_lstm(
                model_cfg=model_cfg, model=lstm_model, df=test_df, origin=origin, target=target
            )
            pred_time = time.perf_counter() - t0

            if target == "jump":
                smooth_q, jump_prob, jump_size = preds
                dist_path = dist_dir / f"jump_{origin.strftime('%Y%m%d%H%M')}.npz"
                np.savez_compressed(dist_path, smooth_quantiles=smooth_q,
                                    jump_prob=jump_prob, jump_size=jump_size, taus=taus)
                dist_kind = "jump"
            else:
                dist_path = dist_dir / f"quantiles_{origin.strftime('%Y%m%d%H%M')}.npz"
                np.savez_compressed(dist_path, quantiles=preds, taus=taus)
                dist_kind = "quantiles"

            for h in range(horizon):
                rows.append(dict(
                    model=model_name,
                    target=target,
                    origin_time=origin,
                    horizon=h + 1,
                    target_time=fut.index[h],
                    y_true=float(fut["y"].values[h]),
                    dist_path=str(dist_path),
                    dist_kind=dist_kind,
                    da_price=float(fut["da_price"].values[h]),
                    wav_detail=float(fut["wav_detail"].values[h]),  # or whatever the column is named
                    train_time=train_time,
                    pred_time=pred_time,                 # already per-origin
                    refit_frequency=1,                   # retrained every step
                ))
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Must be 'garch' or 'lstm'.")
    print(f"[DEBUG] model_name='{model_name}', target='{target}'")
    print(f"[DEBUG] cfg.model.name='{cfg.model.name}'")
 
    # --------------------------------------------------
    # Save forecast index
    # --------------------------------------------------
    out_df = pd.DataFrame(rows)
    print(out_df.head())
    out_df["cost_per_forecast"] = (out_df["train_time"] / out_df["refit_frequency"]) + out_df["pred_time"]
    out_path = run_dir / "forecast_index.parquet"
    out_df.to_parquet(out_path, index=False)

    return run_dir

# ======================================================
# ENTRY
# ======================================================

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    run_rolling_training(
        cfg=cfg,
        model_name=str(cfg.model.name),
        target=_resolve_target(cfg),
    )


if __name__ == "__main__":
    main()