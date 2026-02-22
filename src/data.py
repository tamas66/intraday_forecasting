from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from typing import Literal, Optional, Tuple, Generator
from omegaconf import DictConfig


# ======================================================
# SHARED DATASET BUILDER
# ======================================================

def load_dataset(
    *,
    cfg: DictConfig,
    file_path: Optional[str] = None,
    target: Literal["level", "spread", "wavelet", "jump"] = "level",
    drop_na: bool = True,
    train_end_date: Optional[str] = None,
    test_start_date: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare a generic modelling dataframe for parametric electricity price models.
    No model-specific assumptions are made here.
    """
    df = pd.read_parquet(file_path)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    df = df.drop(columns=["gen_oil"])
    if verbose:
        print(f"Loaded {len(df)} rows from {df.index.min()} to {df.index.max()}")

    target_map = {
        "level":   "intraday_wap",
        "spread":  "id_da_spread",
        "wavelet": "wav_trend",
        "jump":    "smooth_price",
    }

    if target not in target_map:
        raise ValueError("target must be 'level', 'spread', 'wavelet', or 'jump'")

    base_col = target_map[target]

    if base_col not in df.columns:
        raise ValueError(f"Column '{base_col}' not found in dataframe.")

    df["y"] = df[base_col]

    for lag in [1, 24]:
        df[f"{base_col}_lag{lag}"] = df[base_col].shift(lag)

    df["abs_return"] = df[base_col].diff().abs()
    df["abs_return_lag1"] = df["abs_return"].shift(1)

    if drop_na:
        nans = df.isna().sum().sort_values(ascending=False)
        print(f"NaN counts per column:\n{nans[nans > 0]}")
    print(f"After dropna: {len(df)} rows from {df.index.min()} to {df.index.max()}")
    # Normalise index timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    if train_end_date and test_start_date:
        train_end_ts  = pd.Timestamp(train_end_date,  tz="UTC")
        test_start_ts = pd.Timestamp(test_start_date, tz="UTC")
        train_df = df[df.index <= train_end_ts].copy()
        test_df  = df[df.index >= test_start_ts].copy()
        return train_df, test_df

    return df


def _file_path(cfg: DictConfig) -> str:
    return f"{cfg.data.paths.final_data_dir}/{cfg.dataset}.parquet"


# ======================================================
# PARAMETRIC LOADER
# ======================================================

def load_parametric_data(
    *,
    cfg: DictConfig,
    target: Literal["level", "spread", "wavelet", "jump"] = "level",
    verbose: bool = True,
):
    train_df, test_df = load_dataset(
        cfg=cfg,
        file_path=_file_path(cfg),
        target=target,
        train_end_date="2024-12-31 23:30",
        test_start_date="2025-01-01 00:00",
        verbose=verbose,
    )

    y_train = train_df["y"].values
    y_test  = test_df["y"].values
    X_train = train_df.drop(columns=["y"])
    X_test  = test_df.drop(columns=["y"])

    return X_train, y_train, X_test, y_test, test_df.index


# ======================================================
# GARCH LOADER
# ======================================================

# Parsimonious default mean regressors per target.
# Override via cfg.model.garchx.mean_features.
_GARCH_MEAN_FEATURES: dict[str, list[str]] = {
    "level":   ["da_price", "hour_sin", "hour_cos", "dow_sin", "dow_cos"],
    "spread":  ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "net_imbalance"],
    "wavelet": ["da_price", "hour_sin", "hour_cos"],
}


@dataclass
class GarchDataBundle:
    """
    Everything MSGarchEVT.fit() and .forecast_samples() need.

    Fitting  : model.fit(y=y_train, X_mean=X_mean_train)
    Forecast : model.forecast_samples(X_future=X_fut, horizon=H)
    Evaluate : y_test + test_timestamps + test_df (for WAP reconstruction)
    """
    y_train:        np.ndarray
    X_mean_train:   Optional[np.ndarray]
    y_test:         np.ndarray
    X_mean_test:    Optional[np.ndarray]
    test_timestamps: pd.DatetimeIndex
    test_df:        pd.DataFrame
    target:         str
    feature_names:  Optional[list[str]]


def load_garch_data(
    *,
    cfg: DictConfig,
    target: Literal["level", "spread", "wavelet"] = "level",
    verbose: bool = True,
) -> GarchDataBundle:
    """
    Load data for MSGarchEVT. Jump target is excluded by design.
    """
    if target == "jump":
        raise ValueError(
            "GARCH is not used for the jump target. "
            "Choose 'level', 'spread', or 'wavelet'."
        )

    train_df, test_df = load_dataset(
        cfg=cfg,
        file_path=_file_path(cfg),
        target=target,
        train_end_date="2024-12-31 23:30",
        test_start_date="2025-01-01 00:00",
        verbose=verbose,
    )

    y_train = train_df["y"].values.astype(float)
    y_test  = test_df["y"].values.astype(float)

    # Resolve mean features: config override > defaults > None
    try:
        feature_names = list(cfg.model.garchx.mean_features)
    except Exception:
        feature_names = _GARCH_MEAN_FEATURES.get(target, [])

    if feature_names:
        missing = [f for f in feature_names if f not in train_df.columns]
        if missing and verbose:
            print(f"[GARCHLoader] Dropping unavailable features: {missing}")
        feature_names = [f for f in feature_names if f in train_df.columns]

    if feature_names:
        X_mean_train = train_df[feature_names].values.astype(float)
        X_mean_test  = test_df[feature_names].values.astype(float)
        if verbose:
            print(f"[GARCHLoader] target={target} | mean features: {feature_names}")
    else:
        X_mean_train = X_mean_test = None
        if verbose:
            print(f"[GARCHLoader] target={target} | no mean features")

    if verbose:
        print(f"[GARCHLoader] train={len(y_train)} obs | test={len(y_test)} obs")

    return GarchDataBundle(
        y_train=y_train,
        X_mean_train=X_mean_train,
        y_test=y_test,
        X_mean_test=X_mean_test,
        test_timestamps=test_df.index,
        test_df=test_df,
        target=target,
        feature_names=feature_names or None,
    )


def iter_rolling_windows(
    bundle: GarchDataBundle,
    horizon: int,
    refit_every: int,
) -> Generator[Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], pd.DatetimeIndex], None, None]:
    """
    Yield expanding-window slices for rolling GARCH estimation.

    Yields: (y_fit, X_fit, X_fut, timestamps)
        y_fit      : passed to model.fit(y=...)
        X_fit      : passed to model.fit(X_mean=...)
        X_fut      : passed to model.forecast_samples(X_future=...)
        timestamps : DatetimeIndex for the forecast window

    Parameters
    ----------
    horizon      : forecast horizon in HH periods (2=1h, 8=4h, 24=12h)
    refit_every  : refit frequency in HH periods (48=daily, 336=weekly)
    """
    T_test = len(bundle.y_test)

    for test_start in range(0, T_test - horizon, refit_every):
        test_end = test_start + horizon

        y_fit = np.concatenate([bundle.y_train, bundle.y_test[:test_start]])

        X_fit = X_fut = None
        if bundle.X_mean_train is not None:
            X_fit = np.concatenate([bundle.X_mean_train, bundle.X_mean_test[:test_start]], axis=0)
            X_fut = bundle.X_mean_test[test_start:test_end]

        yield y_fit, X_fit, X_fut, bundle.test_timestamps[test_start:test_end]


# ======================================================
# SEQ2SEQ LSTM DATASETS
# ======================================================

class Seq2SeqJumpDataset(Dataset):
    """Seq2Seq dataset for jump-diffusion LSTM forecasting."""

    def __init__(
        self,
        df: pd.DataFrame,
        past_features: list[str],
        future_features: list[str],
        lookback: int,
        horizon: int,
    ):
        self.df = df.reset_index(drop=True)
        self.past_features = past_features
        self.future_features = future_features
        self.L = lookback
        self.H = horizon

        for col in ["smooth_price", "jump_flag", "jump_component"]:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' missing from dataframe.")

        self.valid_idx = np.arange(self.L, len(df) - self.H)

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, i):
        t = self.valid_idx[i]
        enc        = self.df[self.past_features].iloc[t - self.L : t].values
        dec        = self.df[self.future_features].iloc[t : t + self.H].values
        smooth     = self.df["smooth_price"].iloc[t : t + self.H].values
        jump_flag  = self.df["jump_flag"].iloc[t : t + self.H].values
        jump_size  = self.df["jump_component"].iloc[t : t + self.H].values

        return (
            torch.tensor(enc,       dtype=torch.float32),
            torch.tensor(dec,       dtype=torch.float32),
            torch.tensor(smooth,    dtype=torch.float32),
            torch.tensor(jump_flag, dtype=torch.float32),
            torch.tensor(jump_size, dtype=torch.float32),
        )


class Seq2SeqQuantileDataset(Dataset):
    """Seq2Seq dataset for quantile LSTM forecasting."""

    def __init__(
        self,
        df: pd.DataFrame,
        past_features: list[str],
        future_features: list[str],
        lookback: int,
        horizon: int,
    ):
        self.df = df.reset_index(drop=True)
        self.past_features = past_features
        self.future_features = future_features
        self.L = lookback
        self.H = horizon

        if "y" not in df.columns:
            raise ValueError("Required column 'y' missing from dataframe.")

        self.valid_idx = np.arange(self.L, len(df) - self.H)

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, i):
        t = self.valid_idx[i]
        enc = self.df[self.past_features].iloc[t - self.L : t].values
        dec = self.df[self.future_features].iloc[t : t + self.H].values
        y   = self.df["y"].iloc[t : t + self.H].values

        return (
            torch.tensor(enc, dtype=torch.float32),
            torch.tensor(dec, dtype=torch.float32),
            torch.tensor(y,   dtype=torch.float32),
        )


# ======================================================
# LSTM LOADER
# ======================================================

def load_lstm_data(
    *,
    cfg: DictConfig,
    target: Literal["level", "spread", "wavelet", "jump"] = "level",
    verbose: bool = True,
) -> Tuple[DataLoader, DataLoader, pd.DatetimeIndex]:

    jump_target = target == "jump"

    train_df, test_df = load_dataset(
        cfg=cfg,
        file_path=_file_path(cfg),
        target=target,
        train_end_date="2024-12-31 23:30",
        test_start_date="2025-01-01 00:00",
        verbose=verbose,
    )

    lookback = cfg.model.lstm.data.lookback_length
    horizon  = cfg.model.lstm.data.forecast_horizon
    past     = cfg.model.lstm.data.past_features
    future   = cfg.model.lstm.data.known_future_features

    DatasetCls = Seq2SeqJumpDataset if jump_target else Seq2SeqQuantileDataset

    ds_train = DatasetCls(df=train_df, past_features=past, future_features=future, lookback=lookback, horizon=horizon)
    ds_test  = DatasetCls(df=test_df,  past_features=past, future_features=future, lookback=lookback, horizon=horizon)

    train_loader = DataLoader(ds_train, batch_size=cfg.model.lstm.data.batch_size, shuffle=True,  drop_last=True)
    test_loader  = DataLoader(ds_test,  batch_size=cfg.model.lstm.data.batch_size, shuffle=False, drop_last=False)

    test_timestamps = test_df.index[lookback : lookback + len(ds_test)]

    return train_loader, test_loader, test_timestamps