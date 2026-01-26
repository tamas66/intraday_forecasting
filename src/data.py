import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Literal, Optional, Tuple
from omegaconf import DictConfig


# ======================================================
# PARAMETRIC DATASET BUILDER (MODEL-AGNOSTIC)
# ======================================================

def load_dataset(
    *,
    cfg: DictConfig,
    file_path: Optional[str] = None,
    target: Literal["level", "spread"] = "level",
    drop_na: bool = True,
    train_end_date: Optional[str] = None,
    test_start_date: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare a generic modelling dataframe for parametric electricity price models.
    No model-specific assumptions are made here.
    """
    if file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path, parse_dates=True, index_col=0)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()

    if verbose:
        print(f"Loaded {len(df)} rows from {df.index.min()} to {df.index.max()}")

    # --------------------------------------------------
    # Target construction
    # --------------------------------------------------
    df["id_da_spread"] = df["intraday_wap"] - df["da_price"]

    if target == "level":
        base_col = "intraday_wap"
    elif target == "spread":
        base_col = "id_da_spread"
    else:
        raise ValueError("target must be 'level' or 'spread'")
    df["y"] = df[base_col]
    for lag in [1, 24]:
        df[f"{base_col}_lag{lag}"] = df[base_col].shift(lag)

    df["abs_return"] = df[base_col].diff().abs()
    df["abs_return_lag1"] = df["abs_return"].shift(1)

    if target == "spread":
        df = df.drop(columns=["da_price"], errors="ignore")

    if drop_na:
        df = df.dropna()

    if train_end_date and test_start_date:
        train_df = df[df.index <= train_end_date].copy()
        test_df = df[df.index >= test_start_date].copy()
        return train_df, test_df

    return df


# ======================================================
# GENERIC PARAMETRIC LOADER
# ======================================================

def load_parametric_data(
    *,
    cfg: DictConfig,
    target: Literal["level", "spread"] = "level",
    verbose: bool = True,
):
    file_path = (
        f"{cfg.data.paths.final_data_dir}/"
        f"{cfg.dataset}.parquet"
    )

    train_df, test_df = load_dataset(
        cfg=cfg,
        file_path=file_path,
        target=target,
        train_end_date="2024-12-31 23:00",
        test_start_date="2025-01-01 00:00",
        verbose=verbose,
    )

    y_train = train_df["y"].values
    y_test = test_df["y"].values

    X_train = train_df.drop(columns=["y"])
    X_test = test_df.drop(columns=["y"])

    test_timestamps = test_df.index

    return X_train, y_train, X_test, y_test, test_timestamps


# ======================================================
# SEQ2SEQ LSTM DATASET (MULTIVARIATE, QUANTILE + SPIKES)
# ======================================================

class Seq2SeqDataset(Dataset):
    """
    Seq2Seq dataset for probabilistic LSTM forecasting.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        past_features: list[str],
        future_features: list[str],
        lookback: int,
        horizon: int,
        upper_threshold: float,
        lower_threshold: float,
        rolling_std_window: int = 168,
    ):
        self.df = df
        self.past_features = past_features
        self.future_features = future_features
        self.L = lookback
        self.H = horizon

        baseline = df.get("da_price", df["y"].rolling(24).mean())
        resid = df["y"] - baseline
        scale = resid.rolling(rolling_std_window).std()
        z = resid / scale

        self.spike_pos = (z > upper_threshold).astype(int)
        self.spike_neg = (z < lower_threshold).astype(int)

        self.valid_idx = np.arange(self.L, len(df) - self.H)

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, i):
        t = self.valid_idx[i]

        enc = self.df[self.past_features].iloc[t - self.L : t].values
        dec = self.df[self.future_features].iloc[t : t + self.H].values
        y = self.df["y"].iloc[t : t + self.H].values

        spike = np.stack(
            [
                self.spike_pos.iloc[t : t + self.H].values,
                self.spike_neg.iloc[t : t + self.H].values,
            ],
            axis=-1,
        )

        return (
            torch.tensor(enc, dtype=torch.float32),
            torch.tensor(dec, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(spike, dtype=torch.float32),
        )


# ======================================================
# LSTM DATA LOADER
# ======================================================

def load_lstm_data(
    *,
    cfg: DictConfig,
    target: Literal["level", "spread"] = "level",
    verbose: bool = True,
) -> Tuple[DataLoader, DataLoader, pd.DatetimeIndex]:

    file_path = (
        f"{cfg.data.paths.final_data_dir}/"
        f"{cfg.dataset}.parquet"
    )

    train_df, test_df = load_dataset(
        cfg=cfg,
        file_path=file_path,
        target=target,
        train_end_date="2024-12-31 23:00",
        test_start_date="2025-01-01 00:00",
        verbose=verbose,
    )

    lookback = cfg.model.lstm.data.lookback_length
    horizon = cfg.model.lstm.data.forecast_horizon

    ds_train = Seq2SeqDataset(
        df=train_df,
        past_features=cfg.model.lstm.data.past_features,
        future_features=cfg.model.lstm.data.known_future_features,
        lookback=lookback,
        horizon=horizon,
        upper_threshold=cfg.model.lstm.spikes.upper_threshold,
        lower_threshold=cfg.model.lstm.spikes.lower_threshold,
    )

    ds_test = Seq2SeqDataset(
        df=test_df,
        past_features=cfg.model.lstm.data.past_features,
        future_features=cfg.model.lstm.data.known_future_features,
        lookback=lookback,
        horizon=horizon,
        upper_threshold=cfg.model.lstm.spikes.upper_threshold,
        lower_threshold=cfg.model.lstm.spikes.lower_threshold,
    )

    train_loader = DataLoader(
        ds_train,
        batch_size=cfg.model.lstm.data.batch_size,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        ds_test,
        batch_size=cfg.model.lstm.data.batch_size,
        shuffle=False,
        drop_last=False,
    )

    test_timestamps = test_df.index[lookback : lookback + len(ds_test)]

    return train_loader, test_loader, test_timestamps
