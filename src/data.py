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
        df["y"] = df["intraday_wap"]
    elif target == "spread":
        df["y"] = df["id_da_spread"]
    else:
        raise ValueError("target must be 'level' or 'spread'")

    # --------------------------------------------------
    # Spread-specific cleanup
    # --------------------------------------------------
    if target == "spread":
        df = df.drop(columns=["da_price"], errors="ignore")

    # --------------------------------------------------
    # Final cleaning
    # --------------------------------------------------
    if drop_na:
        before = len(df)
        df = df.dropna()
        if verbose:
            print(f"Dropped {before - len(df)} rows with NA")

    # --------------------------------------------------
    # Train / test split
    # --------------------------------------------------
    if train_end_date and test_start_date:
        train_df = df[df.index <= train_end_date].copy()
        test_df = df[df.index >= test_start_date].copy()
        return train_df, test_df

    return df


# ======================================================
# GENERIC PARAMETRIC LOADER (NO MODEL ASSUMPTIONS)
# ======================================================

def load_parametric_data(
    *,
    cfg: DictConfig,
    target: Literal["level", "spread"] = "level",
    verbose: bool = True,
):
    """
    Load generic parametric data.
    Model-specific feature selection happens downstream.
    """

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
# LSTM DATASET
# ======================================================

class SequenceDataset(Dataset):
    def __init__(self, series: np.ndarray, seq_len: int):
        self.series = series
        self.seq_len = seq_len

    def __len__(self):
        return len(self.series) - self.seq_len

    def __getitem__(self, idx):
        x = self.series[idx : idx + self.seq_len]
        y = self.series[idx + self.seq_len]
        return (
            torch.tensor(x, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(y, dtype=torch.float32),
        )


def load_lstm_data(
    *,
    cfg: DictConfig,
    target: Literal["level", "spread"] = "level",
    verbose: bool = True,
) -> Tuple[DataLoader, DataLoader, pd.DatetimeIndex, int]:
    """
    Load univariate LSTM data.
    """

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

    seq_len = cfg.model.lstm.data.sequence_length
    batch_size = cfg.model.lstm.data.batch_size

    train_ds = SequenceDataset(y_train, seq_len)
    test_ds = SequenceDataset(y_test, seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    test_timestamps = test_df.index[seq_len:]
    param_length = len(train_df)

    return train_loader, test_loader, test_timestamps, param_length