import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Literal, Optional, Tuple

from config import FINAL_DATA_DIR
from modelling.model_config import LSTMConfig, CORE_MEAN_EXOG_COLS, ALL_MEAN_EXOG_COLS, VARIANCE_EXOG_COLS

def load_parametric_dataset(
    data: Optional[pd.DataFrame] = None,
    file_path: Optional[str] = None,
    target: Literal["level", "spread"] = "level",
    drop_na: bool = True,
    train_end_date: Optional[str] = None,
    test_start_date: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Prepare a single modelling dataframe for parametric electricity price models
    (SARIMAX, SARIMAX+GARCH-X, Markov-switching ARX).

    Parameters
    ----------
    data : pd.DataFrame, optional
        Pre-loaded dataframe with all raw features.
    file_path : str, optional
        Path to a CSV/Parquet file containing the raw data.
    target : {"level", "spread"}
        "level"  -> intraday_wap as target
        "spread" -> intraday_wap - da_price as target
    drop_na : bool
        Whether to drop rows with NA after lag construction.
    train_end_date : str, optional
        End date for training set (inclusive)
    test_start_date : str, optional
        Start date for test set (inclusive)
    verbose : bool
        Print loading information

    Returns
    -------
    pd.DataFrame or tuple
        Final dataframe ready for model estimation.
        If train/test dates provided, returns (train_df, test_df)
    """

    if data is None and file_path is None:
        raise ValueError("Either `data` or `file_path` must be provided.")

    if data is None:
        if file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path, parse_dates=True, index_col=0)
    else:
        df = data.copy()

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except:
            raise ValueError("Index must be convertible to DatetimeIndex")

    df = df.sort_index()
    
    if verbose:
        print(f"Loaded {len(df)} rows from {df.index.min()} to {df.index.max()}")

    # ------------------------------------------------------------------
    # Target construction
    # ------------------------------------------------------------------
    df["id_da_spread"] = df["intraday_wap"] - df["da_price"]

    if target == "level":
        df["y"] = df["intraday_wap"]
    elif target == "spread":
        df["y"] = df["id_da_spread"]
    else:
        raise ValueError("target must be either 'level' or 'spread'")

    # ------------------------------------------------------------------
    # Forecast / outturn errors
    # ------------------------------------------------------------------
    df["demand_error"] = df["demand_actual"] - df["demand_da_forecast"]
    df["wind_error"] = df["wind_outturn"] - df["wind_forecast_ng"]
    df["solar_error"] = df["solar_outturn"] - df["solar_forecast_ng"]

    # ------------------------------------------------------------------
    # Endogenous lags (for AR terms / Markov switching)
    # ------------------------------------------------------------------
    df["y_lag1"] = df["y"].shift(1)
    df["y_lag24"] = df["y"].shift(24)

    # ------------------------------------------------------------------
    # Exogenous lags based on cross-correlation evidence
    # ------------------------------------------------------------------
    for var in ["demand_actual", "wind_outturn"]:
        df[f"{var}_lag1"] = df[var].shift(1)
        df[f"{var}_lag24"] = df[var].shift(24)

    # Optional short memory for solar if needed later
    df["solar_outturn_lag1"] = df["solar_outturn"].shift(1)

    # ------------------------------------------------------------------
    # Calendar / seasonality
    # ------------------------------------------------------------------
    if "hour" not in df.columns:
        df["hour"] = df.index.hour
    
    if "hour_sin" not in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    
    if "dow_sin" not in df.columns:
        dow = df.index.dayofweek
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    
    if "is_weekend" not in df.columns:
        df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    # ------------------------------------------------------------------
    # Variance-equation regressors (for GARCH-X)
    # ------------------------------------------------------------------
    df["abs_demand_error"] = df["demand_error"].abs()
    df["abs_wind_error"] = df["wind_error"].abs()
    df["abs_solar_error"] = df["solar_error"].abs()
    df["abs_id_da_spread"] = df["id_da_spread"].abs()

    # Peak-hour volatility dummy: 15–18
    df["is_peak_15_18"] = df["hour"].between(15, 18).astype(int)

    # ------------------------------------------------------------------
    # Feature exclusion logic for spread model
    # ------------------------------------------------------------------
    if target == "spread":
        df = df.drop(columns=["da_price"], errors="ignore")

    # ------------------------------------------------------------------
    # Final cleaning
    # ------------------------------------------------------------------
    if drop_na:
        initial_len = len(df)
        df = df.dropna()
        if verbose:
            print(f"Dropped {initial_len - len(df)} rows with NA values")

    # ------------------------------------------------------------------
    # Train/test split
    # ------------------------------------------------------------------
    if train_end_date and test_start_date:
        train_df = df[df.index <= train_end_date].copy()
        test_df = df[df.index >= test_start_date].copy()
        
        if verbose:
            print(f"Train set: {len(train_df)} rows ({train_df.index.min()} to {train_df.index.max()})")
            print(f"Test set: {len(test_df)} rows ({test_df.index.min()} to {test_df.index.max()})")
        
        return train_df, test_df

    return df

# ======================================================
# PUBLIC: parametric comparison loader
# ======================================================

def load_parametric_data(
    target: Literal["level", "spread"] = "level",
    verbose: bool = True,
    exogenous_type: Literal["core", "all", "variance"] = "core",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Load data for parametric models in array form.
    """

    file_path = FINAL_DATA_DIR / "core_2020-01-01_2025-12-31.parquet"

    if exogenous_type == "core":
        exog_cols = CORE_MEAN_EXOG_COLS
    elif exogenous_type == "all":
        exog_cols = ALL_MEAN_EXOG_COLS
    elif exogenous_type == "variance":
        exog_cols = VARIANCE_EXOG_COLS

    # Global split – defined once here
    train_end_date = "2024-12-31 23:00"
    test_start_date = "2025-01-01 00:00"

    train_df, test_df = load_parametric_dataset(
        file_path=str(file_path),
        target=target,
        train_end_date=train_end_date,
        test_start_date=test_start_date,
        verbose=verbose,
    )

    y_train = train_df["y"].values
    y_test = test_df["y"].values

    X_train = train_df[exog_cols].values
    X_test = test_df[exog_cols].values

    test_timestamps = test_df.index

    return X_train, y_train, X_test, y_test, test_timestamps


# ======================================================
# PUBLIC: LSTM dataset + loader
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
    config: LSTMConfig,
    target: Literal["level", "spread"] = "level",
    verbose: bool = True,
) -> Tuple[DataLoader, DataLoader, pd.DatetimeIndex]:
    """
    Load univariate LSTM data (target only).
    """

    file_path = FINAL_DATA_DIR / "core_2020-01-01_2025-12-31.parquet"

    # Same global split as parametric models
    train_end_date = "2024-12-31 23:00"
    test_start_date = "2025-01-01 00:00"

    train_df, test_df = load_parametric_dataset(
        file_path=str(file_path),
        target=target,
        train_end_date=train_end_date,
        test_start_date=test_start_date,
        verbose=verbose,
    )

    y_train = train_df["y"].values
    y_test = test_df["y"].values

    train_ds = SequenceDataset(y_train, config.sequence_length)
    test_ds = SequenceDataset(y_test, config.sequence_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Align timestamps with predictions
    test_timestamps = test_df.index[config.sequence_length :]

    return train_loader, test_loader, test_timestamps