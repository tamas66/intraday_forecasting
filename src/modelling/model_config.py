from dataclasses import dataclass
from typing import List, Optional, Tuple


# ======================================================
# COLUMN REGISTRIES
# ======================================================

CORE_MEAN_EXOG_COLS = [
    "da_price",
    "demand_actual",
    "demand_error",
]


ALL_MEAN_EXOG_COLS = [
    "da_price",
    "demand_actual",
    "demand_error",
    "demand_actual_lag1",
    "demand_actual_lag24",
    "wind_outturn",
    "wind_error",
    "wind_outturn_lag1",
    "wind_outturn_lag24",
    "solar_outturn",
    "solar_error",
    "solar_outturn_lag1",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
]

VARIANCE_EXOG_COLS = [
    "abs_demand_error",
    "abs_wind_error",
    "abs_solar_error",
    "abs_id_da_spread",
    "is_peak_15_18",
]


# =========================
# Parametric model configs
# =========================

@dataclass
class SARIMAXConfig:
    order: Tuple[int, int, int] = (1, 0, 1)
    seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 24)
    trend: str = "c"


@dataclass
class GARCHConfig:
    p: int = 1
    q: int = 1
    dist: str = "t"
    use_exogenous: bool = True


@dataclass
class RollingARXConfig:
    min_train_size: int = 24 * 365 * 2      # 2-year burn-in
    refit_every: int = 1                    # 1 = refit every step
    window_size: Optional[int] = None       # None = expanding window
    exog_cols: List[str] = None             # default handled in runner


# =========================
# ANN / LSTM config
# =========================

@dataclass
class LSTMConfig:
    # Data
    sequence_length: int = 24
    batch_size: int = 64

    # Model architecture
    input_size: int = 1
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.0

    # Training
    learning_rate: float = 1e-3
    num_epochs: int = 10

    # Runtime
    device: str = "cpu"  # overridden in train.py if cuda available
