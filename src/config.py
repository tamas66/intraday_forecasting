import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# API Settings
USER = os.getenv("user")
PASS = os.getenv("password")

# Market Settings
RES = "hourly"
TZ = "UTC"
CUR = "EUR"

# Pathing
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
FINAL_DATA_DIR = Path("data/final")
RESULTS_DIR = Path("outputs/results")
MODELS_DIR = Path("models")
EDA_DIR = Path("outputs/eda")

# Dataframe Settings
INDEX_NAME = "timestamp"
TARGET_COL = "intraday_wap"
CORE_FEATURES = [
    # Prices & spreads
    "intraday_wap",
    "da_price",
    # Wind fundamentals
    "wind_outturn",
    "wind_forecast_ng",
    # Solar fundamentals
    "solar_outturn",
    "solar_forecast_ng",
    # Demand
    "demand_actual",
    "demand_da_forecast",
    # Seasonality / calendar
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
]

BASELINE_FEATURES = [
    # Prices & spreads
    "intraday_wap",
    "da_price",
    "ssp",
    "sbp",

    # Demand (level + key forecast)
    "demand_actual",
    "demand_da_forecast",

    # Wind fundamentals
    "wind_outturn",
    "wind_forecast_ng",
    "wind_capacity",
    "wind_penetration",
    "wind_outturn_delta_1",
    "wind_outturn_delta_4",

    # Solar fundamentals
    "solar_outturn",
    "solar_forecast_ng",
    "solar_capacity",
    "solar_outturn_delta_1",
    "solar_outturn_delta_4",

    # Conventional generation stack (coarse)
    "gen_ccgt",
    "gen_nuclear",
    "gen_biomass",
    "gen_hydro",
    "gen_total_conventional",

    # Interconnection & foreign prices
    "interconnector_total",
    "price_france",
    "price_netherlands",
    "price_norway",
    "price_isem",

    # Seasonality / calendar
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",

    # Rolling statistics for price dynamics
    "intraday_wap_rollmean_4",
    "intraday_wap_rollstd_4",
    "intraday_wap_rollmean_12",
    "intraday_wap_rollstd_12",
    "da_price_rollmean_4",
    "da_price_rollstd_4",
    "da_price_rollmean_12",
    "da_price_rollstd_12",
]



@dataclass
class DataConfig:
    """Data loading configuration"""
    working_data: Path = PROCESSED_DATA_DIR / "core_2020-01-01_2025-12-31.parquet"
    target: str = "level"  # or "spread"
    train_end_date: Optional[str] = "2024-12-31"
    test_start_date: Optional[str] = "2025-01-01"
    validation_split: float = 0.2

@dataclass
class ExperimentConfig:
    """Overall experiment configuration"""
    data: DataConfig
    random_seed: int = 42
    save_results: bool = True
    results_dir: Path = RESULTS_DIR
    plot_dir: Path = RESULTS_DIR / "plots"
    
    def __post_init__(self):
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.plot_dir.mkdir(exist_ok=True, parents=True)


def initialize_directories():
    directories = [
        'data/raw',
        'data/processed',
        'data/final',
        'outputs/eda',
        'outputs/models'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Verified directory: {directory}")

if __name__ == "__main__":
    initialize_directories()