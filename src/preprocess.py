import logging
from datetime import datetime
import os
from pathlib import Path
from typing import Literal, Optional, Union

import hydra
import pandas as pd
from omegaconf import DictConfig

import etl.api as api
import etl.transform as transform
"""
Data pipeline for Great Britain power market data.

Structure:
    data/
    ├── raw/           # Year-by-year domain-specific dataframes
    │   ├── 2023/
    │   │   ├── da_prices.parquet
    │   │   ├── intraday_prices.parquet
    │   │   ├── demand.parquet
    │   │   ├── wind.parquet
    │   │   ├── solar.parquet
    │   │   ├── flows.parquet
    │   │   └── generation.parquet
    │   └── 2024/
    │       └── ...
    |── processed/     # Feature engineered dataframes
    |   ├── 2023.parquet
    |   └── 2024.parquet
    |
    └── final/         # User-defined subsets for modeling
        ├── <features>_<start>_<end>.parquet
        └── <features>_<start>_<end>.parquet
"""


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

# ======================
# RAW DATA (API → Parquet)
# ======================

def fetch_and_save_raw_data(
    cfg: DictConfig,
    year: int,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    raw_dir = Path(cfg.data.paths.raw_data_dir)
    year_dir = raw_dir / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        "da_prices.parquet",
        "intraday_prices.parquet",
        "demand.parquet",
        "wind.parquet",
        "solar.parquet",
        "flows.parquet",
        "generation.parquet",
    ]

    if all((year_dir / f).exists() for f in required_files) and not force_refresh:
        logger.info(f"Loading existing raw data for {year}")
        return load_raw_data(cfg, year)

    logger.info(f"Fetching raw data from API for {year}")

    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)

    logger.info("  Fetching prices...")
    df_da_price = api.query_da_price(cfg=cfg, start=start, end=end)
    df_da_borders = api.query_da_borders(cfg=cfg, start=start, end=end)
    df_intraday = api.query_intraday_wap(cfg=cfg, start=start, end=end)

    logger.info("  Fetching renewables...")
    df_wind = api.query_wind_forecast(cfg=cfg, start=start, end=end)
    df_solar = api.query_solar_forecast(cfg=cfg, start=start, end=end)
    logger.info("  Fetching demand...")
    df_demand_anticipated = api.query_demand_anticipated(cfg=cfg, start=start, end=end)
    df_demand_forecast_error = api.query_demand_forecast_error(cfg=cfg, start=start, end=end)
    df_da_demand_forecast = api.query_da_demand_forecast(cfg=cfg, start=start, end=end)

    logger.info("  Fetching generation mix...")
    df_generation = api.query_generation_vs_forecast(cfg=cfg, start=start, end=end)

    (
        da_prices,
        intraday_prices,
        demand,
        wind,
        solar,
        flows,
        generation,
    ) = transform.build_dataframes_from_api_responses(
        df_da_price=df_da_price,
        df_da_borders=df_da_borders,
        df_intraday=df_intraday,
        df_wind=df_wind,
        df_solar=df_solar,
        df_demand_anticipated=df_demand_anticipated,
        df_demand_forecast_error=df_demand_forecast_error,
        df_da_demand_forecast=df_da_demand_forecast,
        df_generation=df_generation,
    )

    da_prices.to_parquet(year_dir / "da_prices.parquet")
    intraday_prices.to_parquet(year_dir / "intraday_prices.parquet")
    demand.to_parquet(year_dir / "demand.parquet")
    wind.to_parquet(year_dir / "wind.parquet")
    solar.to_parquet(year_dir / "solar.parquet")
    flows.to_parquet(year_dir / "flows.parquet")
    generation.to_parquet(year_dir / "generation.parquet")

    return {
        "da_prices": da_prices,
        "intraday_prices": intraday_prices,
        "demand": demand,
        "wind": wind,
        "solar": solar,
        "flows": flows,
        "generation": generation,
    }


def load_raw_data(cfg: DictConfig, year: int) -> dict[str, pd.DataFrame]:
    raw_dir = Path(cfg.data.paths.raw_data_dir) / str(year)

    def _load(path: Path) -> pd.DataFrame:
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df.sort_index()

    return {
        "da_prices": _load(raw_dir / "da_prices.parquet"),
        "intraday_prices": _load(raw_dir / "intraday_prices.parquet"),
        "demand": _load(raw_dir / "demand.parquet"),
        "wind": _load(raw_dir / "wind.parquet"),
        "solar": _load(raw_dir / "solar.parquet"),
        "flows": _load(raw_dir / "flows.parquet"),
        "generation": _load(raw_dir / "generation.parquet"),
    }


# ======================
# PROCESSED DATA
# ======================

def process_and_save_features(
    cfg: DictConfig,
    year: int,
    price_cols: Optional[list[str]] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    processed_dir = Path(cfg.data.paths.processed_data_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_file = processed_dir / f"{year}.parquet"

    if processed_file.exists() and not force_refresh:
        logger.info(f"Loading existing processed data for {year}")
        df = pd.read_parquet(processed_file)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df.sort_index()

    raw = fetch_and_save_raw_data(cfg, year, force_refresh)

    if price_cols is None:
        price_cols = ["da_price", "intraday_wap"]

    df = transform.build_feature_dataframe(
        da_prices=raw["da_prices"],
        intraday_prices=raw["intraday_prices"],
        demand=raw["demand"],
        wind=raw["wind"],
        solar=raw["solar"],
        flows=raw["flows"],
        generation=raw["generation"],
        price_cols=price_cols,
        tz = cfg.api.settings.timezone,
        index_name="timestamp",
    )

    df.to_parquet(processed_file)
    return df


# ======================
# LOADERS (Integrated)
# ======================

def load_year(
    cfg: DictConfig,
    year: int,
    stage: Literal["raw", "processed"] = "processed",
    force_refresh: bool = False,
) -> Union[pd.DataFrame, dict]:
    """
    Acts as the entry point for a single year of data. 
    If stage is 'processed', it ensures features are built and saved first.
    """
    if stage == "raw":
        return load_raw_data(cfg, year)

    # INTEGRATION: Call process_and_save_features instead of simple read_parquet
    # This ensures that if the file doesn't exist, it gets created.
    df = process_and_save_features(cfg, year, force_refresh=force_refresh)
    
    return df


def load_date_range(
    cfg: DictConfig,
    start_date: datetime | str,
    end_date: datetime | str,
    stage: Literal["raw", "processed"] = "processed",
    force_refresh: bool = False,
):
    tz = cfg.api.settings.timezone
    start = pd.to_datetime(start_date).tz_localize(tz) if not pd.to_datetime(start_date).tzinfo else pd.to_datetime(start_date).tz_convert(tz)
    end = pd.to_datetime(end_date).tz_localize(tz) if not pd.to_datetime(end_date).tzinfo else pd.to_datetime(end_date).tz_convert(tz)

    years = range(start.year, end.year + 1)

    if stage == "processed":
        # Pass force_refresh down to load_year
        df = pd.concat([load_year(cfg, y, force_refresh=force_refresh) for y in years]).sort_index()
        return df.loc[start.tz_convert("UTC"): end.tz_convert("UTC")]

    raw = {k: [] for k in ["da_prices", "intraday_prices", "demand", "wind", "solar", "flows", "generation"]}

    for y in years:
        year_data = load_raw_data(cfg, y)
        for k in raw:
            raw[k].append(year_data[k])
            
    return {k: pd.concat(v).sort_index().loc[start.tz_convert("UTC"): end.tz_convert("UTC")] for k, v in raw.items()}


def load_and_save_working_subset(
    cfg: DictConfig,
    start_date: datetime | str,
    end_date: datetime | str,
    features: Literal["core", "baseline"] | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    # 1. Trigger the processing/loading chain
    df = load_date_range(cfg, start_date, end_date, stage="processed", force_refresh=force_refresh)

    # 2. Column selection
    if features == "core":
        cols = cfg.data.features.preprocessing.core_features
    elif features == "baseline":
        cols = cfg.data.features.preprocessing.baseline_features
    else:
        cols = df.columns.tolist()

    df = df[cols]

    # 3. Save final subset
    out_dir = Path(cfg.data.paths.final_data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_str = pd.to_datetime(start_date).strftime("%Y%m%d")
    end_str = pd.to_datetime(end_date).strftime("%Y%m%d")
    tag = features or "all"

    out_path = out_dir / f"{tag}_{start_str}_{end_str}.parquet"
    df.to_parquet(out_path)

    logger.info(f"Saved working subset to {out_path}")
    return df


# ======================
# MAIN
# ======================

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    initialize_directories()
    
    # Check for a force_refresh flag in your Hydra config, defaulting to False
    force = cfg.get("force_refresh", False)
    
    # This now handles: Raw Download -> Feature Transform -> Saving -> Subset Slicing
    load_and_save_working_subset(
        cfg,
        start_date="2020-01-01",
        end_date="2025-12-31",
        features="core",
        force_refresh=force
    )

if __name__ == "__main__":
    main()