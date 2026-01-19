import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import config as cfg
import helpers.api as api
import helpers.transform as prep
import logging
from pathlib import Path
from typing import Literal, Optional, Tuple, List
from sklearn.preprocessing import StandardScaler
import joblib
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


# ======================
# CONFIGURATION
# ======================

RAW_DATA_DIR = cfg.RAW_DATA_DIR
PROCESSED_DATA_DIR = cfg.PROCESSED_DATA_DIR
FINAL_DATA_DIR = cfg.FINAL_DATA_DIR

# ======================
# RAW DATA (API → Parquet)
# ======================

def fetch_and_save_raw_data(
    year: int,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Fetch raw data from API for a given year and save as parquet files.
    
    Parameters
    ----------
    year : int
        Year to fetch data for
    force_refresh : bool
        If True, fetch from API even if files exist
        
    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with keys: da_prices, intraday_prices, demand, wind, solar, flows, generation
    """
    year_dir = RAW_DATA_DIR / str(year)
    year_dir.mkdir(exist_ok=True)
    
    # Check if all files exist
    required_files = [
        "da_prices.parquet",
        "intraday_prices.parquet",
        "demand.parquet",
        "wind.parquet",
        "solar.parquet",
        "flows.parquet",
        "generation.parquet",
    ]
    
    all_exist = all((year_dir / f).exists() for f in required_files)
    
    if all_exist and not force_refresh:
        logger.info(f"Loading existing raw data for {year}")
        return load_raw_data(year)
    
    # Fetch from API
    logger.info(f"Fetching raw data from API for {year}")
    
    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    
    # Make all API calls
    logger.info("  Fetching prices...")
    df_da_price = api.query_da_price(start=start, end=end)
    df_da_borders = api.query_da_borders(start=start, end=end)
    df_intraday = api.query_intraday_wap(start=start, end=end)
    
    logger.info("  Fetching renewables...")
    df_wind = api.query_wind_forecast(start=start, end=end)
    df_solar = api.query_solar_forecast(start=start, end=end)
    
    logger.info("  Fetching demand...")
    df_demand_anticipated = api.query_demand_anticipated(start=start, end=end)
    df_demand_forecast_error = api.query_demand_forecast_error(start=start, end=end)
    df_da_demand_forecast = api.query_da_demand_forecast(start=start, end=end)
    
    logger.info("  Fetching generation mix...")
    df_generation = api.query_generation_vs_forecast(start=start, end=end)
    
    # Convert to domain dataframes
    logger.info("  Converting to domain dataframes...")
    (
        da_prices,
        intraday_prices,
        demand,
        wind,
        solar,
        flows,
        generation,
    ) = prep.build_dataframes_from_api_responses(
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
    
    # Save to parquet
    logger.info(f"  Saving to {year_dir}")
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


def load_raw_data(year: int) -> dict[str, pd.DataFrame]:
    """
    Load raw data for a given year from parquet files.
    
    Parameters
    ----------
    year : int
        Year to load
        
    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with keys: da_prices, intraday_prices, demand, wind, solar, flows, generation
    """
    year_dir = RAW_DATA_DIR / str(year)
    
    def load_and_fix_index(filepath: Path) -> pd.DataFrame:
        """Load parquet and ensure proper datetime index with timezone."""
        df = pd.read_parquet(filepath)
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Ensure timezone is UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        elif str(df.index.tz) != "UTC":
            df.index = df.index.tz_convert("UTC")
        
        # Ensure sorted
        df = df.sort_index()
        
        # Don't set frequency - pandas will complain if there are any gaps
        
        return df
    
    return {
        "da_prices": load_and_fix_index(year_dir / "da_prices.parquet"),
        "intraday_prices": load_and_fix_index(year_dir / "intraday_prices.parquet"),
        "demand": load_and_fix_index(year_dir / "demand.parquet"),
        "wind": load_and_fix_index(year_dir / "wind.parquet"),
        "solar": load_and_fix_index(year_dir / "solar.parquet"),
        "flows": load_and_fix_index(year_dir / "flows.parquet"),
        "generation": load_and_fix_index(year_dir / "generation.parquet"),
    }


# ======================
# PROCESSED DATA (Features)
# ======================

def process_and_save_features(
    year: int,
    price_cols: list[str] | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Load raw data, build features, and save processed dataframe.
    
    Parameters
    ----------
    year : int
        Year to process
    price_cols : list[str], optional
        Price columns to clean (forward fill, etc.)
    force_refresh : bool
        If True, reprocess even if file exists
        
    Returns
    -------
    pd.DataFrame
        Processed feature dataframe
    """
    processed_file = PROCESSED_DATA_DIR / f"{year}.parquet"
    
    if processed_file.exists() and not force_refresh:
        logger.info(f"Loading existing processed data for {year}")
        df = pd.read_parquet(processed_file)
        
        # Ensure proper datetime index with UTC timezone
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        elif str(df.index.tz) != "UTC":
            df.index = df.index.tz_convert("UTC")
        
        return df.sort_index()
    
    logger.info(f"Processing features for {year}")
    
    # Load raw data
    raw_data = fetch_and_save_raw_data(year)
    
    # Default price columns
    if price_cols is None:
        price_cols = [
            "da_price",
            "intraday_wap",
            "ssp",
            "sbp",
        ]
    
    # Build features
    df = prep.build_feature_dataframe(
        da_prices=raw_data["da_prices"],
        intraday_prices=raw_data["intraday_prices"],
        demand=raw_data["demand"],
        wind=raw_data["wind"],
        solar=raw_data["solar"],
        flows=raw_data["flows"],
        generation=raw_data["generation"],
        price_cols=price_cols,
    )
    
    # Save
    logger.info(f"  Saving to {processed_file}")
    df.to_parquet(processed_file)
    
    return df


# ======================
# BATCH OPERATIONS
# ======================

def fetch_all_years(
    start_year: int,
    end_year: int,
    force_refresh: bool = False,
) -> None:
    """
    Fetch raw data for multiple years.
    
    Parameters
    ----------
    start_year : int
        First year to fetch
    end_year : int
        Last year to fetch (inclusive)
    force_refresh : bool
        If True, refetch even if files exist
    """
    for year in range(start_year, end_year + 1):
        try:
            fetch_and_save_raw_data(year, force_refresh=force_refresh)
        except Exception as e:
            logger.error(f"Failed to fetch data for {year}: {e}")


def process_all_years(
    start_year: int,
    end_year: int,
    price_cols: list[str] | None = None,
    force_refresh: bool = False,
) -> None:
    """
    Process features for multiple years.
    
    Parameters
    ----------
    start_year : int
        First year to process
    end_year : int
        Last year to process (inclusive)
    price_cols : list[str], optional
        Price columns to clean
    force_refresh : bool
        If True, reprocess even if files exist
    """
    for year in range(start_year, end_year + 1):
        try:
            process_and_save_features(year, price_cols=price_cols, force_refresh=force_refresh)
        except Exception as e:
            logger.error(f"Failed to process data for {year}: {e}")


# ======================
# DATA LOADERS (for modeling)
# ======================

def load_year(year: int, stage: Literal["raw", "processed"] = "processed") -> pd.DataFrame | dict:
    """
    Load data for a single year.
    
    Parameters
    ----------
    year : int
        Year to load
    stage : {"raw", "processed"}
        Which stage to load from
        
    Returns
    -------
    pd.DataFrame or dict
        If stage="processed": single DataFrame
        If stage="raw": dict of domain DataFrames
    """
    if stage == "raw":
        return load_raw_data(year)
    else:
        df = pd.read_parquet(PROCESSED_DATA_DIR / f"{year}.parquet")
        
        # Ensure proper datetime index with UTC timezone
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        elif str(df.index.tz) != "UTC":
            df.index = df.index.tz_convert("UTC")
        
        return df.sort_index()


def load_date_range(
    start_date: datetime | str,
    end_date: datetime | str,
    stage: Literal["raw", "processed"] = "processed",
) -> pd.DataFrame | dict:
    """
    Load data for a date range (automatically handles multiple years).
    
    Parameters
    ----------
    start_date : datetime or str
        Start date
    end_date : datetime or str
        End date
    stage : {"raw", "processed"}
        Which stage to load from
        
    Returns
    -------
    pd.DataFrame or dict
        If stage="processed": concatenated DataFrame
        If stage="raw": dict of concatenated domain DataFrames
    """
    if isinstance(start_date, str):
        start_date_parsed = pd.to_datetime(start_date)
        if getattr(start_date_parsed, "tz", None) is None:
            start_date = start_date_parsed.tz_localize(cfg.TZ)
        else:
            start_date = start_date_parsed.tz_convert(cfg.TZ)
    if isinstance(end_date, str):
        end_date_parsed = pd.to_datetime(end_date)
        # If parsed datetime is naive, localize to cfg.TZ; otherwise convert to cfg.TZ
        if getattr(end_date_parsed, "tz", None) is None:
            end_date = end_date_parsed.tz_localize(cfg.TZ)
        else:
            end_date = end_date_parsed.tz_convert(cfg.TZ)
    
    start_year = start_date.year
    end_year = end_date.year
    
    if stage == "processed":
        # Load and concatenate processed data
        dfs = []
        for year in range(start_year, end_year + 1):
            df = load_year(year, stage="processed")
            dfs.append(df)
        
        df_full = pd.concat(dfs, axis=0).sort_index()
        return df_full.loc[start_date:end_date]
    
    else:
        # Load and concatenate raw data
        raw_data = {
            "da_prices": [],
            "intraday_prices": [],
            "demand": [],
            "wind": [],
            "solar": [],
            "flows": [],
            "generation": [],
        }
        
        for year in range(start_year, end_year + 1):
            year_data = load_raw_data(year)
            for key in raw_data.keys():
                raw_data[key].append(year_data[key])
        
        # Concatenate each domain
        for key in raw_data.keys():
            df_full = pd.concat(raw_data[key], axis=0).sort_index()
            raw_data[key] = df_full.loc[start_date:end_date]
        
        return raw_data

def load_and_save_working_subset(
    start_date: datetime | str,
    end_date: datetime | str,
    features: Literal["core", "baseline"] | None = None,
) -> pd.DataFrame | dict:
    """
    Load a date range and save to a single parquet file for quick access.
    
    Parameters
    ----------
    start_date : datetime or str
        Start date
    end_date : datetime or str
        End date
    features : Literal["core", "baseline"] | None
        List of features to include, or None for all
    output_folder : Path or str
        Folder to save the parquet file
    """
    df = load_date_range(start_date, end_date, stage="processed")
    
    if features == "core":
        feature_cols = cfg.CORE_FEATURES
    elif features == "baseline":
        feature_cols = cfg.BASELINE_FEATURES
    else:
        feature_cols = df.columns.tolist()
    
    df_subset = df[feature_cols]
    output_folder = FINAL_DATA_DIR
    output_file = output_folder / f"{features}_{start_date}_{end_date}.parquet"
    df_subset.to_parquet(output_file)
    
    logger.info(f"Saved working subset to {output_file}")
    
    return df_subset

# ======================
# MAIN
# ======================

if __name__ == "__main__":
    load_and_save_working_subset("2020-01-01", "2025-12-31", features="core")