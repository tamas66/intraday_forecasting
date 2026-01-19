import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from config import TZ, INDEX_NAME

# ======================
# INDEX & ALIGNMENT UTILITIES
# ======================

def normalize_time_index(df: pd.DataFrame, index_name: str = INDEX_NAME) -> pd.DataFrame:
    """
    Ensure the DataFrame has a proper DatetimeIndex, sorted, deduplicated, and timezone-aware.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if df.index.tz is None:
        df.index = df.index.tz_localize(TZ)
    else:
        df.index = df.index.tz_convert(TZ)

    df.index.name = index_name
    return df

# ======================
# BASE CLEANING
# ======================

def drop_price_nans(df: pd.DataFrame, price_cols: List[str]) -> pd.DataFrame:
    """
    Drop rows with missing values in critical price columns.
    """
    return df.dropna(subset=price_cols)


def interpolate_non_price_columns(df: pd.DataFrame, price_cols: List[str], limit: int = 4) -> pd.DataFrame:
    """
    Interpolate all columns except the specified price columns.
    """
    interp_cols = [c for c in df.columns if c not in price_cols]
    df[interp_cols] = df[interp_cols].interpolate(method="time", limit=limit)
    return df


def clean_base_dataframe(df: pd.DataFrame, price_cols: List[str]) -> pd.DataFrame:
    """
    Apply basic cleaning to the dataframe: normalize index, drop NaNs in prices, interpolate others.
    """
    df = normalize_time_index(df)
    df = drop_price_nans(df, price_cols)
    df = interpolate_non_price_columns(df, price_cols)
    return df


# ======================
# FEATURE ENGINEERING
# ======================

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical calendar features: hour, day of week, and weekend flag.
    """
    df = df.copy()
    idx = df.index

    # Hour encoding (cyclical)
    hours = idx.hour
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    # Day-of-week encoding (cyclical)
    dow = idx.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # Weekend flag
    df["is_weekend"] = dow >= 5

    return df


def add_wind_penetration(df: pd.DataFrame, wind_col: str, load_col: str) -> pd.DataFrame:
    """
    Compute wind penetration as wind_outturn / demand_actual.
    """
    df = df.copy()
    df["wind_penetration"] = df[wind_col] / df[load_col]
    return df


def add_renewable_ramps(df: pd.DataFrame, col: str, windows: List[int]) -> pd.DataFrame:
    """
    Compute renewable ramp features as differences over specified time windows.
    """
    df = df.copy()
    for w in windows:
        df[f"{col}_delta_{w}"] = df[col].diff(w)
    return df


def add_rolling_features(df: pd.DataFrame, columns: List[str], windows: List[int]) -> pd.DataFrame:
    """
    Add rolling mean and rolling standard deviation features for selected columns and windows.
    """
    df = df.copy()
    for col in columns:
        for w in windows:
            df[f"{col}_rollmean_{w}"] = df[col].rolling(window=w, min_periods=w).mean()
            df[f"{col}_rollstd_{w}"] = df[col].rolling(window=w, min_periods=w).std()
    return df


# ======================
# LAG FEATURES (hook only)
# ======================

def add_lag_features(df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
    """
    Add lagged versions of specified columns. Not called by default.
    Useful for LCF-driven feature selection.
    """
    df = df.copy()
    for col in columns:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


# ======================
# DATAFRAME ASSEMBLY
# ======================

def build_dataframes_from_api_responses(
    *,
    # Prices
    df_da_price: pd.DataFrame,
    df_da_borders: pd.DataFrame,
    df_intraday: pd.DataFrame,
    # Renewables
    df_wind: pd.DataFrame,
    df_solar: pd.DataFrame,
    # Demand
    df_demand_anticipated: pd.DataFrame,
    df_demand_forecast_error: pd.DataFrame,
    df_da_demand_forecast: pd.DataFrame,
    # Generation mix
    df_generation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convert raw API response DataFrames into cleaned domain-specific DataFrames.
    
    Returns
    -------
    tuple of pd.DataFrame
        (da_prices, intraday_prices, demand, wind, solar, flows, generation)
        
    Notes
    -----
    All dataframes maintain hourly resolution with UTC timezone indexing.
    """
    
    # ======================================================
    # DAY-AHEAD PRICES
    # ======================================================
    da_prices = pd.DataFrame(index=df_da_price.index)
    
    # Weighted price as primary
    da_prices["da_price"] = df_da_price["DA Weighted Price (EPEX Nordpool)"]
    
    # Additional price sources for feature richness
    da_prices["da_price_epex"] = df_da_price["DA HR Price (EPEX)"]
    da_prices["da_price_nordpool"] = df_da_price["DA HR Price (Nordpool)"]
    da_prices["da_price_hh"] = df_da_price["DA HH Price"]
    
    # Volumes
    da_prices["da_volume_epex"] = df_da_price["DA HR Volume (EPEX)"]
    da_prices["da_volume_nordpool"] = df_da_price["DA HR Volume (Nordpool)"]
    da_prices["da_volume_hh"] = df_da_price["DA HH Volume"]
    
    da_prices = da_prices.sort_index().astype(float)
    
    # ======================================================
    # INTRADAY PRICES
    # ======================================================
    intraday_prices = pd.DataFrame(index=df_intraday.index)
    
    # Half-hourly weighted average price
    intraday_prices["intraday_wap"] = df_intraday["HH WAP"]
    
    # Trading prices
    intraday_prices["intraday_open"] = df_intraday["OPENING TRADED PRICE"]
    intraday_prices["intraday_close"] = df_intraday["CLOSING TRADED PRICE"]
    intraday_prices["intraday_high"] = df_intraday["HIGH TRADED PRICE"]
    intraday_prices["intraday_low"] = df_intraday["LOW TRADED PRICE"]
    
    # Balancing prices (System Sell/Buy Price)
    intraday_prices["ssp"] = df_intraday["SSP"]
    intraday_prices["sbp"] = df_intraday["SBP"]
    
    intraday_prices = intraday_prices.sort_index().astype(float)
    
    # ======================================================
    # DEMAND
    # ======================================================
    demand = pd.DataFrame(index=df_demand_anticipated.index)
    
    # Actual demand (National Grid as primary)
    demand["demand_actual"] = df_demand_anticipated["DEMAND (NATIONAL GRID)"]
    
    # Alternative actual demand series
    demand["demand_indo"] = df_demand_anticipated["DEMAND (INDO)"]
    demand["demand_itsdo"] = df_demand_anticipated["DEMAND (ITSDO)"]
    
    # Forecasts
    demand["demand_forecast_tsdf"] = df_demand_anticipated["DEMAND FORECAST (TSDF)"]
    demand["demand_forecast_enhanced"] = df_demand_anticipated["ENHANCED DEMAND FORECAST (TSDF)"]
    demand["demand_forecast_ndf"] = df_demand_anticipated["DEMAND FORECAST (NDF)"]
    demand["demand_forecast_enappsys"] = df_demand_anticipated["DEMAND ITSDO FORECAST (ENAPPSYS)"]
    
    # Day-ahead forecast
    demand["demand_da_forecast"] = df_da_demand_forecast["DEMAND FORECAST (TSDF)"]
    demand["demand_da_forecast_enappsys"] = df_da_demand_forecast["DEMAND FORECAST (ENAPPSYS)"]
    
    # Forecast errors
    demand["demand_error_tsdf"] = df_demand_forecast_error["Demand Error (TSDF)"]
    demand["demand_error_enhanced"] = df_demand_forecast_error["Demand Error Forecast (Enhanced)"]
    
    demand = demand.sort_index().astype(float)
    
    # ======================================================
    # WIND
    # ======================================================
    wind = pd.DataFrame(index=df_wind.index)
    
    # Actual wind generation
    wind["wind_outturn"] = df_wind["Wind Outturn"]
    
    # Forecasts
    wind["wind_forecast_ng"] = df_wind["National Grid Forecast"]
    wind["wind_forecast_enappsys_adj"] = df_wind["EnAppSys Forecast Trend-Adjusted"]
    wind["wind_forecast_enappsys_raw"] = df_wind["EnAppSys Forecast Unadjusted"]
    
    # Capacity and balancing
    wind["wind_capacity"] = df_wind["Capacity"]
    wind["wind_balancing"] = df_wind["Total Accepted Balancing Level"]
    
    # Calculate forecast errors
    wind["wind_error_ng"] = wind["wind_outturn"] - wind["wind_forecast_ng"]
    wind["wind_error_enappsys"] = wind["wind_outturn"] - wind["wind_forecast_enappsys_adj"]
    
    wind = wind.sort_index().astype(float)
    
    # ======================================================
    # SOLAR
    # ======================================================
    solar = pd.DataFrame(index=df_solar.index)
    
    # Actual solar generation (Sheffield Solar)
    solar["solar_outturn"] = df_solar["Sheffield Solar Outturn"]
    
    # Forecasts
    solar["solar_forecast_ng"] = df_solar["National Grid Forecast"]
    solar["solar_forecast_enappsys"] = df_solar["Solar Forecast (EnAppSys)"]
    solar["solar_forecast_enappsys_adj"] = df_solar["Trend Adjusted Solar Forecast (EnAppSys)"]
    
    # Bounds (P10/P90)
    solar["solar_p10"] = df_solar["Solar P10 Forecast (EnAppSys)"]
    solar["solar_p90"] = df_solar["Solar P90 Forecast (EnAppSys)"]
    
    # Capacity
    solar["solar_capacity"] = df_solar["Capacity"]
    
    # Calculate forecast errors
    solar["solar_error_ng"] = solar["solar_outturn"] - solar["solar_forecast_ng"]
    solar["solar_error_enappsys"] = solar["solar_outturn"] - solar["solar_forecast_enappsys_adj"]
    
    solar = solar.sort_index().astype(float)
    
    # ======================================================
    # FLOWS (INTERCONNECTORS)
    # ======================================================
    flows = pd.DataFrame(index=df_generation.index)
    
    # Total interconnector flow (from generation mix)
    flows["interconnector_total"] = df_generation["Interconnectors"]
    
    # Border prices (from da_borders) - useful as proxy for flow drivers
    flows["price_belgium"] = df_da_borders["BELGIUM (BE)"]
    flows["price_denmark"] = df_da_borders["DENMARK (DK1)"]
    flows["price_france"] = df_da_borders["FRANCE (FR)"]
    flows["price_netherlands"] = df_da_borders["NETHERLANDS (NL)"]
    flows["price_norway"] = df_da_borders["NORWAY (NO2 - SDAC AUCTION)"]
    flows["price_isem"] = df_da_borders["ISEM (ISEM)"]
    
    # Price differentials (GB vs neighbors) - strong signal for flow direction
    gb_price = df_da_borders["GREAT BRITAIN (GB)"]
    flows["price_diff_belgium"] = gb_price - flows["price_belgium"]
    flows["price_diff_france"] = gb_price - flows["price_france"]
    flows["price_diff_netherlands"] = gb_price - flows["price_netherlands"]
    flows["price_diff_norway"] = gb_price - flows["price_norway"]
    
    flows = flows.sort_index().astype(float)
    
    # ======================================================
    # GENERATION MIX
    # ======================================================
    generation = pd.DataFrame(index=df_generation.index)
    
    # Actual generation by fuel type
    generation["gen_battery"] = df_generation["Battery"]
    generation["gen_biomass"] = df_generation["Biomass"]
    generation["gen_ccgt"] = df_generation["CCGT"]
    generation["gen_coal"] = df_generation["Coal"]
    generation["gen_flex"] = df_generation["Flex Gen"]
    generation["gen_hydro"] = df_generation["Hydro"]
    generation["gen_nuclear"] = df_generation["Nuclear"]
    generation["gen_oil"] = df_generation["Oil"]
    generation["gen_pumped_storage"] = df_generation["Pumped Storage"]
    
    # Forecasts (optional - can add later if needed)
    generation["gen_ccgt_forecast"] = df_generation["CCGT Forecast"]
    generation["gen_nuclear_forecast"] = df_generation["Nuclear Forecast"]
    generation["gen_interconnector_forecast"] = df_generation["Interconnectors Forecast"]
    
    # Total generation calculation
    fuel_cols = [
        "gen_battery", "gen_biomass", "gen_ccgt", "gen_coal", 
        "gen_flex", "gen_hydro", "gen_nuclear", "gen_oil", "gen_pumped_storage"
    ]
    generation["gen_total_conventional"] = generation[fuel_cols].sum(axis=1)
    
    generation = generation.sort_index().astype(float)
    
    # ======================================================
    # RETURN ALL DATAFRAMES
    # ======================================================
    return (
        da_prices,
        intraday_prices,
        demand,
        wind,
        solar,
        flows,
        generation,
    )


def build_feature_dataframe(
    da_prices: pd.DataFrame,
    intraday_prices: pd.DataFrame,
    demand: pd.DataFrame,
    wind: pd.DataFrame,
    solar: pd.DataFrame,
    flows: pd.DataFrame,
    generation: pd.DataFrame,
    price_cols: List[str],
) -> pd.DataFrame:
    """
    Merge all domain DataFrames into the final canonical feature DataFrame and add engineered features.
    """

    # Merge base DataFrames
    df = (
        da_prices
        .join(intraday_prices, how="inner")
        .join(demand, how="left")
        .join(wind, how="left")
        .join(solar, how="left")
        .join(flows, how="left")
        .join(generation, how="left")
    )

    # Clean merged DataFrame
    df = clean_base_dataframe(df, price_cols)

    # Add calendar features
    df = add_calendar_features(df)

    # Add wind penetration
    df = add_wind_penetration(df, wind_col="wind_outturn", load_col="demand_actual")

    # Add renewable ramps (hourly windows instead of quarter-hourly)
    df = add_renewable_ramps(df, col="wind_outturn", windows=[1, 4, 12])
    df = add_renewable_ramps(df, col="solar_outturn", windows=[1, 4])

    # Add rolling mean/std for prices (hourly windows)
    df = add_rolling_features(
        df, 
        columns=["intraday_wap", "da_price"], 
        windows=[4, 12, 24]
    )

    return df


def create_sequences(
    data: np.ndarray,
    sequence_length: int,
    forecast_horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series forecasting.
    
    Parameters
    ----------
    data : np.ndarray
        Input data of shape (n_samples, n_features) or (n_samples,)
    sequence_length : int
        Length of input sequences (lookback window)
    forecast_horizon : int
        Steps ahead to forecast (default: 1)
    
    Returns
    -------
    tuple
        (X_sequences, y_targets)
        X: shape (n_sequences, sequence_length, n_features)
        y: shape (n_sequences,) or (n_sequences, n_features)
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    X, y = [], []
    
    for i in range(len(data) - sequence_length - forecast_horizon + 1):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length + forecast_horizon - 1])
    
    return np.array(X), np.array(y)


def align_sequences_with_index(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Create sequences and maintain alignment with datetime index.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with datetime index
    feature_cols : List[str]
        Column names to use as features
    target_col : str
        Column name for target variable
    sequence_length : int
        Length of lookback window
    
    Returns
    -------
    tuple
        (X_sequences, y_targets, timestamps)
        timestamps align with y_targets (i.e., prediction times)
    """
    X_data = df[feature_cols].values
    y_data = df[target_col].values
    
    X_seq, y_seq = create_sequences(X_data, sequence_length)
    
    # Timestamps for predictions (t+1 for each sequence ending at t)
    timestamps = df.index[sequence_length:]
    
    return X_seq, y_seq, timestamps


def create_sample_weights(
    timestamps: pd.DatetimeIndex,
    peak_hours: Tuple[int, int] = (15, 18),
    peak_weight: float = 2.0,
    weekend_weight: float = 1.0,
) -> np.ndarray:
    """
    Create sample weights based on time characteristics.
    
    Parameters
    ----------
    timestamps : pd.DatetimeIndex
        Timestamps for each sample
    peak_hours : tuple
        (start_hour, end_hour) inclusive for peak period
    peak_weight : float
        Weight multiplier for peak hours
    weekend_weight : float
        Weight multiplier for weekends
    
    Returns
    -------
    np.ndarray
        Sample weights
    """
    weights = np.ones(len(timestamps))
    
    # Peak hours
    hours = timestamps.hour
    is_peak = (hours >= peak_hours[0]) & (hours <= peak_hours[1])
    weights[is_peak] *= peak_weight
    
    # Weekends
    is_weekend = timestamps.dayofweek >= 5
    weights[is_weekend] *= weekend_weight
    
    return weights

if __name__ == "__main__":
    print("This module provides helper functions for data preparation and feature engineering.")