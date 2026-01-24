import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

import hydra
from omegaconf import DictConfig

# ======================
# INDEX & ALIGNMENT UTILITIES
# ======================

def normalize_time_index(
    df: pd.DataFrame,
    tz: str,
    index_name: str,
) -> pd.DataFrame:
    """
    Ensure the DataFrame has a proper DatetimeIndex, sorted, deduplicated,
    timezone-aware, and named.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if df.index.tz is None:
        df.index = df.index.tz_localize(tz)
    else:
        df.index = df.index.tz_convert(tz)

    df.index.name = index_name
    return df


# ======================
# BASE CLEANING
# ======================

def drop_price_nans(df: pd.DataFrame, price_cols: List[str]) -> pd.DataFrame:
    """Drop rows with missing values in critical price columns."""
    return df.dropna(subset=price_cols)


def interpolate_non_price_columns(
    df: pd.DataFrame,
    price_cols: List[str],
    limit: int = 4,
) -> pd.DataFrame:
    """Interpolate all columns except the specified price columns."""
    interp_cols = [c for c in df.columns if c not in price_cols]
    df[interp_cols] = df[interp_cols].interpolate(method="time", limit=limit)
    return df


def clean_base_dataframe(
    df: pd.DataFrame,
    price_cols: List[str],
    tz: str,
    index_name: str,
) -> pd.DataFrame:
    """Apply base cleaning: index normalization, price NaNs, interpolation."""
    df = normalize_time_index(df, tz=tz, index_name=index_name)
    df = drop_price_nans(df, price_cols)
    df = interpolate_non_price_columns(df, price_cols)
    return df


# ======================
# FEATURE ENGINEERING
# ======================

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical hour/day-of-week features and weekend flag."""
    df = df.copy()
    idx = df.index

    hours = idx.hour
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    dow = idx.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    df["is_weekend"] = dow >= 5
    return df


def add_wind_penetration(
    df: pd.DataFrame,
    wind_col: str,
    load_col: str,
) -> pd.DataFrame:
    df = df.copy()
    df["wind_penetration"] = df[wind_col] / df[load_col]
    return df


def add_renewable_ramps(
    df: pd.DataFrame,
    col: str,
    windows: List[int],
) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"{col}_delta_{w}"] = df[col].diff(w)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int],
) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        for w in windows:
            df[f"{col}_rollmean_{w}"] = df[col].rolling(w, min_periods=w).mean()
            df[f"{col}_rollstd_{w}"] = df[col].rolling(w, min_periods=w).std()
    return df


# ======================
# LAG FEATURES (OPTIONAL)
# ======================

def add_lag_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int],
) -> pd.DataFrame:
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
    df_da_price: pd.DataFrame,
    df_da_borders: pd.DataFrame,
    df_intraday: pd.DataFrame,
    df_wind: pd.DataFrame,
    df_solar: pd.DataFrame,
    df_demand_anticipated: pd.DataFrame,
    df_demand_forecast_error: pd.DataFrame,
    df_da_demand_forecast: pd.DataFrame,
    df_generation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convert raw API responses into domain-specific DataFrames.
    All outputs retain hourly UTC indexing.
    """

    # ----------------------
    # DAY-AHEAD PRICES
    # ----------------------
    da_prices = pd.DataFrame(index=df_da_price.index)
    da_prices["da_price"] = df_da_price["DA Weighted Price (EPEX Nordpool)"]
    da_prices["da_price_epex"] = df_da_price["DA HR Price (EPEX)"]
    da_prices["da_price_nordpool"] = df_da_price["DA HR Price (Nordpool)"]
    da_prices["da_price_hh"] = df_da_price["DA HH Price"]
    da_prices["da_volume_epex"] = df_da_price["DA HR Volume (EPEX)"]
    da_prices["da_volume_nordpool"] = df_da_price["DA HR Volume (Nordpool)"]
    da_prices["da_volume_hh"] = df_da_price["DA HH Volume"]
    da_prices = da_prices.sort_index().astype(float)

    # ----------------------
    # INTRADAY PRICES
    # ----------------------
    intraday_prices = pd.DataFrame(index=df_intraday.index)
    intraday_prices["intraday_wap"] = df_intraday["HH WAP"]
    intraday_prices["intraday_open"] = df_intraday["OPENING TRADED PRICE"]
    intraday_prices["intraday_close"] = df_intraday["CLOSING TRADED PRICE"]
    intraday_prices["intraday_high"] = df_intraday["HIGH TRADED PRICE"]
    intraday_prices["intraday_low"] = df_intraday["LOW TRADED PRICE"]
    intraday_prices["ssp"] = df_intraday["SSP"]
    intraday_prices["sbp"] = df_intraday["SBP"]
    intraday_prices = intraday_prices.sort_index().astype(float)

    # ----------------------
    # DEMAND
    # ----------------------
    demand = pd.DataFrame(index=df_demand_anticipated.index)
    demand["demand_actual"] = df_demand_anticipated["DEMAND (NATIONAL GRID)"]
    demand["demand_indo"] = df_demand_anticipated["DEMAND (INDO)"]
    demand["demand_itsdo"] = df_demand_anticipated["DEMAND (ITSDO)"]
    demand["demand_forecast_tsdf"] = df_demand_anticipated["DEMAND FORECAST (TSDF)"]
    demand["demand_forecast_enhanced"] = df_demand_anticipated["ENHANCED DEMAND FORECAST (TSDF)"]
    demand["demand_forecast_ndf"] = df_demand_anticipated["DEMAND FORECAST (NDF)"]
    demand["demand_forecast_enappsys"] = df_demand_anticipated["DEMAND ITSDO FORECAST (ENAPPSYS)"]
    demand["demand_da_forecast"] = df_da_demand_forecast["DEMAND FORECAST (TSDF)"]
    demand["demand_da_forecast_enappsys"] = df_da_demand_forecast["DEMAND FORECAST (ENAPPSYS)"]
    demand["demand_error_tsdf"] = df_demand_forecast_error["Demand Error (TSDF)"]
    demand["demand_error_enhanced"] = df_demand_forecast_error["Demand Error Forecast (Enhanced)"]
    demand = demand.sort_index().astype(float)

    # ----------------------
    # WIND
    # ----------------------
    wind = pd.DataFrame(index=df_wind.index)
    wind["wind_outturn"] = df_wind["Wind Outturn"]
    wind["wind_forecast_ng"] = df_wind["National Grid Forecast"]
    wind["wind_forecast_enappsys_adj"] = df_wind["EnAppSys Forecast Trend-Adjusted"]
    wind["wind_forecast_enappsys_raw"] = df_wind["EnAppSys Forecast Unadjusted"]
    wind["wind_capacity"] = df_wind["Capacity"]
    wind["wind_balancing"] = df_wind["Total Accepted Balancing Level"]
    wind["wind_error_ng"] = wind["wind_outturn"] - wind["wind_forecast_ng"]
    wind["wind_error_enappsys"] = wind["wind_outturn"] - wind["wind_forecast_enappsys_adj"]
    wind = wind.sort_index().astype(float)

    # ----------------------
    # SOLAR
    # ----------------------
    solar = pd.DataFrame(index=df_solar.index)
    solar["solar_outturn"] = df_solar["Sheffield Solar Outturn"]
    solar["solar_forecast_ng"] = df_solar["National Grid Forecast"]
    solar["solar_forecast_enappsys"] = df_solar["Solar Forecast (EnAppSys)"]
    solar["solar_forecast_enappsys_adj"] = df_solar["Trend Adjusted Solar Forecast (EnAppSys)"]
    solar["solar_p10"] = df_solar["Solar P10 Forecast (EnAppSys)"]
    solar["solar_p90"] = df_solar["Solar P90 Forecast (EnAppSys)"]
    solar["solar_capacity"] = df_solar["Capacity"]
    solar["solar_error_ng"] = solar["solar_outturn"] - solar["solar_forecast_ng"]
    solar["solar_error_enappsys"] = solar["solar_outturn"] - solar["solar_forecast_enappsys_adj"]
    solar = solar.sort_index().astype(float)

    # ----------------------
    # FLOWS
    # ----------------------
    flows = pd.DataFrame(index=df_generation.index)
    flows["interconnector_total"] = df_generation["Interconnectors"]
    flows["price_belgium"] = df_da_borders["BELGIUM (BE)"]
    flows["price_denmark"] = df_da_borders["DENMARK (DK1)"]
    flows["price_france"] = df_da_borders["FRANCE (FR)"]
    flows["price_netherlands"] = df_da_borders["NETHERLANDS (NL)"]
    flows["price_norway"] = df_da_borders["NORWAY (NO2 - SDAC AUCTION)"]
    flows["price_isem"] = df_da_borders["ISEM (ISEM)"]

    gb_price = df_da_borders["GREAT BRITAIN (GB)"]
    flows["price_diff_belgium"] = gb_price - flows["price_belgium"]
    flows["price_diff_france"] = gb_price - flows["price_france"]
    flows["price_diff_netherlands"] = gb_price - flows["price_netherlands"]
    flows["price_diff_norway"] = gb_price - flows["price_norway"]
    flows = flows.sort_index().astype(float)

    # ----------------------
    # GENERATION MIX
    # ----------------------
    generation = pd.DataFrame(index=df_generation.index)
    generation["gen_battery"] = df_generation["Battery"]
    generation["gen_biomass"] = df_generation["Biomass"]
    generation["gen_ccgt"] = df_generation["CCGT"]
    generation["gen_coal"] = df_generation["Coal"]
    generation["gen_flex"] = df_generation["Flex Gen"]
    generation["gen_hydro"] = df_generation["Hydro"]
    generation["gen_nuclear"] = df_generation["Nuclear"]
    generation["gen_oil"] = df_generation["Oil"]
    generation["gen_pumped_storage"] = df_generation["Pumped Storage"]
    generation["gen_ccgt_forecast"] = df_generation["CCGT Forecast"]
    generation["gen_nuclear_forecast"] = df_generation["Nuclear Forecast"]
    generation["gen_interconnector_forecast"] = df_generation["Interconnectors Forecast"]

    fuel_cols = [
        "gen_battery", "gen_biomass", "gen_ccgt", "gen_coal",
        "gen_flex", "gen_hydro", "gen_nuclear", "gen_oil", "gen_pumped_storage",
    ]
    generation["gen_total_conventional"] = generation[fuel_cols].sum(axis=1)
    generation = generation.sort_index().astype(float)

    return da_prices, intraday_prices, demand, wind, solar, flows, generation


def build_feature_dataframe(
    da_prices: pd.DataFrame,
    intraday_prices: pd.DataFrame,
    demand: pd.DataFrame,
    wind: pd.DataFrame,
    solar: pd.DataFrame,
    flows: pd.DataFrame,
    generation: pd.DataFrame,
    price_cols: List[str],
    tz: str,
    index_name: str,
) -> pd.DataFrame:
    """Merge domain DataFrames and add engineered features."""
    df = (
        da_prices
        .join(intraday_prices, how="inner")
        .join(demand, how="left")
        .join(wind, how="left")
        .join(solar, how="left")
        .join(flows, how="left")
        .join(generation, how="left")
    )

    df = clean_base_dataframe(df, price_cols, tz, index_name)
    df = add_calendar_features(df)
    df = add_wind_penetration(df, "wind_outturn", "demand_actual")
    df = add_renewable_ramps(df, "wind_outturn", [1, 4, 12])
    df = add_renewable_ramps(df, "solar_outturn", [1, 4])
    df = add_rolling_features(df, ["intraday_wap", "da_price"], [4, 12, 24])

    return df


# ======================
# SEQUENCING UTILITIES
# ======================

def create_sequences(
    data: np.ndarray,
    sequence_length: int,
    forecast_horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    X, y = [], []
    for i in range(len(data) - sequence_length - forecast_horizon + 1):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length + forecast_horizon - 1])

    return np.asarray(X), np.asarray(y)


def align_sequences_with_index(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    X_data = df[feature_cols].values
    y_data = df[target_col].values

    X_seq, y_seq = create_sequences(X_data, sequence_length)
    timestamps = df.index[sequence_length:]

    return X_seq, y_seq, timestamps


def create_sample_weights(
    timestamps: pd.DatetimeIndex,
    peak_hours: Tuple[int, int] = (15, 18),
    peak_weight: float = 2.0,
    weekend_weight: float = 1.0,
) -> np.ndarray:
    weights = np.ones(len(timestamps))

    hours = timestamps.hour
    weights[(hours >= peak_hours[0]) & (hours <= peak_hours[1])] *= peak_weight
    weights[timestamps.dayofweek >= 5] *= weekend_weight

    return weights


# ======================
# MAIN
# ======================

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("Data preparation & feature engineering module loaded.")
    print(f"Timezone: {cfg.market.settings.timezone}")
    print(f"Index name: {cfg.data.dataframe.index_name}")


if __name__ == "__main__":
    main()