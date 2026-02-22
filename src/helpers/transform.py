import pandas as pd
import numpy as np
import pywt
from typing import List
import hydra
from omegaconf import DictConfig
import holidays

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



def add_calendar_features(df: pd.DataFrame, country: str = "UK") -> pd.DataFrame:
    """Add cyclical features, bank holidays, and specific GB market flags."""
    df = df.copy()
    idx = df.index

    # 1. Cyclical Hours (24h)
    hours = idx.hour
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    # 2. Cyclical Day of Week (7d)
    dow = idx.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    
    # 3. Cyclical Month (12m) - Important for heating/cooling seasonality
    month = idx.month - 1 # 0-indexed for sine/cos
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # 4. Binary Flags
    df["is_weekend"] = (dow >= 5).astype(int)
    
    # Bank Holidays (UK specific)
    uk_holidays = holidays.CountryHoliday(country)
    df["is_holiday"] = pd.Series(idx).apply(lambda x: x in uk_holidays).values.astype(int)
    
    # "Sunday-like" behavior: treat holidays as weekends
    df["is_off_day"] = ((df["is_weekend"] == 1) | (df["is_holiday"] == 1)).astype(int)

    # 5. Energy Market Specifics
    # Peak Pricing usually 16:00 - 19:00 in GB (Triads/Winter peaks)
    df["is_peak_15_18"] = ((hours >= 15) & (hours <= 18)).astype(int)
    
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
# DECOMPOSITION FEATURES (OPTIONAL)
# ======================

def add_wavelet_targets(
    df: pd.DataFrame,
    target_col: str,
    window: int = 336,
    wavelet: str = "db4",
    level: int = 3,
) -> pd.DataFrame:
    """
    Rolling causal wavelet decomposition.
    Extracts wavelet components at time t using only past data.
    """

    series = df[target_col].values

    wav_trend = np.full(len(series), np.nan)
    wav_d1 = np.full(len(series), np.nan)
    wav_d2 = np.full(len(series), np.nan)
    wav_d3 = np.full(len(series), np.nan)
    wav_resid = np.full(len(series), np.nan)

    for t in range(window, len(series)):
        window_slice = series[t - window:t]

        if np.isnan(window_slice).any():
            continue

        coeffs = pywt.wavedec(window_slice, wavelet, level=level)

        # coeffs: [A3, D3, D2, D1]
        A3, D3, D2, D1 = coeffs

        # Reconstruct components
        A3_rec = pywt.waverec([A3] + [None]*level, wavelet)
        D3_rec = pywt.waverec([None, D3] + [None]*(level-1), wavelet)
        D2_rec = pywt.waverec([None, None, D2] + [None]*(level-2), wavelet)
        D1_rec = pywt.waverec([None, None, None, D1], wavelet)

        # Trim padding
        A3_rec = A3_rec[-window:]
        D3_rec = D3_rec[-window:]
        D2_rec = D2_rec[-window:]
        D1_rec = D1_rec[-window:]

        reconstructed = A3_rec + D3_rec + D2_rec + D1_rec
        resid = window_slice - reconstructed

        wav_trend[t] = A3_rec[-1]
        wav_d3[t] = D3_rec[-1]
        wav_d2[t] = D2_rec[-1]
        wav_d1[t] = D1_rec[-1]
        wav_resid[t] = resid[-1]

    df["wav_trend"] = wav_trend
    df["wav_detail_3"] = wav_d3
    df["wav_detail_2"] = wav_d2
    df["wav_detail_1"] = wav_d1
    df["wav_residual"] = wav_resid

    return df

def add_jump_targets(
    df: pd.DataFrame,
    target_col: str,
    window: int = 336,
    wavelet: str = "db4",
    level: int = 1,
    threshold_k: float = 5.0,
) -> pd.DataFrame:
    """
    Rolling wavelet-based jump extraction.
    Conservative threshold (k=5).
    """

    series = df[target_col].values

    jump_flag = np.zeros(len(series))
    jump_component = np.full(len(series), np.nan)
    smooth_price = np.full(len(series), np.nan)

    for t in range(window, len(series)):
        window_slice = series[t - window:t]

        if np.isnan(window_slice).any():
            continue

        coeffs = pywt.wavedec(window_slice, wavelet, level=level)

        # Highest frequency detail
        D1 = coeffs[-1]

        # Robust threshold using MAD
        mad = np.median(np.abs(D1 - np.median(D1)))
        tau = threshold_k * mad

        jump_mask = np.abs(D1) > tau

        # Reconstruct jump-only signal
        D1_jump = np.zeros_like(D1)
        D1_jump[jump_mask] = D1[jump_mask]

        jump_signal = pywt.waverec([None, D1_jump], wavelet)
        jump_signal = jump_signal[-window:]

        jump_component[t] = jump_signal[-1]
        smooth_price[t] = series[t] - jump_signal[-1]
        jump_flag[t] = 1 if abs(jump_signal[-1]) > 0 else 0

    df["jump_flag"] = jump_flag
    df["jump_component"] = jump_component
    df["smooth_price"] = smooth_price

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
    # da_prices["da_price_epex"] = df_da_price["DA HR Price (EPEX)"]
    # da_prices["da_price_nordpool"] = df_da_price["DA HR Price (Nordpool)"]
    # da_prices["da_price_hh"] = df_da_price["DA HH Price"]
    # da_prices["da_volume_epex"] = df_da_price["DA HR Volume (EPEX)"]
    # da_prices["da_volume_nordpool"] = df_da_price["DA HR Volume (Nordpool)"]
    # da_prices["da_volume_hh"] = df_da_price["DA HH Volume"]
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
    # demand["demand_indo"] = df_demand_anticipated["DEMAND (INDO)"]
    # demand["demand_itsdo"] = df_demand_anticipated["DEMAND (ITSDO)"]
    demand["demand_forecast_tsdf"] = df_demand_anticipated["DEMAND FORECAST (TSDF)"]
    # demand["demand_forecast_enhanced"] = df_demand_anticipated["ENHANCED DEMAND FORECAST (TSDF)"]
    # demand["demand_forecast_ndf"] = df_demand_anticipated["DEMAND FORECAST (NDF)"]
    # demand["demand_forecast_enappsys"] = df_demand_anticipated["DEMAND ITSDO FORECAST (ENAPPSYS)"]
    demand["demand_da_forecast"] = df_da_demand_forecast["DEMAND FORECAST (TSDF)"]
    # demand["demand_da_forecast_enappsys"] = df_da_demand_forecast["DEMAND FORECAST (ENAPPSYS)"]
    demand["demand_error"] = df_demand_forecast_error["Demand Error (TSDF)"]
    # demand["demand_error_enhanced"] = df_demand_forecast_error["Demand Error Forecast (Enhanced)"]
    demand = demand.sort_index().astype(float)

    # ----------------------
    # WIND
    # ----------------------
    wind = pd.DataFrame(index=df_wind.index)
    wind["wind_outturn"] = df_wind["Wind Outturn"]
    wind["wind_forecast_ng"] = df_wind["National Grid Forecast"]
    # wind["wind_forecast_enappsys_adj"] = df_wind["EnAppSys Forecast Trend-Adjusted"]
    # wind["wind_forecast_enappsys_raw"] = df_wind["EnAppSys Forecast Unadjusted"]
    wind["wind_capacity"] = df_wind["Capacity"]
    # wind["wind_balancing"] = df_wind["Total Accepted Balancing Level"]
    wind["wind_error"] = wind["wind_outturn"] - wind["wind_forecast_ng"]
    # wind["wind_error_enappsys"] = wind["wind_outturn"] - wind["wind_forecast_enappsys_adj"]
    wind = wind.sort_index().astype(float)

    # ----------------------
    # SOLAR
    # ----------------------
    solar = pd.DataFrame(index=df_solar.index)
    solar["solar_outturn"] = df_solar["Sheffield Solar Outturn"]
    solar["solar_forecast_ng"] = df_solar["National Grid Forecast"]
    # solar["solar_forecast_enappsys"] = df_solar["Solar Forecast (EnAppSys)"]
    # solar["solar_forecast_enappsys_adj"] = df_solar["Trend Adjusted Solar Forecast (EnAppSys)"]
    # solar["solar_p10"] = df_solar["Solar P10 Forecast (EnAppSys)"]
    # solar["solar_p90"] = df_solar["Solar P90 Forecast (EnAppSys)"]
    solar["solar_capacity"] = df_solar["Capacity"]
    solar["solar_error"] = solar["solar_outturn"] - solar["solar_forecast_ng"]
    # solar["solar_error_enappsys"] = solar["solar_outturn"] - solar["solar_forecast_enappsys_adj"]
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
    """
    Merge domain DataFrames handling the Hourly (DA) vs Half-Hourly (ID) frequency mismatch.
    """

    # 1. Identify the 'Master Index' (the highest frequency)
    # Usually intraday_prices is 30min or 15min.
    master_index = intraday_prices.index

    # 2. Reindex and Forward Fill Hourly data to match Intraday frequency
    # This ensures an hourly DA price of £60 at 10:00 is also applied to 10:15, 10:30, 10:45
    da_prices_up = da_prices.reindex(master_index).ffill()
    demand_up = demand.reindex(master_index).ffill()
    wind_up = wind.reindex(master_index).ffill()
    solar_up = solar.reindex(master_index).ffill()
    flows_up = flows.reindex(master_index).ffill()
    generation_up = generation.reindex(master_index).ffill()

    # 3. Join on the higher frequency index
    df = (
        intraday_prices  # Start with ID as the left/base df
        .join(da_prices_up, how="left")
        .join(demand_up, how="left")
        .join(wind_up, how="left")
        .join(solar_up, how="left")
        .join(flows_up, how="left")
        .join(generation_up, how="left")
    )

    # --- Rest of your cleaning and feature engineering ---
    df = clean_base_dataframe(df, price_cols, tz, index_name)
    df = add_calendar_features(df)
    
    # Update: Include High/Low/Close spread features if available in your df_intraday
    # These are high-value for probabilistic volatility forecasting
    if "intraday_high" in df.columns and "intraday_low" in df.columns:
        df["id_range"] = df["intraday_high"] - df["intraday_low"]

    df = add_wind_penetration(df, "wind_outturn", "demand_actual")
    df = add_renewable_ramps(df, "wind_outturn", [2, 4, 8, 24, 48]) # Adjusted windows for HH
    df = add_renewable_ramps(df, "solar_outturn", [2, 4, 8, 24 , 48]) # Adjusted windows for HH
    df = add_rolling_features(df, ["intraday_wap", "da_price"], [1, 2, 3, 4, 24, 48])

    # Lag features (Note: lag 1 is now 30 mins, lag 48 is 24 hours)
    for var in ["demand_actual", "wind_outturn"]:
        df[f"{var}_lag1"] = df[var].shift(2)
        df[f"{var}_lag48"] = df[var].shift(48) # 24h lag for Half-Hourly

    df["solar_outturn_lag1"] = df["solar_outturn"].shift(2)
    df["id_da_spread"] = df["intraday_wap"] - df["da_price"]
    
    # Variance regressors
    df["abs_demand_error"] = df["demand_error"].abs()
    df["abs_wind_error"] = df["wind_error"].abs()
    df["abs_solar_error"] = df["solar_error"].abs()
    df["abs_id_da_spread"] = df["id_da_spread"].abs()

    return df

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