from datetime import datetime
from dotenv import load_dotenv
import os
from enappsys import EnAppSys
from config import RES, TZ, CUR

# ======================
# CLIENT SETUP
# ======================

load_dotenv()

client = EnAppSys(
    user=os.getenv("user"),
    secret=os.getenv("password"),
)


# ======================
# GENERIC CHART WRAPPER
# ======================

def chart_query(
    *,
    code: str,
    currency: str | None = None,
    version: int | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
):
    """
    Unified EnAppSys chart query using chart `code`.
    
    Parameters
    ----------
    code : str
        Chart code (e.g., "gb/elec/epex/daprices/gb1")
    currency : str, optional
        Currency code (default: EUR)
    version : int, optional
        Data version (for forecasts)
    start : str, optional
        Start datetime (default: global START)
    end : str, optional
        End datetime (default: global END)
    
    Returns
    -------
    pd.DataFrame
        Data with datetime index
    """
    # Handle version parameter by appending to code
    full_code = code
    if version is not None:
        full_code = f"{code}&version={version}"
    
    return client.chart.get(
        "csv",
        code=full_code,
        start_dt=start,
        end_dt=end,
        resolution=RES,
        time_zone=TZ,
        currency=currency or CUR,
        min_avg_max=False,
    ).to_df()

# ======================
# LOW-LEVEL: Direct URL Construction
# ======================

def chart_query_with_version(
    *,
    code: str,
    version: int | None = None,
    currency: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
):
    """
    Direct URL construction for Chart API with version support.
    
    This bypasses the client wrapper to handle version parameters correctly.
    """
    import requests
    from datetime import datetime
    


    # Format dates as YYYYMMDDHHmm
    start_str = start.strftime("%Y%m%d%H%M")
    end_str = end.strftime("%Y%m%d%H%M")
    
    # Build parameters
    params = {
        "code": code,
        "start": start_str,
        "end": end_str,
        "res": RES,
        "timezone": TZ,
        "currency": currency or CUR,
        "minavmax": "false",
        "tag": "csv",
        "delimiter": "comma",
        "user": os.getenv("user"),
        "pass": os.getenv("password"),
    }
    
    # Add version if provided
    if version is not None:
        params["version"] = version
    
    # Make request
    url = "https://app.enappsys.com/datadownload"
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        raise Exception(f"API Error {response.status_code}: {response.text[:500]}")
    
    # Parse CSV
    import pandas as pd
    import io
    
    df = pd.read_csv(
        io.StringIO(response.text),
        header=[0, 1],
        index_col=0,
        parse_dates=True,
        date_format="[%d/%m/%Y %H:%M]",
    )
    df.index.name = "dateTime"
    df.index = df.index.tz_localize(TZ, ambiguous="infer")
    
    # Flatten column names (remove unit row)
    df.columns = df.columns.get_level_values(0)
    
    return df

# ======================
# DAY-AHEAD BORDER PRICES
# ======================
def query_da_borders(start: str | None = None, end: str | None = None):
    """
    Day-ahead system / border prices (gb)
    """
    return chart_query(
        code="gb/elec/pricing/daprices",
        currency=CUR,
        start=start,
        end=end,
    )

# ======================
# DAY-AHEAD AUCTIONS
# ======================

def query_da_price(start: str | None = None, end: str | None = None):
    """
    Day-ahead auction prices (gb)
    """
    return chart_query(
        code="gb/elec/epex/daprices",
        currency=CUR,
        start=start,
        end=end,
    )

# ======================
# INTRADAY
# ======================

def query_intraday_wap(start: str | None = None, end: str | None = None):
    """
    Intraday weighted average prices (delivery-hour indexed)
    
    Parameters
    ----------
    start : str, optional
        Start datetime (format: "YYYY-MM-DDTHH:MM")
    end : str, optional
        End datetime (format: "YYYY-MM-DDTHH:MM")
    """
    return chart_query(
        code="gb/elec/epex/hh",
        currency=CUR,
        start=start,
        end=end,
    )


# ======================
# RENEWABLE FORECASTS
# ======================

def query_wind_forecast(start: str | None = None, end: str | None = None):
    """
    Wind generation intraday forecast (v2)
    Uses direct URL construction to handle version parameter
    """
    return chart_query(
        code="gb/elec/renewables/wind",
        currency=CUR,
        start=start,
        end=end,
    )


def query_solar_forecast(start: str | None = None, end: str | None = None):
    """
    Solar generation intraday forecast (v2)
    Uses direct URL construction to handle version parameter
    """
    return chart_query(
        code="gb/elec/renewables/solar",
        currency=CUR,
        start=start,
        end=end,
    )

# ======================
# DEMAND FORECASTS
# ======================
def query_demand_anticipated(start: str | None = None, end: str | None = None):
    """
    Anticipated electricity demand forecast
    """
    return chart_query(
        code="gb/elec/demand/anticipated",
        start=start,
        end=end,
    )

def query_demand_forecast_error(start: str | None = None, end: str | None = None):
    """
    Demand forecast error time series
    """
    return chart_query(
        code="gb/elec/demand/tsdf/error",
        start=start,
        end=end,
    )

def query_da_demand_forecast(start: str | None = None, end: str | None = None):
    """
    Day-ahead electricity demand forecast
    """
    return chart_query(
        code="gb/elec/demand/anticipated/tsdf",
        start=start,
        end=end,
    )

# ======================
# GENERATION vs FORECAST
# ======================
def query_generation_vs_forecast(start: str | None = None, end: str | None = None):
    """
    Wind generation vs forecast
    """
    return chart_query(
        code="gb/elec/generation/forecast",
        currency=CUR,
        start=start,
        end=end,
    )