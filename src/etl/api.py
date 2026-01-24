import io
import pandas as pd
import requests
from datetime import datetime
from enappsys import EnAppSys
from omegaconf import DictConfig


# ======================
# CLIENT SETUP
# ======================

def get_client(cfg: DictConfig) -> EnAppSys:
    """Initialize EnAppSys client using Hydra config."""
    return EnAppSys(
        user=cfg.api.api.user,
        secret=cfg.api.api.password,
    )


# ======================
# GENERIC CHART WRAPPERS
# ======================

def chart_query(
    cfg: DictConfig,
    *,
    code: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """
    Unified EnAppSys chart query using client API.
    """
    client = get_client(cfg)

    return (
        client.chart.get(
            "csv",
            code=code,
            start_dt=start,
            end_dt=end,
            resolution=cfg.api.settings.resolution,
            time_zone=cfg.api.settings.timezone,
            currency=cfg.api.settings.currency,
            min_avg_max=False,
        )
        .to_df()
    )


def chart_query_with_version(
    cfg: DictConfig,
    *,
    code: str,
    version: int,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """
    Direct URL construction for charts requiring explicit versioning.
    """
    params = {
        "code": code,
        "start": start.strftime("%Y%m%d%H%M"),
        "end": end.strftime("%Y%m%d%H%M"),
        "res": cfg.api.settings.resolution,
        "timezone": cfg.api.settings.timezone,
        "currency": cfg.api.settings.currency,
        "minavmax": "false",
        "tag": "csv",
        "delimiter": "comma",
        "user": cfg.api.api.user,
        "pass": cfg.api.api.password,
        "version": version,
    }

    response = requests.get(
        "https://app.enappsys.com/datadownload",
        params=params,
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"EnAppSys API error {response.status_code}: {response.text[:300]}"
        )

    df = pd.read_csv(
        io.StringIO(response.text),
        index_col=0,
        parse_dates=True,
        date_format="[%d/%m/%Y %H:%M]",
    )

    df.index.name = "timestamp"
    df.index = df.index.tz_localize(
        cfg.api.settings.timezone, ambiguous="infer"
    )

    return df


# ======================
# DATA-SPECIFIC QUERIES
# ======================

def query_da_price(cfg: DictConfig, start=None, end=None) -> pd.DataFrame:
    """Day-ahead auction prices (GB)."""
    return chart_query(
        cfg,
        code="gb/elec/epex/daprices",
        start=start,
        end=end,
    )


def query_da_borders(cfg: DictConfig, start=None, end=None) -> pd.DataFrame:
    """Day-ahead border / system prices (GB + neighbors)."""
    return chart_query(
        cfg,
        code="gb/elec/pricing/daprices",
        start=start,
        end=end,
    )


def query_intraday_wap(cfg: DictConfig, start=None, end=None) -> pd.DataFrame:
    """Intraday weighted average prices (half-hourly)."""
    return chart_query(
        cfg,
        code="gb/elec/epex/hh",
        start=start,
        end=end,
    )


def query_wind_forecast(cfg: DictConfig, start=None, end=None) -> pd.DataFrame:
    """Wind generation forecast."""
    return chart_query(
        cfg,
        code="gb/elec/renewables/wind",
        start=start,
        end=end,
    )


def query_solar_forecast(cfg: DictConfig, start=None, end=None) -> pd.DataFrame:
    """Solar generation forecast."""
    return chart_query(
        cfg,
        code="gb/elec/renewables/solar",
        start=start,
        end=end,
    )


def query_demand_anticipated(cfg: DictConfig, start=None, end=None) -> pd.DataFrame:
    """Anticipated electricity demand."""
    return chart_query(
        cfg,
        code="gb/elec/demand/anticipated",
        start=start,
        end=end,
    )


def query_demand_forecast_error(cfg: DictConfig, start=None, end=None) -> pd.DataFrame:
    """Demand forecast error (TSDF)."""
    return chart_query(
        cfg,
        code="gb/elec/demand/tsdf/error",
        start=start,
        end=end,
    )


def query_da_demand_forecast(cfg: DictConfig, start=None, end=None) -> pd.DataFrame:
    """Day-ahead electricity demand forecast."""
    return chart_query(
        cfg,
        code="gb/elec/demand/anticipated/tsdf",
        start=start,
        end=end,
    )


def query_generation_vs_forecast(cfg: DictConfig, start=None, end=None) -> pd.DataFrame:
    """Generation mix vs forecast."""
    return chart_query(
        cfg,
        code="gb/elec/generation/forecast",
        start=start,
        end=end,
    )
