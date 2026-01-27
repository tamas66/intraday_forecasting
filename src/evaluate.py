# src/evaluate.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal

import numpy as np
import pandas as pd
from omegaconf import DictConfig
import hydra
from scipy.stats import norm


# ======================================================
# CRPS implementations
# ======================================================

def crps_from_samples(samples: np.ndarray, y: float) -> float:
    """
    CRPS for empirical distribution given samples.
    samples: shape (M,)
    """
    samples = np.asarray(samples, dtype=float)
    y = float(y)

    term1 = np.mean(np.abs(samples - y))
    term2 = 0.5 * np.mean(np.abs(samples[:, None] - samples[None, :]))
    return term1 - term2


def crps_from_quantiles(
    quantiles: np.ndarray,
    taus: np.ndarray,
    y: float,
) -> float:
    """
    CRPS from quantile forecasts using the pinball-loss identity.

    quantiles: shape (Q,)
    taus: shape (Q,)
    """
    y = float(y)
    q = np.asarray(quantiles, dtype=float)
    tau = np.asarray(taus, dtype=float)

    return np.mean(
        (tau - (y <= q).astype(float)) * (y - q)
    ) * 2.0


# ======================================================
# Distribution loaders
# ======================================================

def load_garch_distribution(path: Path, h: int) -> np.ndarray:
    """
    Load samples for a specific horizon step.
    """
    samples = np.load(path)  # shape (M, H)
    return samples[:, h]


def load_lstm_distribution(path: Path, h: int) -> Dict[str, np.ndarray]:
    """
    Load quantiles and spike prob for a horizon step.
    """
    data = np.load(path)
    return {
        "quantiles": data["quantiles"][h],
        "taus": data["taus"],
        "spike_prob": data["spike_prob"][h],
    }

# ======================================================
# PIT
# ======================================================
# ======================================================
# CRPS-consistent residual diagnostics
# ======================================================

def pit_from_samples(samples: np.ndarray, y: float) -> float:
    """
    Probability Integral Transform using empirical samples.
    """
    samples = np.asarray(samples, dtype=float)
    return float(np.mean(samples <= y))


def pit_from_quantiles(
    quantiles: np.ndarray,
    taus: np.ndarray,
    y: float,
) -> float:
    """
    Approximate PIT using quantile function inversion.
    """
    q = np.asarray(quantiles, dtype=float)
    tau = np.asarray(taus, dtype=float)

    if y <= q.min():
        return float(tau[0])
    if y >= q.max():
        return float(tau[-1])

    return float(np.interp(y, q, tau))


def quantile_residual(pit: float) -> float:
    """
    Normalised PIT residual ~ N(0,1) if calibrated.
    """
    eps = 1e-10
    pit = np.clip(pit, eps, 1 - eps)
    return float(norm.ppf(pit))



# ======================================================
# Main evaluation
# ======================================================

def evaluate_run(run_dir: Path, verbose: bool = True) -> pd.DataFrame:
    """
    Evaluate a single model run directory.
    """
    index_path = run_dir / "forecast_index.parquet"
    if not index_path.exists():
        raise FileNotFoundError(index_path)

    df = pd.read_parquet(index_path)

    crps_values = []
    pit_values = []
    qres_values = []

    for _, row in df.iterrows():
        y = row["y_true"]
        h = int(row["horizon"]) - 1
        dist_path = Path(row["dist_path"])

        if row["dist_kind"] == "samples":
            samples = load_garch_distribution(dist_path, h)
            crps = crps_from_samples(samples, y)
            pit = pit_from_samples(samples, y)

        elif row["dist_kind"] == "quantiles":
            d = load_lstm_distribution(dist_path, h)
            crps = crps_from_quantiles(d["quantiles"], d["taus"], y)
            pit = pit_from_quantiles(d["quantiles"], d["taus"], y)

        else:
            raise ValueError(f"Unknown dist_kind: {row['dist_kind']}")

        crps_values.append(crps)
        pit_values.append(pit)
        qres_values.append(quantile_residual(pit))

    df["crps"] = crps_values
    df["pit"] = pit_values
    df["qres"] = qres_values

    return df


# ======================================================
# Aggregations
# ======================================================

def aggregate_crps(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Produce standard aggregation tables.
    """
    out = {}

    out["by_horizon"] = (
        df.groupby(["model", "horizon"])
        .crps.mean()
        .reset_index()
        .sort_values(["model", "horizon"])
    )

    out["by_hour"] = (
        df.assign(hour=lambda x: pd.to_datetime(x.target_time).dt.hour)
        .groupby(["model", "hour"])
        .crps.mean()
        .reset_index()
        .sort_values(["model", "hour"])
    )

    out["overall"] = (
        df.groupby("model")
        .crps.mean()
        .reset_index()
    )

    return out


# ======================================================
# Hydra entry point
# ======================================================

@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """
    Evaluate all trained models for a given target & horizon.
    """
    target = cfg.get("target", "level")

    base_dir = Path(cfg.data.paths.models_dir) / target

    all_rows = []

    for model_type_dir in base_dir.iterdir():
            # 1. Format the integer into the folder name 'horizon_X'
            horizon_folder = f"horizon_{cfg.horizon}"
            run_dir = model_type_dir / horizon_folder
            
            if not run_dir.is_dir():
                continue

            # 2. Check for the file inside the correctly formatted folder
            if (run_dir / "forecast_index.parquet").exists():
                if cfg.get("verbose", True):
                    print(f"[EVAL] Evaluating {model_type_dir.name}")

                df_model = evaluate_run(run_dir)
                all_rows.append(df_model)

    if not all_rows:
        raise RuntimeError("No forecast_index.parquet found")

    df_all = pd.concat(all_rows, ignore_index=True)

    out_dir = (
        Path(cfg.paths.outputs_dir)
        / target
        / f"horizon_{cfg.horizon}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save raw CRPS
    df_all.to_parquet(out_dir / "crps_raw.parquet", index=False)

    # Aggregates
    aggs = aggregate_crps(df_all)
    for name, df_agg in aggs.items():
        df_agg.to_parquet(out_dir / f"crps_{name}.parquet", index=False)

    print(f"[EVAL] Results written to {out_dir}")


if __name__ == "__main__":
    main()
