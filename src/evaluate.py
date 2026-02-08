# src/evaluate.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import hydra
from scipy.stats import norm
from scipy.special import expit

# ======================================================
# Helper functions for evaluation
# ======================================================

def pinball_loss(q: float, tau: float, y: float) -> float:
    e = y - q
    return float(max(tau * e, (tau - 1) * e))

def safe_logloss(p: float, y: int, eps: float = 1e-12) -> float:
    p = float(np.clip(p, eps, 1 - eps))
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)))

def spike_event(y: float, upper: float, lower: float) -> int:
    return int((y >= upper) or (y <= lower))


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

def evaluate_run(run_dir: Path, verbose: bool = True):
    """
    Returns:
      df_row: row-level table with CRPS/PIT/qres (+ timing columns if present)
      df_qdiag: long quantile diagnostics (only for quantile models)
      df_spike: spike diagnostics (only for quantile models with spike_prob)
    """

    index_path = run_dir / "forecast_index.parquet"
    if not index_path.exists():
        raise FileNotFoundError(index_path)

    cfg_path = run_dir / "config_snapshot.yaml"
    upper_thr = None
    lower_thr = None
    if cfg_path.exists():
        cfg = OmegaConf.load(cfg_path)
        # LSTM spikes thresholds live in cfg.model.spikes.*
        try:
            upper_thr = float(cfg.model.spikes.upper_threshold)
            lower_thr = float(cfg.model.spikes.lower_threshold)
        except Exception:
            upper_thr, lower_thr = None, None


    df = pd.read_parquet(index_path)

    crps_values = []
    pit_values = []
    qres_values = []
    qdiag_rows = []
    spike_rows = []
    crossing_rows = []


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
            qs = np.asarray(d["quantiles"], dtype=float)   # (Q,)
            taus = np.asarray(d["taus"], dtype=float)      # (Q,)
            sp = float(d["spike_prob"])                    # scalar

            crps = crps_from_quantiles(qs, taus, y)
            pit = pit_from_quantiles(qs, taus, y)

            # --- quantile diagnostics (per tau) ---
            # Quantile crossing check requires monotone qs in taus order
            is_monotone = int(np.all(np.diff(qs) >= 0))
            crossing_rows.append(
                {
                    "model": row["model"],
                    "target": row.get("target", None),
                    "origin_time": row["origin_time"],
                    "target_time": row["target_time"],
                    "horizon": int(row["horizon"]),
                    "is_monotone": is_monotone,
                    "n_crossings": int(np.sum(np.diff(qs) < 0)),
                }
            )

            for qv, tau in zip(qs, taus):
                qdiag_rows.append(
                    {
                        "model": row["model"],
                        "target": row.get("target", None),
                        "origin_time": row["origin_time"],
                        "target_time": row["target_time"],
                        "horizon": int(row["horizon"]),
                        "tau": float(tau),
                        "q_pred": float(qv),
                        "y_true": float(y),
                        "hit": int(y <= qv),                 # coverage indicator
                        "pinball": pinball_loss(qv, float(tau), float(y)),
                        "abs_err": float(abs(y - qv)),
                    }
                )

            # --- spike diagnostics (needs thresholds) ---
            if (upper_thr is not None) and (lower_thr is not None) and np.isfinite(sp):
                sev = spike_event(y=float(y), upper=upper_thr, lower=lower_thr)
                spike_rows.append(
                    {
                        "model": row["model"],
                        "target": row.get("target", None),
                        "origin_time": row["origin_time"],
                        "target_time": row["target_time"],
                        "horizon": int(row["horizon"]),
                        "spike_prob": float(sp),
                        "spike_event": int(sev),
                        "brier": float((sp - sev) ** 2),
                        "logloss": safe_logloss(sp, sev),
                    }
                )


        else:
            raise ValueError(f"Unknown dist_kind: {row['dist_kind']}")

        crps_values.append(crps)
        pit_values.append(pit)
        qres_values.append(quantile_residual(pit))

    df["crps"] = crps_values
    df["pit"] = pit_values
    df["qres"] = qres_values

    df_qdiag = pd.DataFrame(qdiag_rows) if qdiag_rows else pd.DataFrame()
    df_cross = pd.DataFrame(crossing_rows) if crossing_rows else pd.DataFrame()
    df_spike = pd.DataFrame(spike_rows) if spike_rows else pd.DataFrame()

    return df, df_qdiag, df_cross, df_spike


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

    all_rows, all_qdiag, all_cross, all_spike = [], [], [], []

    for model_type_dir in base_dir.iterdir():
        if not model_type_dir.is_dir():
            continue

        # model_type_dir = models/{target}/{model}
        for run_name_dir in model_type_dir.iterdir():
            if not run_name_dir.is_dir():
                continue

            # run_name_dir = models/{target}/{model}/{run_name}
            run_dir = run_name_dir / f"horizon_{cfg.horizon}"
            if not run_dir.is_dir():
                continue

            index_path = run_dir / "forecast_index.parquet"
            if not index_path.exists():
                continue

            if cfg.get("verbose", True):
                print(f"[EVAL] Evaluating model={model_type_dir.name}, run_name={run_name_dir.name}")

            df_row, df_qdiag, df_cross, df_spike = evaluate_run(run_dir)

            # Tag outputs for grouping
            for d in (df_row, df_qdiag, df_cross, df_spike):
                if d is not None and not d.empty:
                    d["model_type"] = model_type_dir.name
                    d["run_name"] = run_name_dir.name

        all_rows.append(df_row)
        if not df_qdiag.empty:
            all_qdiag.append(df_qdiag)
        if not df_cross.empty:
            all_cross.append(df_cross)
        if not df_spike.empty:
            all_spike.append(df_spike)

    if not all_rows:
        raise RuntimeError("No forecast_index.parquet found")

    df_all = pd.concat(all_rows, ignore_index=True)
    df_all["hour"] = pd.to_datetime(df_all.target_time).dt.hour
    df_all["month"] = pd.to_datetime(df_all.target_time).dt.month

    # y-level bins (per model) for "regions"
    df_all["y_bin"] = df_all.groupby("model")["y_true"].transform(
        lambda s: pd.qcut(s, q=10, duplicates="drop")
    )

    out_dir = (
        Path(cfg.paths.outputs_dir)
        / target
        / f"horizon_{cfg.horizon}"
        / (cfg.run_name if cfg.run_name is not None else "default")
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    region = (
        df_all.groupby(["model", "horizon", "hour"])
        .crps.mean()
        .reset_index()
        .rename(columns={"crps": "crps_by_hour"})
    )
    region.to_parquet(out_dir / "region_crps_by_hour.parquet", index=False)

    region2 = (
        df_all.groupby(["model", "horizon", "month"])
        .crps.mean()
        .reset_index()
        .rename(columns={"crps": "crps_by_month"})
    )
    region2.to_parquet(out_dir / "region_crps_by_month.parquet", index=False)

    region3 = (
        df_all.groupby(["model", "horizon", "y_bin"])
        .crps.mean()
        .reset_index()
        .rename(columns={"crps": "crps_by_ybin"})
    )
    region3.to_parquet(out_dir / "region_crps_by_ybin.parquet", index=False)




    # Save raw CRPS
    df_all.to_parquet(out_dir / "crps_raw.parquet", index=False)

    # Aggregates
    aggs = aggregate_crps(df_all)
    for name, df_agg in aggs.items():
        df_agg.to_parquet(out_dir / f"crps_{name}.parquet", index=False)

    if all_qdiag:
        df_q = pd.concat(all_qdiag, ignore_index=True)
        df_q.to_parquet(out_dir / "quantile_diag.parquet", index=False)

        # Coverage by (model, horizon, tau)
        cov = (
            df_q.groupby(["model", "horizon", "tau"])
            .hit.mean()
            .reset_index()
            .rename(columns={"hit": "coverage"})
        )
        cov.to_parquet(out_dir / "quantile_coverage.parquet", index=False)

        # Pinball by (model, horizon, tau)
        pb = (
            df_q.groupby(["model", "horizon", "tau"])
            .pinball.mean()
            .reset_index()
        )
        pb.to_parquet(out_dir / "quantile_pinball.parquet", index=False)
    
    if all_spike:
        df_s = pd.concat(all_spike, ignore_index=True)
        df_s.to_parquet(out_dir / "spike_diag.parquet", index=False)

        # Aggregate scores
        spike_scores = (
            df_s.groupby(["model", "horizon"])
            .agg(brier=("brier", "mean"), logloss=("logloss", "mean"), spike_rate=("spike_event", "mean"))
            .reset_index()
        )
        spike_scores.to_parquet(out_dir / "spike_scores.parquet", index=False)

        # Reliability bins (per model, horizon)
        df_s["p_bin"] = pd.cut(df_s["spike_prob"], bins=np.linspace(0, 1, 11), include_lowest=True)
        rel = (
            df_s.groupby(["model", "horizon", "p_bin"])
            .agg(mean_p=("spike_prob", "mean"), emp_freq=("spike_event", "mean"), n=("spike_event", "size"))
            .reset_index()
        )
        rel.to_parquet(out_dir / "spike_reliability.parquet", index=False)


    print(f"[EVAL] Results written to {out_dir}")


if __name__ == "__main__":
    main()
