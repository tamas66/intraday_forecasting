# src/evaluate.py
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from omegaconf import DictConfig
import hydra
from scipy.stats import norm


# ======================================================
# CRPS
# ======================================================

def crps_from_samples(samples: np.ndarray, y: float) -> float:
    n = len(samples)
    samples = np.sort(samples)
    term1 = np.mean(np.abs(samples - y))
    term2 = np.sum(samples * (2 * np.arange(n) - n + 1)) / (n ** 2)
    return float(term1 - term2)


def crps_from_quantiles(q: np.ndarray, taus: np.ndarray, y: float) -> float:
    errors = y - q
    pinball = np.maximum(taus * errors, (taus - 1) * errors)
    return float(np.trapz(2.0 * pinball, x=taus))


# ======================================================
# PIT
# ======================================================

def pit_from_samples(samples: np.ndarray, y: float) -> float:
    return float(np.mean(samples <= y))


def pit_from_quantiles(q: np.ndarray, taus: np.ndarray, y: float) -> float:
    if y <= q.min():
        return float(taus[0])
    if y >= q.max():
        return float(taus[-1])
    return float(np.interp(y, q, taus))


def quantile_residual(pit: float) -> float:
    return float(norm.ppf(np.clip(pit, 1e-10, 1 - 1e-10)))


# ======================================================
# Distribution loaders
# ======================================================

def load_samples(path: Path, h: int) -> np.ndarray:
    """Load GARCH samples. train.py saves shape (horizon, n_sim) under key 'samples'."""
    return np.load(path)["samples"][h]          # (n_sim,)


def load_quantiles(path: Path, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Load LSTM quantiles. train.py saves shape (horizon, n_quantiles) under key 'quantiles'."""
    data = np.load(path)
    return data["quantiles"][h], data["taus"]   # (n_quantiles,), (n_quantiles,)


def load_jump(path: Path, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load jump-LSTM outputs. Returns smooth_quantiles, jump_prob, jump_size for step h."""
    data = np.load(path)
    return (
        data["smooth_quantiles"][h],  # (n_quantiles,)
        data["jump_prob"][h],         # (n_quantiles,) or scalar
        data["jump_size"][h],         # (n_quantiles,)
    )


# ======================================================
# WAP RECONSTRUCTION
# ======================================================

def _reconstruct(arr: np.ndarray, target: str, row: pd.Series) -> np.ndarray:
    """
    Shift forecast distribution back to WAP space.
    - level : no shift needed
    - spread : add known DA price
    - wavelet: add wavelet detail residual
    - jump   : already in WAP space (smooth + jump reconstructed upstream)
    """
    if target == "level":
        return arr
    if target == "spread":
        return arr + float(row["da_price"])
    if target == "wavelet":
        return arr + float(row["wav_detail"])
    if target == "jump":
        return arr
    raise ValueError(f"Unknown target '{target}'")


# ======================================================
# MAIN EVALUATION
# ======================================================

def evaluate_run(run_dir: Path, taus: np.ndarray) -> pd.DataFrame:
    """
    Score every forecast in a run directory.

    Parameters
    ----------
    run_dir : Path   — must contain forecast_index.parquet
    taus    : array  — quantile levels used by the LSTM (from cfg)
    """
    df = pd.read_parquet(run_dir / "forecast_index.parquet")

    crps_vals, pit_vals, qres_vals, hit_vals = [], [], [], []

    for _, row in df.iterrows():
        y_true    = float(row["y_true"])
        h         = int(row["horizon"]) - 1
        dist_path = Path(row["dist_path"])
        target    = row["target"]

        # ---- GARCH samples ------------------------------------------------
        if row["dist_kind"] == "samples":
            samples = load_samples(dist_path, h)
            samples = _reconstruct(samples, target, row)
            crps    = crps_from_samples(samples, y_true)
            pit     = pit_from_samples(samples, y_true)
            q05, q95 = np.quantile(samples, [0.05, 0.95])

        # ---- Quantile LSTM ------------------------------------------------
        elif row["dist_kind"] == "quantiles":
            q, _ = load_quantiles(dist_path, h)
            q    = _reconstruct(q, target, row)
            crps = crps_from_quantiles(q, taus, y_true)
            pit  = pit_from_quantiles(q, taus, y_true)
            q05, q95 = np.interp([0.05, 0.95], taus, q)

        # ---- Jump-diffusion LSTM ------------------------------------------
        elif row["dist_kind"] == "jump":
            smooth_q, jump_prob, jump_size = load_jump(dist_path, h)
            # Reconstruct WAP quantiles: smooth component + expected jump contribution
            # jump_prob * jump_size gives the mean jump adjustment per quantile level
            q    = smooth_q + jump_prob * jump_size
            q    = _reconstruct(q, target, row)
            crps = crps_from_quantiles(q, taus, y_true)
            pit  = pit_from_quantiles(q, taus, y_true)
            q05, q95 = np.interp([0.05, 0.95], taus, q)

        else:
            raise ValueError(f"Unknown dist_kind '{row['dist_kind']}'")

        hit = int(q05 <= y_true <= q95)
        crps_vals.append(crps)
        pit_vals.append(pit)
        qres_vals.append(quantile_residual(pit))
        hit_vals.append(hit)

    df["crps"]       = crps_vals
    df["pit"]        = pit_vals
    df["qres"]       = qres_vals
    df["hit_05_95"]  = hit_vals

    return df


# ======================================================
# AGGREGATION
# ======================================================

def aggregate(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    agg_spec = {"crps": ("crps", "mean"), "hit_rate_05_95": ("hit_05_95", "mean")}

    return {
        "overall": (
            df.groupby(["model", "target"]).agg(**agg_spec).reset_index()
        ),
        "by_horizon": (
            df.groupby(["model", "target", "horizon"]).agg(**agg_spec).reset_index()
        ),
        "by_hour": (
            df.assign(hour=pd.to_datetime(df["target_time"]).dt.hour)
            .groupby(["model", "target", "hour"]).agg(**agg_spec).reset_index()
        ),
    }


# ======================================================
# HYDRA ENTRY
# ======================================================

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):

    # run dir structure matches train.py: models_dir / target / model_name / horizon_N /
    base_dir = Path(cfg.data.paths.models_dir)
    taus     = np.array(cfg.model.lstm.architecture.outputs.quantiles)
    horizon  = int(cfg.horizon)

    all_rows = []

    for target_dir in sorted(base_dir.iterdir()):
        for model_dir in sorted(target_dir.iterdir()):

            run_dir = model_dir / f"horizon_{horizon}"
            if not run_dir.exists():
                continue

            print(f"[EVAL] {target_dir.name}/{model_dir.name}/horizon_{horizon}")

            df = evaluate_run(run_dir, taus=taus)
            df["model"]  = model_dir.name
            df["target"] = target_dir.name

            all_rows.append(df)

    if not all_rows:
        print("[EVAL] No runs found.")
        return

    df_all = pd.concat(all_rows, ignore_index=True)

    out_dir = Path(cfg.paths.outputs_dir) / "evaluation" / f"horizon_{horizon}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all.to_parquet(out_dir / "crps_raw.parquet", index=False)

    for name, d in aggregate(df_all).items():
        d.to_parquet(out_dir / f"crps_{name}.parquet", index=False)

    print(f"[EVAL] Written to {out_dir}")


if __name__ == "__main__":
    main()