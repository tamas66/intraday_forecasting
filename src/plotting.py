# src/plotting.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig
import hydra
from scipy.stats import norm

# ======================================================
# Plot style (thesis-friendly)
# ======================================================

def set_plot_style():
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.2,
        rc={
            "figure.figsize": (6.5, 4.0),
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        },
    )


# ======================================================
# Utility
# ======================================================

def _read_parquet_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_parquet(path)
    return None


# ======================================================
# Core plots (existing)
# ======================================================

def plot_crps_by_horizon(df: pd.DataFrame, out_path: Path, title: Optional[str] = None):
    plt.figure()
    sns.lineplot(data=df, x="horizon", y="crps", hue="model", marker="o")
    plt.xlabel("Forecast horizon (hours)")
    plt.ylabel("CRPS")
    plt.title(title or "CRPS by forecast horizon")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_crps_by_hour(df: pd.DataFrame, out_path: Path, title: Optional[str] = None):
    plt.figure()
    sns.lineplot(data=df, x="hour", y="crps", hue="model")
    plt.xlabel("Hour of day")
    plt.ylabel("CRPS")
    plt.title(title or "CRPS by hour of day")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_overall_crps(df: pd.DataFrame, out_path: Path, title: Optional[str] = None):
    plt.figure()
    sns.barplot(data=df, x="model", y="crps", errorbar=None)
    plt.ylabel("Average CRPS")
    plt.xlabel("")
    plt.title(title or "Overall CRPS")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_pit_histogram(df: pd.DataFrame, out_path: Path, title: Optional[str] = None):
    plt.figure()
    sns.histplot(
        data=df,
        x="pit",
        hue="model",
        bins=20,
        stat="density",
        common_norm=False,
        element="step",
    )
    plt.axhline(1.0, linestyle="--", color="black", linewidth=1)
    plt.xlabel("PIT")
    plt.ylabel("Density")
    plt.title(title or "PIT histogram (calibration check)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_quantile_residuals_time(df: pd.DataFrame, out_path: Path, title: Optional[str] = None):
    plt.figure()
    sns.lineplot(
        data=df.sort_values("target_time"),
        x="target_time",
        y="qres",
        hue="model",
        alpha=0.8,
    )
    plt.axhline(0.0, linestyle="--", color="black", linewidth=1)
    plt.ylabel("Quantile residual")
    plt.xlabel("")
    plt.title(title or "Quantile residuals over time")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_quantile_residual_distribution(df: pd.DataFrame, out_path: Path, title: Optional[str] = None):
    plt.figure()
    sns.histplot(
        data=df,
        x="qres",
        hue="model",
        stat="density",
        bins=30,
        element="step",
        common_norm=False,
    )

    x = np.linspace(-4, 4, 400)
    plt.plot(x, norm.pdf(x), linestyle="--", color="black", label="N(0,1)")

    plt.xlabel("Quantile residual")
    plt.ylabel("Density")
    plt.title(title or "Quantile residual distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_crps_vs_level(df: pd.DataFrame, out_path: Path, title: Optional[str] = None):
    plt.figure()
    sns.scatterplot(
        data=df,
        x="y_true",
        y="crps",
        hue="model",
        alpha=0.3,
        edgecolor=None,
    )
    plt.xlabel("Realised price")
    plt.ylabel("CRPS")
    plt.title(title or "CRPS vs realised price level")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ======================================================
# New: Quantile diagnostics plots
# ======================================================

def plot_quantile_coverage_heatmap(
    df_cov: pd.DataFrame,
    out_path: Path,
    model: Optional[str] = None,
    title: Optional[str] = None,
):
    """
    Expects columns: model, horizon, tau, coverage
    Produces a heatmap: tau (rows) x horizon (cols)
    """
    if model is not None:
        df_cov = df_cov[df_cov["model"] == model].copy()
        if df_cov.empty:
            return

    pivot = df_cov.pivot_table(index="tau", columns="horizon", values="coverage", aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)

    plt.figure(figsize=(7.0, 4.5))
    sns.heatmap(pivot, annot=False, cbar=True)
    plt.xlabel("Horizon")
    plt.ylabel("Quantile level τ")
    base = "Quantile coverage heatmap"
    if model is not None:
        base += f" ({model})"
    plt.title(title or base)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_quantile_pinball_by_tau(
    df_pinball: pd.DataFrame,
    out_path: Path,
    horizon: Optional[int] = None,
    title: Optional[str] = None,
):
    """
    Expects columns: model, horizon, tau, pinball
    Line plot of pinball vs tau, optionally for a single horizon.
    """
    if horizon is not None:
        df_pinball = df_pinball[df_pinball["horizon"] == horizon].copy()
        if df_pinball.empty:
            return

    plt.figure()
    sns.lineplot(data=df_pinball, x="tau", y="pinball", hue="model", marker="o")
    plt.xlabel("Quantile level τ")
    plt.ylabel("Mean pinball loss")
    if horizon is None:
        plt.title(title or "Pinball loss by quantile level")
    else:
        plt.title(title or f"Pinball loss by quantile level (horizon={horizon})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_quantile_crossing_rate(
    df_cross_rate: pd.DataFrame,
    out_path: Path,
    title: Optional[str] = None,
):
    """
    Expects columns: model, horizon, cross_rate, avg_crossings
    """
    plt.figure()
    sns.lineplot(data=df_cross_rate, x="horizon", y="cross_rate", hue="model", marker="o")
    plt.xlabel("Horizon")
    plt.ylabel("Crossing rate")
    plt.title(title or "Quantile crossing rate by horizon")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ======================================================
# New: Spike diagnostics plots
# ======================================================

def plot_spike_scores_by_horizon(
    df_scores: pd.DataFrame,
    out_path: Path,
    metric: str = "brier",
    title: Optional[str] = None,
):
    """
    Expects columns: model, horizon, brier, logloss, spike_rate
    """
    if metric not in df_scores.columns:
        return

    plt.figure()
    sns.lineplot(data=df_scores, x="horizon", y=metric, hue="model", marker="o")
    plt.xlabel("Horizon")
    plt.ylabel(metric)
    plt.title(title or f"Spike probability {metric} by horizon")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_spike_reliability_curve(
    df_rel: pd.DataFrame,
    out_path: Path,
    horizon: Optional[int] = None,
    title: Optional[str] = None,
):
    """
    Expects columns: model, horizon, p_bin, mean_p, emp_freq, n
    p_bin is string labels; we will plot mean_p vs emp_freq.
    """
    dfp = df_rel.copy()
    if horizon is not None:
        dfp = dfp[dfp["horizon"] == horizon].copy()
        if dfp.empty:
            return

    plt.figure()
    sns.lineplot(data=dfp, x="mean_p", y="emp_freq", hue="model", marker="o")
    # perfect calibration
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    plt.xlabel("Mean predicted spike probability")
    plt.ylabel("Empirical spike frequency")
    if horizon is None:
        plt.title(title or "Spike reliability curve")
    else:
        plt.title(title or f"Spike reliability curve (horizon={horizon})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ======================================================
# Main plotting routine
# ======================================================

def make_all_plots(eval_dir: Path, plot_dir: Path):
    plot_dir.mkdir(parents=True, exist_ok=True)

    # --- Core evaluation outputs ---
    crps_raw = _read_parquet_if_exists(eval_dir / "crps_raw.parquet")
    crps_horizon = _read_parquet_if_exists(eval_dir / "crps_by_horizon.parquet")
    crps_hour = _read_parquet_if_exists(eval_dir / "crps_by_hour.parquet")
    crps_overall = _read_parquet_if_exists(eval_dir / "crps_overall.parquet")

    if crps_horizon is not None and not crps_horizon.empty:
        plot_crps_by_horizon(crps_horizon, plot_dir / "crps_by_horizon.png")

    if crps_hour is not None and not crps_hour.empty:
        plot_crps_by_hour(crps_hour, plot_dir / "crps_by_hour.png")

    if crps_overall is not None and not crps_overall.empty:
        plot_overall_crps(crps_overall, plot_dir / "crps_overall.png")

    if crps_raw is not None and not crps_raw.empty:
        # Calibration + residual diagnostics
        if "pit" in crps_raw.columns:
            plot_pit_histogram(crps_raw.dropna(subset=["pit"]), plot_dir / "pit_histogram.png")

        if "qres" in crps_raw.columns:
            df_q = crps_raw.dropna(subset=["qres"]).copy()
            if not df_q.empty:
                plot_quantile_residuals_time(df_q, plot_dir / "quantile_residuals_time.png")
                plot_quantile_residual_distribution(df_q, plot_dir / "quantile_residual_distribution.png")

        if "crps" in crps_raw.columns and "y_true" in crps_raw.columns:
            plot_crps_vs_level(crps_raw.dropna(subset=["crps", "y_true"]), plot_dir / "crps_vs_price.png")

    # --- Quantile diagnostics (if present) ---
    df_cov = _read_parquet_if_exists(eval_dir / "quantile_coverage.parquet")
    df_pinball = _read_parquet_if_exists(eval_dir / "quantile_pinball.parquet")
    df_cross_rate = _read_parquet_if_exists(eval_dir / "quantile_crossing_rate.parquet")

    if df_cov is not None and not df_cov.empty:
        # heatmap per model (more interpretable than overlay)
        for m in sorted(df_cov["model"].dropna().unique()):
            out = plot_dir / f"quantile_coverage_heatmap_{m}.png"
            plot_quantile_coverage_heatmap(df_cov, out, model=m)

    if df_pinball is not None and not df_pinball.empty:
        plot_quantile_pinball_by_tau(df_pinball, plot_dir / "quantile_pinball_by_tau.png")
        # also show for horizon 1 if available
        if "horizon" in df_pinball.columns and (df_pinball["horizon"] == 1).any():
            plot_quantile_pinball_by_tau(df_pinball, plot_dir / "quantile_pinball_by_tau_h1.png", horizon=1)

    if df_cross_rate is not None and not df_cross_rate.empty:
        plot_quantile_crossing_rate(df_cross_rate, plot_dir / "quantile_crossing_rate.png")

    # --- Spike diagnostics (if present) ---
    df_spike_scores = _read_parquet_if_exists(eval_dir / "spike_scores.parquet")
    df_spike_rel = _read_parquet_if_exists(eval_dir / "spike_reliability.parquet")

    if df_spike_scores is not None and not df_spike_scores.empty:
        plot_spike_scores_by_horizon(df_spike_scores, plot_dir / "spike_brier_by_horizon.png", metric="brier")
        plot_spike_scores_by_horizon(df_spike_scores, plot_dir / "spike_logloss_by_horizon.png", metric="logloss")
        plot_spike_scores_by_horizon(df_spike_scores, plot_dir / "spike_rate_by_horizon.png", metric="spike_rate")

    if df_spike_rel is not None and not df_spike_rel.empty:
        plot_spike_reliability_curve(df_spike_rel, plot_dir / "spike_reliability_curve.png")
        # also show horizon 1 if present
        if "horizon" in df_spike_rel.columns and (df_spike_rel["horizon"] == 1).any():
            plot_spike_reliability_curve(df_spike_rel, plot_dir / "spike_reliability_curve_h1.png", horizon=1)


# ======================================================
# Hydra entry point
# ======================================================

@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    set_plot_style()

    target = cfg.get("target", "level")
    run_name = cfg.get("run_name", None)

    eval_dir = (
        Path(cfg.paths.outputs_dir)
        / target
        / f"horizon_{cfg.horizon}"
        / (run_name if run_name is not None else "default")
    )

    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")

    plot_dir = eval_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    make_all_plots(eval_dir, plot_dir)

    print(f"[PLOTS] Figures written to {plot_dir}")


if __name__ == "__main__":
    main()
