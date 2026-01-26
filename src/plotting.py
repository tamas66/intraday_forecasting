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
# Individual plots
# ======================================================

def plot_crps_by_horizon(
    df: pd.DataFrame,
    out_path: Path,
    title: Optional[str] = None,
):
    plt.figure()
    sns.lineplot(
        data=df,
        x="horizon",
        y="crps",
        hue="model",
        marker="o",
    )
    plt.xlabel("Forecast horizon (hours)")
    plt.ylabel("CRPS")
    plt.title(title or "CRPS by forecast horizon")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_crps_by_hour(
    df: pd.DataFrame,
    out_path: Path,
    title: Optional[str] = None,
):
    plt.figure()
    sns.lineplot(
        data=df,
        x="hour",
        y="crps",
        hue="model",
    )
    plt.xlabel("Hour of day")
    plt.ylabel("CRPS")
    plt.title(title or "CRPS by hour of day")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_overall_crps(
    df: pd.DataFrame,
    out_path: Path,
    title: Optional[str] = None,
):
    plt.figure()
    sns.barplot(
        data=df,
        x="model",
        y="crps",
        errorbar=None,
    )
    plt.ylabel("Average CRPS")
    plt.xlabel("")
    plt.title(title or "Overall CRPS")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ======================================================
# PIL
# ======================================================
def plot_pit_histogram(
    df: pd.DataFrame,
    out_path: Path,
    title: Optional[str] = None,
):
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

def plot_quantile_residuals_time(
    df: pd.DataFrame,
    out_path: Path,
    title: Optional[str] = None,
):
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

def plot_quantile_residual_distribution(
    df: pd.DataFrame,
    out_path: Path,
    title: Optional[str] = None,
):
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

def plot_crps_vs_level(
    df: pd.DataFrame,
    out_path: Path,
    title: Optional[str] = None,
):
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
# Main plotting routine
# ======================================================

def make_all_plots(eval_dir: Path, plot_dir: Path):
    plot_dir.mkdir(parents=True, exist_ok=True)

    crps_raw = pd.read_parquet(eval_dir / "crps_raw.parquet")
    crps_horizon = pd.read_parquet(eval_dir / "crps_by_horizon.parquet")
    crps_hour = pd.read_parquet(eval_dir / "crps_by_hour.parquet")
    crps_overall = pd.read_parquet(eval_dir / "crps_overall.parquet")

    plot_crps_by_horizon(
        crps_horizon,
        plot_dir / "crps_by_horizon.png",
    )

    plot_crps_by_hour(
        crps_hour,
        plot_dir / "crps_by_hour.png",
    )

    plot_overall_crps(
        crps_overall,
        plot_dir / "crps_overall.png",
    )

    # === CRPS-consistent residual diagnostics ===

    plot_pit_histogram(
        crps_raw,
        plot_dir / "pit_histogram.png",
    )

    plot_quantile_residuals_time(
        crps_raw,
        plot_dir / "quantile_residuals_time.png",
    )

    plot_quantile_residual_distribution(
        crps_raw,
        plot_dir / "quantile_residual_distribution.png",
    )

    plot_crps_vs_level(
        crps_raw,
        plot_dir / "crps_vs_price.png",
    )


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

    eval_dir = (
        Path(cfg.paths.outputs_dir)
        / target
        / f"horizon_{cfg.horizon}"
    )

    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")

    plot_dir = eval_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    make_all_plots(eval_dir, plot_dir)

    print(f"[PLOTS] Figures written to {plot_dir}")


if __name__ == "__main__":
    main()
