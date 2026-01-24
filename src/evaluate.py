from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import hydra

from evaluation.compare import run_comparison


# ======================================================
# PLOTTING
# ======================================================

def _plot_fixed_split(cfg: DictConfig, results: dict, target: str):
    plt.figure(figsize=(14, 6))
    fixed_models = {
        k: v for k, v in results.items()
        if v.get("evaluation") == "fixed_split"
    }
    if not fixed_models:
        return

    first = next(iter(fixed_models.values()))
    timestamps = first["timestamps"]
    y_true = first["y_true"]

    plt.plot(timestamps, y_true, label="Actual", linewidth=2)

    for model_name, res in fixed_models.items():
        plt.plot(res["timestamps"], res["y_pred"], label=model_name, alpha=0.85)

    plt.title(f"Fixed-split comparison – {target}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    out = Path(cfg.data.paths.results_dir) / f"forecast_comparison_fixed_{target}.png"
    plt.savefig(out, dpi=150)
    plt.close()


def _plot_rolling_arx(cfg: DictConfig, target: str, max_points: int = 24 * 60):
    """
    Plot Rolling / Expanding ARX forecast vs actual.
    Uses the saved CSV from rolling_arx.py.
    """
    csv_path = Path(cfg.data.paths.results_dir) / f"rolling_arx_{target}.csv"
    if not csv_path.exists():
        print(f"[Plot] Rolling ARX CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")

    if len(df) == 0:
        print("[Plot] Rolling ARX CSV is empty.")
        return

    if len(df) > max_points:
        df = df.iloc[-max_points:]

    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df["y_true"], label="Actual", linewidth=2)
    plt.plot(df.index, df["y_pred"], label="Rolling ARX", alpha=0.85)

    plt.title(f"Rolling ARX (expanding refit) – {target}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    out = Path(cfg.data.paths.results_dir) / f"forecast_rolling_arx_{target}.png"
    plt.savefig(out, dpi=150)
    plt.close()

    print(f"[Plot] Saved rolling ARX plot to: {out}")


# ======================================================
# SUMMARY
# ======================================================

def _build_summary_table(results: dict) -> pd.DataFrame:
    rows = []

    for model, res in results.items():
        rows.append(
            {
                "model": model,
                "evaluation": res.get("evaluation", "fixed_split"),
                "rmse": res["rmse"],
                "mae": res["mae"],
                "train_time_sec": res["train_time"],
            }
        )

    df = pd.DataFrame(rows).set_index("model")
    return df.sort_values(["evaluation", "rmse"])


# ======================================================
# MAIN
# ======================================================

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    target = cfg.target

    results_dir = Path(cfg.data.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results = run_comparison(
        cfg=cfg,
        target=target,
        force_retrain=False,
    )

    _plot_fixed_split(cfg, results, target)
    _plot_rolling_arx(cfg, target)

    summary = _build_summary_table(results)

    out_csv = results_dir / f"model_comparison_{target}.csv"
    summary.to_csv(out_csv)

    print("\nModel comparison summary:\n")
    print(summary.round(4))
    print(f"\nSaved summary to: {out_csv}")


if __name__ == "__main__":
    main()