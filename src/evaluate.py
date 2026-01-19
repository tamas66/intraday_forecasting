from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt

from evaluation.compare import run_comparison
from config import RESULTS_DIR


def _plot_fixed_split(results: dict, target: str):
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

    out = RESULTS_DIR / f"forecast_comparison_fixed_{target}.png"
    plt.savefig(out, dpi=150)
    plt.close()


def _load_rolling_arx_meta(target: str):
    """
    Load Rolling ARX metadata if it exists.
    """
    meta_path = RESULTS_DIR / f"rolling_arx_{target}_meta.json"
    if not meta_path.exists():
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return {
        "model": "RollingARX",
        "evaluation": meta.get("evaluation", "expanding_refit"),
        "rmse": meta.get("rmse"),
        "mae": meta.get("mae"),
        "train_time_sec": meta.get("total_fit_time_sec"),
    }


def _build_summary_table(results: dict, target: str) -> pd.DataFrame:
    rows = []

    # Fixed-split models
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

    # Rolling ARX (if available)
    rolling_arx = _load_rolling_arx_meta(target)
    if rolling_arx is not None:
        rows.append(rolling_arx)

    df = pd.DataFrame(rows).set_index("model")
    return df.sort_values(["evaluation", "rmse"])


def main(target: str = "level"):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Fixed-split comparison (SARIMAX, GARCH-X, LSTM)
    # --------------------------------------------------
    results = run_comparison(target=target, force_retrain=False)

    # --------------------------------------------------
    # Plots
    # --------------------------------------------------
    _plot_fixed_split(results, target)

    # --------------------------------------------------
    # Summary table
    # --------------------------------------------------
    summary = _build_summary_table(results, target)

    out_csv = RESULTS_DIR / f"model_comparison_{target}.csv"
    summary.to_csv(out_csv)

    print("\nModel comparison summary:\n")
    print(summary.round(4))
    print(f"\nSaved summary to: {out_csv}")


if __name__ == "__main__":
    main(target="level")
