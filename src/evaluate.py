# src/evaluation/evaluate.py

import matplotlib.pyplot as plt
import pandas as pd
import sys
from pathlib import Path
from evaluation.metrics import (
    plot_mean_diagnostics,
    plot_conditional_volatility,
    plot_regime_probabilities,
)
from data import load_parametric_dataset
from models.parametric_model import (
    fit_sarimax,
    fit_sarimax_garch,
    fit_markov_switching,
    SARIMAXConfig,
    GARCHConfig,
    MarkovConfig,
)


# ======================================================
# GENERIC MODEL EVALUATION
# ======================================================

def evaluate_parametric_model(
    df: pd.DataFrame,
    model_result,
    model_type: str,
):
    """
    Basic evaluation and diagnostic plots for parametric models.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_parametric_dataset().
    model_result :
        Fitted model object or dictionary (SARIMAX+GARCH).
    model_type : {"sarimax", "sarimax_garch", "markov"}
        Type of model being evaluated.
    """

    y = df["y"]

    # --------------------------------------------------
    # SARIMAX
    # --------------------------------------------------
    if model_type == "sarimax":
        res = model_result

        print("=== SARIMAX ===")
        print(f"LogLik: {res.llf:.2f}")
        print(f"AIC:    {res.aic:.2f}")
        print(f"BIC:    {res.bic:.2f}")

        fitted = res.fittedvalues
        resid = res.resid

        plot_mean_diagnostics(y, fitted, resid, title="SARIMAX")

    # --------------------------------------------------
    # SARIMAX + GARCH
    # --------------------------------------------------
    elif model_type == "sarimax_garch":
        sarimax_res = model_result["sarimax"]
        garch_res = model_result["garch"]

        print("=== SARIMAX (Mean) ===")
        print(f"AIC: {sarimax_res.aic:.2f}")

        print("\n=== GARCH (Variance) ===")
        print(f"LogLik: {garch_res.loglikelihood:.2f}")
        print(f"AIC:    {garch_res.aic:.2f}")
        print(f"BIC:    {garch_res.bic:.2f}")

        fitted = sarimax_res.fittedvalues
        resid = sarimax_res.resid

        plot_mean_diagnostics(y, fitted, resid, title="SARIMAX + GARCH")
        plot_conditional_volatility(garch_res)

    # --------------------------------------------------
    # MARKOV-SWITCHING
    # --------------------------------------------------
    elif model_type == "markov":
        res = model_result

        print("=== Markov-Switching ARX ===")
        print(f"LogLik: {res.llf:.2f}")
        print(f"AIC:    {res.aic:.2f}")
        print(f"BIC:    {res.bic:.2f}")

        fitted = res.fittedvalues
        resid = y - fitted

        plot_mean_diagnostics(y, fitted, resid, title="Markov-Switching ARX")
        plot_regime_probabilities(res)

    else:
        raise ValueError("model_type must be 'sarimax', 'sarimax_garch', or 'markov'")





# ======================================================
# MAIN EXECUTION
# ======================================================

# src/evaluation/evaluate.py - Updated __main__ with timing

if __name__ == "__main__":
    
    from config import DataConfig, ExperimentConfig
    from evaluation.metrics import calculate_forecast_metrics, format_time
    from evaluation.compare import ModelComparison
    from helpers.utils import save_results, create_experiment_id
    import time
    
    print("=" * 70)
    print("PARAMETRIC ELECTRICITY PRICE MODEL EVALUATION PIPELINE")
    print("=" * 70)
    
 # ------------------------------------------------------------------
    # 1. SETUP
    # ------------------------------------------------------------------
    
    experiment_id = create_experiment_id()
    print(f"\nExperiment ID: {experiment_id}")
    
    # Configuration
    data_config = DataConfig(
        processed_data_path=Path("data/processed/electricity_features.csv"),
        target="level",
        train_end_date="2023-12-31",
        test_start_date="2024-01-01",
    )
    
    exp_config = ExperimentConfig(
        data=data_config,
        results_dir=Path(f"results/{experiment_id}"),
        plot_dir=Path(f"plots/{experiment_id}"),
    )
    
    sarimax_cfg = SARIMAXConfig(order=(1, 0, 1), seasonal_order=(1, 0, 1, 24), trend="c")
    garch_cfg = GARCHConfig(p=1, q=1, dist="t", use_exogenous=True)
    markov_cfg = MarkovConfig(k_regimes=2, order=1, switching_variance=True)
    
    # ------------------------------------------------------------------
    # 2. DATA LOADING
    # ------------------------------------------------------------------
    
    print("\n[1/5] Loading data...")
    
    train_df, test_df = load_parametric_dataset(
        file_path=str(data_config.processed_data_path),
        target=data_config.target,
        drop_na=True,
        train_end_date=data_config.train_end_date,
        test_start_date=data_config.test_start_date,
        verbose=True,
    )
    
    # ------------------------------------------------------------------
    # 3. MODEL FITTING WITH TIMING
    # ------------------------------------------------------------------
    
    print("\n[2/5] Fitting models on training data...")
    
    models = {}
    training_times = {}
    
    # SARIMAX
    print("\n  [A] SARIMAX...")
    try:
        model, train_time = fit_sarimax(train_df, sarimax_cfg, return_time=True)
        models["SARIMAX"] = model
        training_times["SARIMAX"] = train_time
        print(f"    ✓ Success - Training time: {format_time(train_time)}")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
    
    # SARIMAX + GARCH
    print("\n  [B] SARIMAX + GARCH...")
    try:
        model = fit_sarimax_garch(train_df, sarimax_cfg, garch_cfg, return_time=True)
        models["SARIMAX+GARCH"] = model
        training_times["SARIMAX+GARCH"] = model["timing"]["total_time"]
        print(f"    ✓ Success - Training time: {format_time(model['timing']['total_time'])}")
        print(f"      - SARIMAX: {format_time(model['timing']['sarimax_time'])}")
        print(f"      - GARCH: {format_time(model['timing']['garch_time'])}")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
    
    # Markov-Switching
    print("\n  [C] Markov-Switching...")
    try:
        model, train_time = fit_markov_switching(train_df, markov_cfg, return_time=True)
        models["Markov-Switching"] = model
        training_times["Markov-Switching"] = train_time
        print(f"    ✓ Success - Training time: {format_time(train_time)}")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
    
    # ------------------------------------------------------------------
    # 5. OUT-OF-SAMPLE FORECASTING WITH TIMING
    # ------------------------------------------------------------------
    
    print("\n[4/5] Out-of-sample forecasting...")
    
    comparison = ModelComparison()
    y_test = test_df["y"].values
    
    for name, model in models.items():
        print(f"\n  Forecasting with {name}...")
        
        try:
            # Time the inference
            inference_start = time.time()
            
            if name == "SARIMAX":
                forecast = model.forecast(
                    steps=len(test_df), 
                    exog=_prepare_exog_data(test_df, MEAN_EXOG_COLS)
                )
            elif name == "SARIMAX+GARCH":
                forecast = model["sarimax"].forecast(
                    steps=len(test_df),
                    exog=_prepare_exog_data(test_df, MEAN_EXOG_COLS)
                )
            else:  # Markov
                exog_cols = ["y_lag1", "y_lag24"] + [c for c in MEAN_EXOG_COLS if c in test_df.columns]
                forecast = model.forecast(
                    steps=len(test_df),
                    exog=_prepare_exog_data(test_df, exog_cols)
                )
            
            inference_time = time.time() - inference_start
            
            # Calculate metrics with timing
            metrics = calculate_forecast_metrics(
                y_test, 
                forecast,
                training_time=training_times.get(name),
                inference_time=inference_time
            )
            
            comparison.add_model(
                name, 
                metrics, 
                y_test, 
                forecast,
                training_time=training_times.get(name),
                inference_time=inference_time
            )
            
            print(f"    MAE: {metrics['MAE']:.2f}")
            print(f"    RMSE: {metrics['RMSE']:.2f}")
            print(f"    MAPE: {metrics['MAPE']:.2f}%")
            print(f"    Training time: {format_time(training_times.get(name, 0))}")
            print(f"    Inference time: {format_time(inference_time)}")
            print(f"    Time per prediction: {metrics.get('inference_time_per_prediction_ms', 0):.2f}ms")
            
        except Exception as e:
            print(f"    ✗ Forecasting failed: {e}")
    
    # ------------------------------------------------------------------
    # 6. MODEL COMPARISON WITH TIMING ANALYSIS
    # ------------------------------------------------------------------
    
    print("\n[5/5] Model comparison...")
    print("\n" + "="*70)
    print("OUT-OF-SAMPLE PERFORMANCE COMPARISON")
    print("="*70)
    
    # Full comparison table
    comparison_table = comparison.get_comparison_table(include_timing=True)
    print(comparison_table.round(3))
    
    # Timing-specific summary
    print("\n" + "="*70)
    print("TIMING SUMMARY")
    print("="*70)
    timing_summary = comparison.get_timing_summary()
    print(timing_summary)
    
    # Save results
    if exp_config.save_results:
        save_results(
            comparison_table.to_dict(),
            f"comparison_{experiment_id}.json",
            exp_config.results_dir
        )
        save_results(
            timing_summary.to_dict(),
            f"timing_{experiment_id}.json",
            exp_config.results_dir
        )
    
    # ------------------------------------------------------------------
    # 7. TIMING VISUALIZATIONS
    # ------------------------------------------------------------------
    
    print("\nGenerating comparison plots...")
    
    # Standard plots
    comparison.plot_forecast_comparison()
    comparison.plot_error_distributions()
    
    # Timing-specific plots
    print("\nGenerating timing analysis plots...")
    comparison.plot_timing_breakdown()
    comparison.plot_accuracy_vs_time(accuracy_metric="RMSE")
    comparison.plot_efficiency_frontier()
    
    # Metric heatmap with timing
    comparison.plot_metric_heatmap(
        metrics_subset=["MAE", "RMSE", "MAPE", "R2", "training_time_seconds"]
    )
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    
    # Print winner summary
    print("\n" + "="*70)
    print("WINNER SUMMARY")
    print("="*70)
    
    comp_df = comparison.get_comparison_table(include_timing=False)
    print(f"Best MAE: {comp_df['MAE'].idxmin()} ({comp_df['MAE'].min():.2f})")
    print(f"Best RMSE: {comp_df['RMSE'].idxmin()} ({comp_df['RMSE'].min():.2f})")
    print(f"Best R²: {comp_df['R2'].idxmax()} ({comp_df['R2'].max():.3f})")
    
    if 'training_time_seconds' in comp_df.columns:
        fastest = comp_df['training_time_seconds'].idxmin()
        print(f"Fastest training: {fastest} ({format_time(comp_df.loc[fastest, 'training_time_seconds'])})")
    
    if 'efficiency_score' in comp_df.columns:
        most_efficient = comp_df['efficiency_score'].idxmax()
        print(f"Most efficient: {most_efficient} (score: {comp_df.loc[most_efficient, 'efficiency_score']:.3f})")