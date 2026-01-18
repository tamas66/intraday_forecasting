# src/evaluation/compare.py - Enhanced with timing

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from evaluation.metrics import format_time

class ModelComparison:
    """Framework for comparing multiple models including training time"""
    
    def __init__(self):
        self.results = {}
    
    def add_model(
        self, 
        name: str, 
        metrics: Dict[str, float], 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        training_time: Optional[float] = None,
        inference_time: Optional[float] = None,
    ):
        """Add a model's results including timing information"""
        self.results[name] = {
            "metrics": metrics,
            "y_true": y_true,
            "y_pred": y_pred,
            "training_time": training_time,
            "inference_time": inference_time,
        }
    
    def get_comparison_table(self, include_timing: bool = True) -> pd.DataFrame:
        """Generate comparison table with optional timing metrics"""
        data = []
        for name, res in self.results.items():
            row = {"Model": name}
            row.update(res["metrics"])
            
            if include_timing and res.get("training_time"):
                row["training_time_formatted"] = format_time(res["training_time"])
            if include_timing and res.get("inference_time"):
                row["inference_time_formatted"] = format_time(res["inference_time"])
                
            data.append(row)
        
        df = pd.DataFrame(data)
        df = df.set_index("Model")
        return df
    
    def plot_accuracy_vs_time(self, accuracy_metric: str = "RMSE"):
        """
        Plot accuracy vs training time scatter plot.
        Lower RMSE and lower training time is better (bottom-left corner).
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = []
        times = []
        accuracies = []
        
        for name, res in self.results.items():
            if res.get("training_time") and accuracy_metric in res["metrics"]:
                models.append(name)
                times.append(res["training_time"])
                accuracies.append(res["metrics"][accuracy_metric])
        
        if not models:
            print("No timing data available for plotting")
            return
        
        # Create scatter plot
        scatter = ax.scatter(times, accuracies, s=200, alpha=0.6, c=range(len(models)), cmap='tab10')
        
        # Annotate points
        for i, model in enumerate(models):
            ax.annotate(model, (times[i], accuracies[i]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Training Time (seconds)', fontsize=12)
        ax.set_ylabel(f'{accuracy_metric}', fontsize=12)
        ax.set_title(f'Model Efficiency: {accuracy_metric} vs Training Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add quadrant lines (median)
        if len(times) > 1:
            ax.axvline(np.median(times), color='red', linestyle='--', alpha=0.3, label='Median Time')
            ax.axhline(np.median(accuracies), color='blue', linestyle='--', alpha=0.3, label='Median Accuracy')
            ax.legend()
        
        # Annotate best corner
        ax.text(0.02, 0.98, '← Faster, More Accurate', 
                transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
    
    def plot_timing_breakdown(self):
        """Plot training time comparison across models"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Absolute times
        models = []
        times = []
        
        for name, res in self.results.items():
            if res.get("training_time"):
                models.append(name)
                times.append(res["training_time"])
        
        if models:
            colors = plt.cm.viridis(np.linspace(0, 0.8, len(models)))
            bars = ax1.barh(models, times, color=colors)
            ax1.set_xlabel('Training Time (seconds)', fontsize=11)
            ax1.set_title('Absolute Training Times', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for bar, time_val in zip(bars, times):
                width = bar.get_width()
                ax1.text(width, bar.get_y() + bar.get_height()/2, 
                        f'  {format_time(time_val)}',
                        ha='left', va='center', fontweight='bold')
        
        # Relative times (normalized)
        if models and len(times) > 1:
            min_time = min(times)
            relative_times = [t / min_time for t in times]
            
            bars = ax2.barh(models, relative_times, color=colors)
            ax2.set_xlabel('Relative Training Time (×fastest)', fontsize=11)
            ax2.set_title('Relative Training Times', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for bar, rel_time in zip(bars, relative_times):
                width = bar.get_width()
                ax2.text(width, bar.get_y() + bar.get_height()/2, 
                        f'  {rel_time:.2f}×',
                        ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_efficiency_frontier(self):
        """
        Plot Pareto frontier showing trade-off between accuracy and speed.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = []
        times = []
        maes = []
        
        for name, res in self.results.items():
            if res.get("training_time") and "MAE" in res["metrics"]:
                models.append(name)
                times.append(res["training_time"])
                maes.append(res["metrics"]["MAE"])
        
        if not models:
            print("No data available for efficiency frontier")
            return
        
        # Normalize both metrics to [0, 1] for fair comparison
        times_norm = np.array(times) / max(times)
        maes_norm = np.array(maes) / max(maes)
        
        # Calculate efficiency score (lower is better for both)
        efficiency_scores = times_norm + maes_norm  # Simple sum
        
        # Plot
        scatter = ax.scatter(times, maes, s=300, c=efficiency_scores, 
                           cmap='RdYlGn_r', alpha=0.7, edgecolors='black', linewidth=2)
        
        # Annotate
        for i, model in enumerate(models):
            ax.annotate(model, (times[i], maes[i]), 
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
        
        ax.set_xlabel('Training Time (seconds)', fontsize=12)
        ax.set_ylabel('MAE (Mean Absolute Error)', fontsize=12)
        ax.set_title('Efficiency Frontier: Accuracy vs Speed Trade-off', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Combined Score\n(lower = better)', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def get_timing_summary(self) -> pd.DataFrame:
        """Get detailed timing summary table"""
        data = []
        
        for name, res in self.results.items():
            row = {"Model": name}
            
            if res.get("training_time"):
                row["Training Time"] = format_time(res["training_time"])
                row["Training (sec)"] = res["training_time"]
            
            if res.get("inference_time"):
                row["Inference Time"] = format_time(res["inference_time"])
                row["Time per Prediction"] = format_time(res["inference_time"] / len(res["y_pred"]))
            
            # Add efficiency metrics
            if res.get("training_time") and "MAE" in res["metrics"]:
                row["MAE"] = res["metrics"]["MAE"]
                row["MAE per Second"] = res["metrics"]["MAE"] / res["training_time"]
            
            data.append(row)
        
        return pd.DataFrame(data).set_index("Model")