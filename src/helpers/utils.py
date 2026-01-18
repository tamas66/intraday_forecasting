# src/utils.py 
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import logging

def save_results(results: dict, filename: str, results_dir: Path):
    """Save model results to JSON"""
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Convert numpy types to native Python types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return obj
    
    results_json = json.loads(
        json.dumps(results, default=convert)
    )
    
    filepath = results_dir / filename
    with open(filepath, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"Results saved to {filepath}")

def create_experiment_id():
    """Generate unique experiment ID"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def setup_logging(experiment_id: str, log_dir: Path):
    """Setup logging configuration"""
    log_dir.mkdir(exist_ok=True, parents=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'{experiment_id}.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)