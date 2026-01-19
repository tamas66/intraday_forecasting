# British Energy Price Forecasting (Research)

## Overview

This repository contains a **research-oriented framework for forecasting British electricity prices**, built using high-resolution market and system data sourced from the **EnAppSys API**. The project focuses on understanding and modeling the structural drivers of power prices in Denmark rather than producing production-ready trading signals.

The work is intended for **energy traders, analysts, and power market researchers** interested in:
- Day-ahead price formation
- The impact of generation availability and renewables
- Cross-market and system fundamentals
- Feature engineering and data alignment for power price models

The emphasis is on **data transparency, reproducibility, and explainability**, not automated execution or live trading.

---

## Data Sources

The project relies primarily on **EnAppSys Chart and Bulk APIs**, which provide access to a wide range of European power market fundamentals, including:

- Day-ahead electricity prices
- Generation and renewable forecasts (wind, solar)
- Unit availability and outages
- System-level supply and demand indicators
- Timezone- and resolution-aware historical data

All data ingestion is handled programmatically, with careful control over:
- Time alignment (local vs UTC)
- Resolution consistency (hourly, daily, etc.)
- Metadata preservation (units, last-updated timestamps)

> **Note:** Access to EnAppSys data requires valid credentials. No credentials or proprietary raw data are included in this repository.

---

## Project Scope & Philosophy

This project is **explicitly research-focused**:

- No live trading, order execution, or brokerage integration
- No claims of profitability or trading performance
- Models are evaluated for **forecast quality**, not P&L

The goal is to explore questions such as:
- Which fundamentals matter most for British price formation?
- How stable are price–feature relationships across regimes?
- How do forecast horizons affect feature relevance?
- What is the trade-off between model complexity and interpretability?

This makes the repository suitable for **idea generation, model comparison, and methodological experimentation**.

---

## Architecture & Workflow

At a high level, the workflow follows:

1. **Data Ingestion**
   - Structured API calls to EnAppSys (Chart + Bulk APIs)
   - Parameterized by market, resolution, and time range

2. **Data Normalization**
   - Consistent timestamp handling
   - Standardized column naming and units
   - Missing data and availability checks

3. **Feature Engineering**
   - Lagged prices and fundamentals
   - Renewable penetration metrics
   - System stress and availability indicators

4. **Modeling & Evaluation**
   - Baseline statistical models
   - Interpretable ML models where appropriate
   - Out-of-sample and rolling-window evaluation

Each step is modular to encourage experimentation and reuse.

---

## 🛠 Setup and Execution

### 1. Prerequisites
This project requires **Python 3.11 or higher**. To ensure a reproducible research environment and avoid dependency conflicts with global packages, it is strongly recommended to use a Python virtual environment (`venv`).

```powershell
# Navigate to the project root
cd C:\Users\tamas\Documents\GitHub\thesis

# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
# On Windows (PowerShell):
.\venv\Scripts\activate

# On Windows (Command Prompt):
.\venv\Scripts\Projects\activate.bat

# On macOS/Linux:
source venv/bin/activate

```

### 2. Initialization and Installation

The project follows a "clean repository" philosophy where ephemeral data and large binary outputs are generated locally rather than stored in version control.

**Step A: Initialize Directory Structure**
Run the configuration script to build the local file system. This script creates the `data/` and `outputs/` hierarchies required by the source code.

```powershell
python src/config.py

```

**Step B: Install Dependencies**
With the virtual environment active, install the necessary libraries for data processing, statistical modeling, and deep learning.

```powershell
pip install --upgrade pip
pip install -r requirements.txt

```

> **Note on Environment Variables:** This project utilizes a `.env` file for sensitive configurations (such as API keys for ENTSO-E or database credentials). Ensure a `.env` file exists in the root directory before attempting to fetch live data.

---

### 3. Project Structure

The repository is structured to separate raw data, processed features, and modeling logic. This modularity allows for independent testing of the forecasting components.

```text
thesis/
├── data/               # Local storage for datasets (Git-ignored)
│   ├── raw/            # Original yearly Parquet files (2015-2025)
│   ├── processed/      # Cleaned and feature-engineered Parquet files
│   └── final/          # Merged datasets ready for model input
├── docs/               # Academic documentation and thesis focus papers
├── notebooks/          # Exploratory Data Analysis (EDA) and prototyping
│   ├── 01_Inspection.ipynb     # Initial data validation
│   ├── 02_Danish_EDA.ipynb     # Price/demand analysis for DK region
│   └── 03_British_EDA.ipynb    # Price/demand analysis for UK region
├── outputs/            # Generated figures, logs, and model weights
│   ├── eda/            # Statistical plots (ACF/PACF, correlations)
│   └── models/         # Saved model states and performance metrics
├── src/                # Core Python source code
│   ├── helpers/        # Utility modules (API, statistics, eval metrics)
│   ├── models/         # Architecture definitions (Parametric & ANN)
│   ├── config.py       # Global constants and system initialization
│   ├── data.py         # ETL pipeline logic
│   ├── train.py        # Main training loop script
│   └── evaluate.py     # Model validation and comparison logic
├── .gitignore          # Rules for excluding large data and cache files
├── README.md           # Project overview and setup instructions
└── requirements.txt    # List of required Python packages

```

---

### 4. Running the Pipeline

The forecasting workflow is designed to be executed sequentially. Each stage relies on the outputs generated by the previous step.

**Stage 1: Data Acquisition and Feature Engineering**
Transform the raw yearly Parquet files into a unified format suitable for time-series modeling.

```powershell
python src/preprocess.py

```

**Stage 2: Model Training**
Fit both the Parametric (statistical) and ANN (Artificial Neural Network) models. This will generate model weights in the `outputs/models/` directory.

```powershell
python src/train.py

```

**Stage 3: Evaluation and Benchmarking**
Run the evaluation suite to calculate performance metrics (MAE, RMSE, sMAPE) and generate comparison visualizations between the different forecasting approaches.

```powershell
python src/evaluate.py

```

```

```

## Intended Audience

This repository is best suited for:

- Power traders exploring quantitative research ideas
- Energy analysts building price intuition through data
- Researchers studying electricity market dynamics
- Data scientists transitioning into power markets

It is not intended to function as a turnkey trading system or signal generator.
