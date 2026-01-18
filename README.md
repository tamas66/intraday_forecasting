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

Ensure you have **Python 3.11 or higher** installed. It is highly recommended to use a virtual environment to manage dependencies:

```powershell
# Create a virtual environment
python -m venv venv

# Activate the environment
# On Windows:
.\venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

```
### 2. Setup
Before running the pipeline, initialize the local directory structure.
This script creates the required data/ and outputs/ folders, which are excluded from Git:
``` python src/config.py ```

Install the required Python dependencies using the provided requirements.txt file:
``` pip install -r requirements.txt ```

### 3. Project Structure

The project is organized to clearly separate data logic, modeling, and exploratory analysis:
thesis/
├── data/               # (Auto-generated) Local Parquet storage
├── docs/               # Thesis focus and documentation
├── notebooks/          # Jupyter notebooks for EDA and inspection
├── outputs/            # (Auto-generated) Plots and evaluation results
├── src/                # Source code
│   ├── helpers/        # Statistics, API calls, and data cleaning
│   ├── models/         # Parametric and ANN model architectures
│   ├── config.py       # Global settings and folder initialization
│   ├── data.py         # Data loading and processing logic
│   ├── train.py        # Model training entry point
│   └── evaluate.py     # Performance metric calculations
├── .gitignore          # Excludes data, cache, and large files
└── requirements.txt    # Python dependencies

## Intended Audience

This repository is best suited for:

- Power traders exploring quantitative research ideas
- Energy analysts building price intuition through data
- Researchers studying electricity market dynamics
- Data scientists transitioning into power markets

It is not intended to function as a turnkey trading system or signal generator.
