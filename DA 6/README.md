## 1. Project Overview

This repository contains the complete, reproducible laboratory implementation for **Experiment 06: Time-Series Analysis and Forecasting of Reported Crime Incidents by Time and Location using AR and ARIMA Models**.

The experiment formulates public safety incident tracking as a rigorous temporal count-forecasting problem. Raw, irregular incident records are extracted directly from official municipal public open data APIs, filtered for specific police districts, aggregated into regular weekly time series, and modeled using classical time-series techniques including **Naive Persistence**, **Autoregressive (AR)**, **Autoregressive Integrated Moving Average (ARIMA)**, **Seasonal ARIMA (SARIMA)**, and **Rolling-Origin Backtesting**.

All data modeling adheres strictly to chronological ordering to prevent temporal data leakage. All code cells in `DA6.ipynb` are self-contained, reproducible, and formatted without inline comments as per laboratory guidelines.

---

## 2. Directory Structure

```text
.
├── DA6.ipynb                          # Executable Jupyter Notebook (Zero-comment code implementation)
├── README.md                          # Comprehensive experiment guide & execution documentation
├── report.md                          # Detailed laboratory report in academic, humanized analytical tone
└── output/                            # Output folder containing all generated plots and manifests
    ├── fig01_raw_timeseries_split.png       # Raw weekly series with train/test boundary
    ├── fig02_rolling_stats.png              # Rolling mean and volatility dynamics
    ├── fig03_training_acf_pacf.png          # Autocorrelation and Partial Autocorrelation plots
    ├── fig04_forecast_comparison.png        # Forecast trajectories vs actuals with 95% CI
    ├── fig05_residual_diagnostics.png       # Residual distribution, ACF, and Ljung-Box test
    ├── fig06_cross_location_comparison.png  # Multi-location comparison (District 1 vs District 18)
    ├── fig07_rolling_origin_cv.png          # Walk-forward rolling-origin error across folds
    ├── manifest.json                        # Environment and experiment execution metadata
    └── analysis/                            # Structured CSV datasets and results
        ├── arima_candidate_grid.csv         # Candidate ARIMA orders ranked by AIC/BIC
        ├── location_comparison.csv          # Comparative metrics between District 1 & District 18
        ├── manifest.json                    # Execution manifest copy
        ├── model_comparison.csv             # Unified model performance comparison table
        ├── rolling_origin_results.csv       # Multi-fold backtesting error metrics
        └── test_predictions.csv             # Out-of-sample point forecasts and confidence intervals
```

---

## 3. Dataset Provenance and Public API Access

- **Source:** City of Chicago Open Data Portal / Chicago Police Department
- **Dataset Title:** Crimes - 2001 to Present
- **API Endpoint:** `https://data.cityofchicago.org/resource/ijzp-q8t2.json`
- **Socrata API Query Parameters:**
  - Endpoint: `https://data.cityofchicago.org/resource/ijzp-q8t2.json`
  - `$limit`: `50000`
  - `$where`: `date >= '2021-01-01T00:00:00.000' and date <= '2024-12-31T23:59:59.000' and (district='001' or district='018')`
  - `$order`: `date ASC`
- **Geographic Focus Units:**
  - **Primary District:** District 001 (Central / Downtown Loop)
  - **Secondary District:** District 018 (Near North / Magnificent Mile)
- **Temporal Aggregation Frequency:** Weekly starting Monday (`W-MON`)
- **Holdout Test Window:** Fixed chronologically at the last 12 weeks of the series ($H=12$)

---

## 4. Software Requirements and Environment Setup

This project is built and validated with Python 3.10+ (tested on Python 3.14).

### Required Python Libraries
- `pandas` (>= 2.0.0)
- `numpy` (>= 1.24.0)
- `matplotlib` (>= 3.7.0)
- `statsmodels` (>= 0.14.0)
- `scikit-learn` (>= 1.3.0)
- `scipy` (>= 1.10.0)
- `requests` (>= 2.28.0)
- `nbformat` & `nbclient` / `jupyter`

### Installation
To install the required dependencies in your local Python environment:

```bash
pip install pandas numpy matplotlib statsmodels scikit-learn scipy requests nbformat nbclient jupyter
```

---

## 5. Execution Instructions

### Running the Jupyter Notebook
You can open and execute the notebook interactively:

```bash
jupyter notebook DA6.ipynb
```

Alternatively, to execute the entire notebook headlessly from the terminal and update all figures and output CSVs:

```bash
python -m nbclient DA6.ipynb
```

### Execution Flow in `DA6.ipynb`:
1. **Configuration & Seed Setting:** Configures random seeds (`42`), paths, and query parameters.
2. **API Data Ingestion:** Fetches incident records directly from the Chicago Data Portal API.
3. **Time-Series Construction:** Parses timestamps, cleans missing values, filters target police districts, resamples to regular weekly frequency (`W-MON`), and fills missing intervals with true zero counts.
4. **Chronological Splitting:** Partitions the series into historical training data and a locked 12-week out-of-sample test horizon without data shuffling.
5. **Stationarity Diagnostics:** Computes the Augmented Dickey-Fuller (ADF) test and evaluates ACF/PACF on training data only.
6. **Model Estimation:**
   - Heuristic Naive Persistence baseline
   - Autoregressive model $\text{AR}(4)$ with constant and linear time trend
   - Grid search across candidate ARIMA orders $[(1,0,0), (2,0,0), (1,1,1), (2,1,1), (1,1,2), (2,1,2)]$ selecting optimal candidate by AIC
   - Seasonal ARIMA $\text{SARIMA}(1,1,1)\times(1,0,1)_{12}$
7. **Forecast Evaluation:** Computes MAE and RMSE on the held-out test period; extracts 95% Gaussian prediction intervals.
8. **Residual Diagnostics:** Analyzes residual time series, normality distributions, ACF, and Ljung-Box portmanteau tests.
9. **Cross-Location Comparison:** Replicates estimation on Chicago Police District 18 under identical evaluation constraints.
10. **Walk-Forward Cross-Validation:** Executes 7-fold rolling-origin backtesting to measure error stability over time.
11. **Artifact Export & Acceptance Tests:** Writes all plots to `output/`, all structured CSVs to `output/analysis/`, and validates assertions.

---

## 6. Model Results Summary

| Model | Order / Specification | Training AIC | Test MAE (Incidents/Wk) | Test RMSE | Ljung-Box $p$-value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive Persistence** | Last Observed Value | — | **18.583** | **29.996** | — |
| **ARIMA(1,1,1)** | $(p=1, d=1, q=1)$ | **1133.580** | **20.101** | **31.731** | **0.9987** |
| **SARIMA(1,1,1)x(1,0,1)₁₂** | $(1,1,1)\times(1,0,1)_{12}$ | **991.720** | **20.932** | **32.644** | **0.9986** |
| **AR(4)** | $p=4, \text{trend}='ct'$ | 1102.472 | 55.165 | 61.785 | 0.9990 |

- **Cross-Location Comparison:**
  - **District 1 (Central / Downtown):** Mean Weekly Incidents = 215.99 ($\sigma = 76.10$), ARIMA MAE = 20.101, RMSE = 31.731.
  - **District 18 (Near North / Magnificent Mile):** Mean Weekly Incidents = 218.74 ($\sigma = 51.12$), ARIMA MAE = 27.275, RMSE = 33.631.
- **Rolling-Origin Backtesting:** Mean MAE across 7 expanding folds = **29.993** ($\pm 19.117$), capturing seasonal demand fluctuations across different temporal regimes.

---

## 7. Responsible Analytics and Ethical Boundaries

1. **Measurement Reality:** The models forecast counts of *reported incidents* logged by administrative systems, which represent a composite of actual events, victim reporting propensity, and police deployment practices.
2. **No Individual Profiling:** Forecasts are strictly macro-level spatial counts (district-level aggregates) and must **never** be used for individual-level behavioral prediction, person-level risk scoring, or automated policing dispatch.
3. **Fairness and Spatial Context:** Crime counts naturally correlate with daytime commercial foot traffic, tourism density, and population turnover; raw counts must not be equated to community culpability.

---

## 8. Author Information

- **Student Name:** Raghav Mritunajaya KS
- **Registration Number:** 23MID0078
- **Program / School:** Integrated M.Tech / SCOPE, Vellore Institute of Technology, Vellore
- **Course:** MDI3003 - Advanced Predictive Analytics
- **Lab Instructor:** Dr. Durgesh Kumar
