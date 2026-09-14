# Experiment 08: Agricultural Predictive Analytics
## Crop Yield Prediction with a Guided Crop-Label Classification Extension

**Course:** MDI3003 — Advanced Predictive Analytics  
**Laboratory Assignment:** Digital Assignment 08 (DA 8)  
**Student Name:** Raghav Mritunajaya KS  
**Registration Number:** 23MID0078  
**Department:** School of Computer Science and Engineering (SCOPE), VIT Vellore  
**Academic Term:** Fall Semester 2026–2027  

---

### Project Overview
This repository contains the complete implementation, rigorous prospective backtesting, diagnostic error audits, and academic reporting for **Experiment 08: Agricultural Predictive Analytics**. The study investigates prospective district-level rice yield forecasting across Indian agro-ecological zones, coupled with a guided soil and climate multiclass crop recommendation classification extension.

The core regression pipeline utilizes the versioned ICRISAT district panel dataset (D5, DOI: `10.17632/ywp3y5j9vv.1`, Souryabrata Mohapatra), dynamically retrieved through the Mendeley Data Public REST API. Across an unbalanced panel of 12,751 district-harvest observations spanning 26 continuous years (1990–2015), 20 states, and 555 unique districts, a strict prospective chronological evaluation boundary is established:
- **Training Set (1990–2011):** 10,541 observations across 22 earlier historical years.
- **Validation Set (2012–2013):** 1,096 observations across 2 intermediate prospective years.
- **Locked Test Set (2014–2015):** 1,114 observations across the final 2 untouched evaluation years.

To eliminate data leakage, all categorical encodings and numerical normalizations are learned exclusively on training rows. The feature space strictly adheres to a prospective pre-harvest whitelist: `['state', 'district', 'season', 'year']`. Post-harvest production figures and harvested area are explicitly barred to prevent mathematical target leakage.

Candidate pipelines include:
1. **Median Baseline:** `DummyRegressor(strategy='median')` — the optimal constant estimator under Mean Absolute Error (MAE) loss.
2. **Ridge Linear Trend Model:** `Ridge(alpha=1.0)` — estimates district-specific spatial fixed offsets alongside an additive secular global year trend.
3. **Decision Tree Regressor:** `DecisionTreeRegressor(max_depth=6, min_samples_leaf=10)`.
4. **Random Forest Regressor:** `RandomForestRegressor(n_estimators=60, max_depth=12, min_samples_leaf=5)`.
5. **Depth Ablation:** `RandomForestRegressor(n_estimators=60, max_depth=6, min_samples_leaf=5)`.
6. **Feature Ablation:** Evaluating performance degradation upon omitting district spatial keys.
7. **Prospective Rolling Origins:** 5 temporal cutoffs (2009–2013) within development data to audit forecasting stability.

The guided extension explores soil nutrient and climatic crop recommendation using the Atharva Ingle dataset (D3, 2,200 rows across 22 crop classes), benchmarking a Majority Baseline, Multinomial Logistic Regression, and a 60-tree Random Forest Classifier.

---

### Repository Structure

```text
DA 8/
├── DA8.ipynb                                         # Main Jupyter Notebook (executed, zero comments)
├── README.md                                         # Complete instructions & documentation
├── report.md                                         # Academic laboratory report
├── build_notebook.py                                 # Python script to build and execute DA8.ipynb
├── data/                                             # Public API acquired and canonicalized datasets
│   ├── icrisat_d5_raw.xls                            # Raw D5 Excel export from Mendeley Data API
│   ├── rice_canonical.csv                            # Cleaned canonical rice yield panel (12,751 rows)
│   ├── crop_recommendation_raw.csv                   # Raw D3 CSV download
│   └── crop_labels_canonical.csv                     # Cleaned canonical crop recommendation dataset
├── output/                                           # Output artifacts directory
│   ├── TEST_LOCK                                     # One-time evaluation lockfile
│   ├── selection.json                                # Model selection record and configuration hash
│   ├── *.png                                         # Generated figures (mirrored at top-level)
│   ├── figures/                                      # High-resolution visualization figures
│   │   ├── target.png                                # Training yield distribution histogram
│   │   ├── feature.png                               # Training harvest year coverage bar chart
│   │   ├── comparison.png                            # Validation MAE candidate comparison
│   │   ├── depth_ablation.png                        # Random Forest tree depth ablation
│   │   ├── rolling_origins.png                       # Prospective rolling origins MAE trajectory
│   │   ├── actual_predicted.png                      # Locked test actual vs predicted scatter
│   │   ├── residuals.png                             # Locked test residual diagnostic scatter
│   │   ├── crop_distribution.png                     # D3 crop label distribution
│   │   ├── soil_ph_distribution.png                  # D3 training soil pH distribution
│   │   ├── classification_metrics.png                # Classification Macro F1 score comparison
│   │   └── confusion.png                             # 22-class classification confusion matrix
│   ├── analysis/                                     # Analytical CSV outputs and reports
│   │   ├── 23MID0078_Lab08_Validation_Results.csv    # Validation results (Registration ID)
│   │   ├── 23MID0078_Lab08_Test_Results.csv          # Locked test results (Registration ID)
│   │   ├── 23MID0078_Lab08_Error_Analysis.csv        # Top 5 failure case diagnostics
│   │   ├── RegistrationNumber_Lab08_Validation_Results.csv
│   │   ├── RegistrationNumber_Lab08_Test_Results.csv
│   │   ├── RegistrationNumber_Lab08_Error_Analysis.csv
│   │   ├── dataset_profile.csv                       # Partition statistics across splits
│   │   ├── split_manifest.csv                        # Exact row_id to split assignments
│   │   ├── feature_ablation.csv                      # Spatial feature removal ablation
│   │   ├── rolling_origins.csv                       # 5-cutoff rolling origins temporal metrics
│   │   ├── year_robustness.csv                       # Disaggregated test performance by year
│   │   ├── classification_validation_results.csv     # D3 classification validation scores
│   │   ├── classification_test_results.csv           # D3 classification test scores
│   │   ├── acceptance.json                           # Reproducibility audit checklist
│   │   ├── per_class.json                            # 22-class precision/recall/F1 metrics
│   │   └── versions.json                             # Exact software environment snapshot
│   └── models/                                       # Serialized pipeline bundles
│       ├── selected_bundle.joblib                    # Locked final model bundle
│       ├── ridge_trend.joblib                        # Fitted Ridge pipeline
│       ├── forest.joblib                             # Fitted Random Forest pipeline
│       ├── forest_depth6.joblib                      # Fitted shallow Random Forest pipeline
│       ├── tree.joblib                               # Fitted Decision Tree pipeline
│       └── median.joblib                             # Fitted Median Baseline pipeline
```

---

### Prerequisites and Software Environment

The pipeline was executed and validated under Python 3.14 / Python 3.11+ on Windows 64-bit architecture.

#### Required Dependencies
```bash
pip install numpy pandas scikit-learn matplotlib joblib xlrd nbformat nbclient
```

Key library versions recorded in `output/analysis/versions.json`:
- **Python:** 3.14.0a2 / 3.11+
- **scikit-learn:** 1.8.0
- **pandas:** 2.3.3
- **numpy:** 2.4.2
- **matplotlib:** 3.10.8
- **joblib:** 1.5.3
- **xlrd:** 2.0.2

---

### Execution Instructions

#### Method 1: Running via build_notebook.py
To programmatically generate and execute `DA8.ipynb` with all outputs, plots, and zero comments:
```bash
python build_notebook.py
```

#### Method 2: Running the Complete Jupyter Notebook
To execute or interact with the notebook directly:
```bash
jupyter notebook DA8.ipynb
```
Select **Kernel -> Restart & Run All**. The notebook will:
1. Query the Mendeley Data Public REST API to download the D5 dataset and fetch D3.
2. Build the canonical representations in `data/`.
3. Perform chronological partitioning and training-only feature transformation.
4. Fit and compare all candidate regressors, selecting the minimum validation MAE pipeline.
5. Create `output/TEST_LOCK` and evaluate the selected model once on the locked 2014–2015 test set.
6. Conduct diagnostic qualitative analysis on the top 5 error cases.
7. Execute the 22-class crop recommendation classification extension.
8. Save all plots to `output/figures/` and all CSVs to `output/analysis/`.

---

### Key Experimental Findings Summary

#### 1. Core Regression Prospective Validation (2012–2013)
| Candidate Model | Validation MAE (t/ha) | Validation RMSE (t/ha) | Validation R² | Fit Time (s) | Selection Decision |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ridge Linear Trend** | **0.4770** | **0.6452** | **0.6641** | **0.0327** | **Selected (Minimum MAE)** |
| Random Forest (depth 12) | 0.6196 | 0.7874 | 0.4998 | 0.4369 | Sub-optimal extrapolation |
| Random Forest (depth 6) | 0.7555 | 0.9258 | 0.3084 | 0.2278 | Underfitted depth ablation |
| Decision Tree (depth 6) | 0.7727 | 0.9523 | 0.2683 | 0.0203 | Step-function plateau |
| Median Constant Baseline | 0.9816 | 1.2391 | -0.2389 | 0.0185 | Unconditional benchmark |

*Why Ridge Trend Outperformed Ensembles:* Tree ensembles partition input space and assign terminal leaf sample means, making them fundamentally incapable of extrapolating beyond historical training target ranges. Because Indian agricultural yields exhibit a secular upward trend across 1990–2015 due to high-yielding varieties and mechanization, the Ridge model’s positive linear year coefficient effectively extrapolates yield growth, while district fixed effects capture spatial heterogeneity.

#### 2. Locked Test Evaluation (2014–2015)
- **Selected Ridge Trend Model:** Test MAE = **0.4957 t/ha**, RMSE = **0.6630 t/ha**, R² = **0.6889** (Inference Latency: 0.0065s).
- **Median Baseline Reference:** Test MAE = **1.0195 t/ha**, RMSE = **1.2794 t/ha**, R² = **-0.1584**.
- **Year-by-Year Robustness:**
  - 2014 (n=557): MAE = 0.5247 t/ha, RMSE = 0.7024 t/ha, R² = 0.6443.
  - 2015 (n=557): MAE = 0.4667 t/ha, RMSE = 0.6212 t/ha, R² = 0.7267.

#### 3. Feature Ablation (Spatial Granularity)
- Removing the district spatial identifier degrades Ridge validation MAE from 0.4770 t/ha to 0.6668 t/ha (**+39.79% error increase**), proving that localized district administrative identity is indispensable for capturing agro-climatic baselines.

#### 4. Guided Classification Extension (22-Class Crop Recommendation)
- **Random Forest Classifier:** Test Accuracy = **99.32%**, Macro F1 = **0.9931**.
- **Logistic Regression:** Validation Accuracy = **97.27%**, Macro F1 = **0.9726**.
- **Majority Baseline:** Test Accuracy = **4.55%**, Macro F1 = **0.0040**.

---

### Academic Integrity and Reproducibility Statement
All code, models, and analytical outputs adhere strictly to institutional academic integrity guidelines. Dataset D5 is distributed under CC BY 4.0 by Souryabrata Mohapatra (Mendeley Data). Dataset D3 is attributed to Atharva Ingle. The pipeline uses a fixed global pseudorandom seed (`SEED = 42`) and guarantees deterministic reproducibility across all partitions, models, and serializations.
