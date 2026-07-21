# MD13003 - Advanced Predictive Analytics Laboratory
## End-to-End Predictive Analytics Pipeline Project

**Registration Number:** `23MID0078`  
**Course:** MD13003 - Advanced Predictive Analytics  

---

## Project Structure

```text
DA 1/
├── datasets/
│   └── kaggle_housing_predictive_dataset.csv
├── output/
│   ├── tables/
│   │   ├── 23MID0078_Lab01_Model_Comparison.csv
│   │   ├── 23MID0078_Lab01_CV_Results.csv
│   │   └── 23MID0078_Lab01_Feature_Importances.csv
│   ├── figures/
│   │   ├── 23MID0078_Target_Distribution.png
│   │   ├── 23MID0078_Correlation_Heatmap.png
│   │   ├── 23MID0078_Model_Comparison_Chart.png
│   │   ├── 23MID0078_Feature_Importances_Chart.png
│   │   └── 23MID0078_Residuals_Plot.png
│   ├── models/
│   │   └── 23MID0078_Best_Model_Pipeline.joblib
│   └── metadata/
│       └── 23MID0078_Run_Metadata.json
├── 23MID0078_Lab01_Predictive_Analysis.ipynb
├── requirements.txt
└── README.md
```

---

## Quick Start & Execution Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Notebook
Open and run all cells in `23MID0078_Lab01_Predictive_Analysis.ipynb`:
```bash
jupyter notebook 23MID0078_Lab01_Predictive_Analysis.ipynb
```

Running all cells in the Jupyter notebook automatically executes the full predictive modeling pipeline and exports all outputs, figures, tables, trained models, and metadata JSON into the `output/` subdirectories.

---

## Features & Methodology

1. **Dataset**: Kaggle Housing Predictive Dataset stored in `datasets/kaggle_housing_predictive_dataset.csv`.
2. **Leak-Proof Preprocessing**: Imputation and scaling encapsulated using `ColumnTransformer` and `Pipeline` objects.
3. **Multi-Model Comparison**: `RandomForestRegressor`, `GradientBoostingRegressor`, and `RidgeRegressor`.
4. **Validation**: 5-Fold Cross-Validation (`cross_validate`).
5. **Outputs & Serialization**:
   - `output/tables/`: CSV performance comparison tables.
   - `output/figures/`: High-resolution figures (distribution, correlation heatmap, model comparison, feature importances, residuals).
   - `output/models/`: Serialized model pipeline artifact (`joblib`).
   - `output/metadata/`: Execution metadata JSON.
