import json

REGISTRATION_NO = "23MID0078"


def make_code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }


def make_md_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines,
    }


cells = []

cells.append(make_md_cell([
    "# MD13003 - Advanced Predictive Analytics Laboratory 01\n",
    "## End-to-End Predictive Analytics Pipeline with Kaggle Housing Dataset\n",
    "\n",
    "**Student Registration Number:** `23MID0078`  \n",
    "**Course Code:** `MD13003`  \n",
    "**Global Seed:** `SEED = 42`  \n",
    "\n",
    "---\n",
    "### Overview & Objectives\n",
    "1. **Data Ingestion & EDA** - target distribution & correlation heatmap\n",
    "2. **Leak-Proof Preprocessing** - `ColumnTransformer` & `Pipeline`\n",
    "3. **Multi-Model Evaluation** - RandomForest, GradientBoosting, Ridge\n",
    "4. **5-Fold Cross-Validation** analysis\n",
    "5. **Feature Importance & Residual Analysis**\n",
    "6. **Artifact Export** to `output/` directory",
]))

cells.append(make_md_cell(["## Section 1: Environment Setup & Output Directory Initialization"]))

cells.append(make_code_cell([
    "import os\n",
    "import json\n",
    "import datetime\n",
    "import platform\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split, cross_validate\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
    "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\n",
    "from sklearn.linear_model import Ridge\n",
    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
    "import joblib\n",
    "\n",
    "REGISTRATION_NO = '23MID0078'\n",
    "SEED = 42\n",
    "np.random.seed(SEED)\n",
    "\n",
    "for folder in ['output/tables', 'output/figures', 'output/models', 'output/metadata']:\n",
    "    os.makedirs(folder, exist_ok=True)\n",
    "\n",
    "print(f'[{REGISTRATION_NO}] Directories initialized. SEED = {SEED}')",
]))

cells.append(make_md_cell(["## Section 2: Data Ingestion & Exploratory Data Analysis (EDA)"]))

cells.append(make_code_cell([
    "dataset_path = 'datasets/kaggle_housing_predictive_dataset.csv'\n",
    "df = pd.read_csv(dataset_path)\n",
    "print('Dataset Shape:', df.shape)\n",
    "print(df.describe())\n",
    "\n",
    "plt.figure(figsize=(8, 4))\n",
    "sns.histplot(df['price_k'], kde=True, color='royalblue')\n",
    "plt.title(f'[{REGISTRATION_NO}] Target Variable Distribution')\n",
    "plt.xlabel('Price ($k)')\n",
    "plt.ylabel('Frequency')\n",
    "plt.tight_layout()\n",
    "plt.savefig('output/figures/23MID0078_Target_Distribution.png', dpi=300)\n",
    "plt.show()\n",
    "\n",
    "plt.figure(figsize=(7, 5))\n",
    "num_df = df.select_dtypes(include=[np.number])\n",
    "sns.heatmap(num_df.corr(), annot=True, cmap='Blues', fmt='.2f')\n",
    "plt.title(f'[{REGISTRATION_NO}] Feature Correlation Heatmap')\n",
    "plt.tight_layout()\n",
    "plt.savefig('output/figures/23MID0078_Correlation_Heatmap.png', dpi=300)\n",
    "plt.show()",
]))

cells.append(make_md_cell(["## Section 3: Data Preprocessing & ColumnTransformer Assembly"]))

cells.append(make_code_cell([
    "X = df.drop(columns=['price_k'])\n",
    "y = df['price_k']\n",
    "\n",
    "num_cols = ['square_feet', 'age_years', 'num_rooms', 'distance_km']\n",
    "cat_cols = ['location_code', 'property_type', 'energy_rating']\n",
    "\n",
    "num_pipeline = Pipeline([\n",
    "    ('imputer', SimpleImputer(strategy='median')),\n",
    "    ('scaler', StandardScaler())\n",
    "])\n",
    "\n",
    "cat_pipeline = Pipeline([\n",
    "    ('imputer', SimpleImputer(strategy='most_frequent')),\n",
    "    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))\n",
    "])\n",
    "\n",
    "preprocessor = ColumnTransformer(transformers=[\n",
    "    ('num', num_pipeline, num_cols),\n",
    "    ('cat', cat_pipeline, cat_cols)\n",
    "])\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)\n",
    "print(f'Train: {X_train.shape}, Test: {X_test.shape}')",
]))

cells.append(make_md_cell(["## Section 4: Multi-Model Training & Evaluation"]))

cells.append(make_code_cell([
    "models = {\n",
    "    'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=SEED),\n",
    "    'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=100, random_state=SEED),\n",
    "    'RidgeRegressor': Ridge(random_state=SEED)\n",
    "}\n",
    "\n",
    "results_list = []\n",
    "fitted_pipelines = {}\n",
    "\n",
    "for name, estimator in models.items():\n",
    "    pipe = Pipeline([('preprocessor', preprocessor), ('regressor', estimator)])\n",
    "    pipe.fit(X_train, y_train)\n",
    "    y_pred = pipe.predict(X_test)\n",
    "    mae = mean_absolute_error(y_test, y_pred)\n",
    "    mse = mean_squared_error(y_test, y_pred)\n",
    "    rmse = np.sqrt(mse)\n",
    "    r2 = r2_score(y_test, y_pred)\n",
    "    fitted_pipelines[name] = pipe\n",
    "    results_list.append({'Model': name, 'Registration_No': REGISTRATION_NO,\n",
    "                         'MAE': round(mae, 4), 'MSE': round(mse, 4),\n",
    "                         'RMSE': round(rmse, 4), 'R2_Score': round(r2, 4)})\n",
    "\n",
    "comparison_df = pd.DataFrame(results_list)\n",
    "comparison_df.to_csv('output/tables/23MID0078_Lab01_Model_Comparison.csv', index=False)\n",
    "print(comparison_df)\n",
    "\n",
    "plt.figure(figsize=(7, 4))\n",
    "sns.barplot(data=comparison_df, x='Model', y='R2_Score', palette='viridis')\n",
    "plt.title(f'[{REGISTRATION_NO}] Model Performance Comparison (R2 Score)')\n",
    "plt.ylabel('R2 Score')\n",
    "plt.xticks(rotation=15)\n",
    "plt.tight_layout()\n",
    "plt.savefig('output/figures/23MID0078_Model_Comparison_Chart.png', dpi=300)\n",
    "plt.show()",
]))

cells.append(make_md_cell(["## Section 5: 5-Fold Cross-Validation Analysis"]))

cells.append(make_code_cell([
    "best_model_name = comparison_df.sort_values(by='R2_Score', ascending=False).iloc[0]['Model']\n",
    "best_pipeline = fitted_pipelines[best_model_name]\n",
    "print(f'Best model: {best_model_name}')\n",
    "\n",
    "cv_results = cross_validate(\n",
    "    best_pipeline, X_train, y_train, cv=5,\n",
    "    scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2']\n",
    ")\n",
    "\n",
    "cv_df = pd.DataFrame({\n",
    "    'Fold': [f'Fold_{i+1}' for i in range(5)],\n",
    "    'Registration_No': REGISTRATION_NO,\n",
    "    'Fold_MAE': np.round(-cv_results['test_neg_mean_absolute_error'], 4),\n",
    "    'Fold_MSE': np.round(-cv_results['test_neg_mean_squared_error'], 4),\n",
    "    'Fold_RMSE': np.round(-cv_results['test_neg_root_mean_squared_error'], 4),\n",
    "    'Fold_R2': np.round(cv_results['test_r2'], 4)\n",
    "})\n",
    "cv_df.to_csv('output/tables/23MID0078_Lab01_CV_Results.csv', index=False)\n",
    "print(cv_df)",
]))

cells.append(make_md_cell(["## Section 6: Feature Importances & Residual Plotting"]))

cells.append(make_code_cell([
    "ohe_cols = list(\n",
    "    best_pipeline.named_steps['preprocessor']\n",
    "    .named_transformers_['cat']\n",
    "    .named_steps['encoder']\n",
    "    .get_feature_names_out(cat_cols)\n",
    ")\n",
    "all_features = num_cols + ohe_cols\n",
    "\n",
    "if hasattr(best_pipeline.named_steps['regressor'], 'feature_importances_'):\n",
    "    importances = best_pipeline.named_steps['regressor'].feature_importances_\n",
    "    feat_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})\n",
    "    feat_df = feat_df.sort_values(by='Importance', ascending=False)\n",
    "    feat_df.to_csv('output/tables/23MID0078_Lab01_Feature_Importances.csv', index=False)\n",
    "    plt.figure(figsize=(8, 4))\n",
    "    sns.barplot(data=feat_df, x='Importance', y='Feature', palette='mako')\n",
    "    plt.title(f'[{REGISTRATION_NO}] Feature Importances ({best_model_name})')\n",
    "    plt.tight_layout()\n",
    "    plt.savefig('output/figures/23MID0078_Feature_Importances_Chart.png', dpi=300)\n",
    "    plt.show()\n",
    "\n",
    "y_pred_best = best_pipeline.predict(X_test)\n",
    "residuals = y_test - y_pred_best\n",
    "plt.figure(figsize=(7, 4))\n",
    "sns.scatterplot(x=y_pred_best, y=residuals, color='darkred')\n",
    "plt.axhline(0, color='black', linestyle='--')\n",
    "plt.title(f'[{REGISTRATION_NO}] Residual Analysis Plot')\n",
    "plt.xlabel('Predicted Price ($k)')\n",
    "plt.ylabel('Residuals')\n",
    "plt.tight_layout()\n",
    "plt.savefig('output/figures/23MID0078_Residuals_Plot.png', dpi=300)\n",
    "plt.show()",
]))

cells.append(make_md_cell(["## Section 7: Model Serialization & Run Metadata Export"]))

cells.append(make_code_cell([
    "model_path = 'output/models/23MID0078_Best_Model_Pipeline.joblib'\n",
    "joblib.dump(best_pipeline, model_path)\n",
    "\n",
    "run_metadata = {\n",
    "    'registration_number': REGISTRATION_NO,\n",
    "    'course_code': 'MD13003 - Advanced Predictive Analytics',\n",
    "    'execution_timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),\n",
    "    'seed': SEED,\n",
    "    'environment': {\n",
    "        'python_version': platform.python_version(),\n",
    "        'platform': platform.platform(),\n",
    "    },\n",
    "    'dataset': {\n",
    "        'path': dataset_path,\n",
    "        'samples': len(df),\n",
    "        'features': X.shape[1],\n",
    "    },\n",
    "    'best_model': best_model_name,\n",
    "    'test_metrics': comparison_df[\n",
    "        comparison_df['Model'] == best_model_name\n",
    "    ].to_dict(orient='records')[0],\n",
    "}\n",
    "\n",
    "with open('output/metadata/23MID0078_Run_Metadata.json', 'w') as f:\n",
    "    json.dump(run_metadata, f, indent=4)\n",
    "\n",
    "print(f'Model saved to: {model_path}')\n",
    "print('Metadata saved to: output/metadata/23MID0078_Run_Metadata.json')\n",
    "print(f'[{REGISTRATION_NO}] All outputs generated successfully.')",
]))


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output_path = "23MID0078_Lab01_Predictive_Analysis.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"Notebook generated successfully: {output_path}")
