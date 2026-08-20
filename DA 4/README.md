# Digital Assignment 4 - Advanced Predictive Analytics

Raghav Mrituanjaya KS (23MID0078)

## Title
Probabilistic Customer Segmentation and Segment Prediction Using Demographic, Psychographic, and Behavioral Data with Naive Bayes Classifiers

## Project Contents
- DA4_Probabilistic_Customer_Segmentation.ipynb
- Output/ (all generated figures and analysis exports)

## Aim
Build a complete supervised customer-segmentation pipeline that predicts predefined customer segments using demographic, psychographic, and behavioral data, then compare Naive Bayes models and select the best model based on Macro F1.

## Learning Objectives
1. Perform dataset quality checks before modeling.
2. Conduct focused EDA for segment understanding.
3. Apply leakage-safe preprocessing.
4. Compare Dummy, GaussianNB, BernoulliNB, and CategoricalNB fairly.
5. Evaluate with class-sensitive metrics and diagnostics.
6. Analyze prediction confidence and representative errors.

## What This Notebook Does
- Downloads customer-segmentation data from Kaggle
- Performs dataset quality checks and EDA
- Trains and compares DummyClassifier, GaussianNB, BernoulliNB, and CategoricalNB
- Selects the best model using 5-fold Macro F1
- Evaluates on test data with confusion matrices and class-wise metrics
- Performs confidence and top-error analysis
- Saves all required result images in Output/

## Core Working Pipeline
1. Data acquisition:
   - Downloads dataset ZIP from Kaggle using API.
   - Extracts files and loads CSV into a pandas DataFrame.
2. Dataset quality analysis:
   - Infers/identifies target segment column.
   - Detects possible identifier columns and excludes them from modeling.
   - Checks duplicates, missing values, and class distribution.
3. Exploratory data analysis:
   - Numerical distribution plot.
   - Segment-wise comparison plot.
   - Categorical relationship/frequency plot (when applicable).
4. Train-test split:
   - Stratified split to preserve segment proportions.
   - Test set remains locked until final model selection.
5. Model-specific preprocessing:
   - GaussianNB: median imputation + scaling for numeric, one-hot for categorical.
   - BernoulliNB: numeric binarization (KBins) + one-hot categorical.
   - CategoricalNB: numeric discretization + ordinal encoding categorical.
6. Cross-validation and comparison:
   - 5-fold Stratified CV.
   - Metrics: Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1.
   - Model selected primarily using Macro F1 mean and consistency.
7. Feature-group analysis:
   - Demographic only.
   - Psychographic only.
   - Behavioral only.
   - All features combined.
8. Final test evaluation:
   - Retrains selected model on full training data.
   - Evaluates locked test data.
   - Prints confusion matrices (count and normalized) and shows heatmaps.
9. Confidence and error diagnostics:
   - Uses max class probability as confidence.
   - Plots confidence distribution.
   - Exports top representative errors.
10. New customer prediction:
   - Applies same trained pipeline.
   - Returns predicted segment, probabilities, confidence, and review flag.

## Model Selection Logic
1. Macro F1 (primary criterion).
2. Fold-to-fold stability (standard deviation of Macro F1).
3. Class-wise behavior from confusion matrix and classification report.
4. Practical suitability of model assumptions for the feature representation.

## Responsible Analytics Notes
1. Use predictions for decision support, not as autonomous decision making.
2. Avoid discriminatory usage and high-impact decisions without human oversight.
3. Do not include direct identifiers as predictive features.
4. Treat low-confidence predictions as manual-review candidates.
5. Re-validate periodically for data drift and class imbalance changes.

## Setup
1. Install Python packages if required:
   - pip install kaggle pandas numpy scikit-learn matplotlib seaborn
2. Create Kaggle API token from:
   - https://www.kaggle.com/settings
3. Place kaggle.json at:
   - C:/Users/<your-username>/.kaggle/kaggle.json

## Run Instructions
1. Open DA4_Probabilistic_Customer_Segmentation.ipynb
2. Update KAGGLE_DATASET slug in the Kaggle cell if needed
3. Run all cells from top to bottom
4. Check Output/ for all generated figures

## Output Artifacts
1. 01_customer_segment_distribution.png
2. 02_missing_values.png (generated when missingness exists)
3. 03_numerical_distribution.png
4. 04_segment_comparison.png
5. 04b_categorical_relationship.png (if categorical columns are available)
6. 05_model_macro_f1_comparison.png
7. 06_feature_group_macro_f1_comparison.png
8. 07_confusion_matrix_count.png
9. 07b_confusion_matrix_row_normalized.png
10. 07c_per_class_metrics.png
11. 08_confidence_distribution.png
12. top_5_errors.csv

## Output Folder
All required output images are saved to:
C:/Users/mribl/Documents/GitHub/College/Sem 7/Advance Predictive Analysis/Lab/DA 4/Output

## Expected Figures
1. Customer segment distribution
2. Missing values chart (if applicable)
3. Numerical distribution plot
4. Segment comparison plot
5. Model Macro F1 comparison
6. Feature-group Macro F1 comparison
7. Confusion matrix (count and normalized)
8. Confidence distribution

## Reproducibility
1. Random state is fixed in train-test split and cross-validation.
2. All preprocessing is embedded inside model pipelines.
3. Transformers are fitted only on training folds/split to prevent leakage.
4. Test set is evaluated only after model selection.

## Quick Troubleshooting
- Kaggle download fails:
  - Verify kaggle.json path and account permissions
  - Recheck KAGGLE_DATASET slug format: owner/dataset-name
- Empty Output folder:
  - Ensure all cells ran without errors
  - Confirm the notebook has write access to the Output path
- CategoricalNB errors due to unexpected values:
   - Re-run all cells so preprocessing and model pipelines are initialized consistently
   - Ensure extracted dataset file is the expected customer-segmentation CSV
