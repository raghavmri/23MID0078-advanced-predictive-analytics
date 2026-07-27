# DA 2 Executive README

## Project Overview
This notebook implements a binary classification workflow for the Breast Cancer Wisconsin dataset using a decision tree model with pruning and threshold tuning. The goal is to distinguish malignant cases from benign cases and to save all generated figures and the final trained artifact to the project output directory.

## Notebook Specification
- Notebook: `DA 2.ipynb`
- Language: Python
- Random seed: `42`
- Output directory: `C:\Users\mribl\Documents\GitHub\College\Sem 7\Advance Predictive Analysis\Lab\DA 2\output`
- Saved model artifact: `lab02_decision_tree_artifact.joblib`

## Data Specification
- Dataset source: `sklearn.datasets.load_breast_cancer(as_frame=True)`
- Feature matrix: `X = raw.data.copy()`
- Target definition: malignant class is encoded as `1`, benign class as `0`
- Target name: `malignant`

## Processing Pipeline
1. Load the dataset and inspect class balance.
2. Check data quality for missing values and duplicates.
3. Split the data into train and test sets using stratification.
4. Build a pipeline with median imputation and a decision tree classifier.
5. Compute cost-complexity pruning candidates and evaluate them with cross-validation.
6. Run grid search over pruning strength and class weighting.
7. Select a decision threshold using out-of-fold probabilities and a minimum sensitivity target of `0.90`.
8. Fit the final model and evaluate it on the test set.
9. Generate interpretation plots for the final tree and feature importance.
10. Save the trained model, threshold, and metadata as a joblib artifact.

## Model Specification
- Estimator: `DecisionTreeClassifier`
- Preprocessing: `SimpleImputer(strategy="median")`
- Cross-validation: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Search method: `GridSearchCV`
- Scoring metric for tuning: `roc_auc`
- Threshold selection: first threshold meeting sensitivity >= `0.90`, with the highest specificity among candidates

## Evaluation Outputs
The notebook produces the following figures in the output directory:
- `figure1_class_distribution.png`
- `figure2_data_quality_audit.png`
- `figure3_pruning_path.png`
- `figure4_test_evaluation.png`
- `figure5_decision_tree.png`
- `figure5_feature_importance.png`
- `figure6_threshold_sensitivity_stability.png`

The notebook also prints the following summary values during execution:
- Dataset shape and class counts
- Missing value count and duplicate count
- Train/test split shapes
- Best pruning alpha
- Selected threshold
- Test accuracy
- Test ROC-AUC
- Tree depth and leaf count

## Final Saved Artifact
The saved `joblib` artifact contains:
- `model`: the fitted final pipeline
- `threshold`: the selected classification threshold
- `positive_class`: `malignant`
- `feature_names`: list of feature column names
- `random_state`: `42`

## Reproducibility Notes
- Run the notebook from the repository root or any location, because all outputs use the absolute `output` path.
- The notebook creates the output folder automatically if it does not already exist.
- The executed cells produce the same artifact layout as long as the dataset and library versions remain compatible.

## Repository Structure
- `DA 2.ipynb`: main analysis notebook
- `output/`: generated figures and model artifact
