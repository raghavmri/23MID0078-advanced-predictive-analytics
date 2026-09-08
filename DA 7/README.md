### Project Overview
This repository contains the complete implementation, analytical evaluation, diagnostic visualizations, and academic reporting for Experiment 07. The project formulates personalized product recommendation as a supervised point-wise candidate scoring and ranking problem on retail transaction data.

Using the verified UCI Machine Learning Repository Online Retail transaction corpus (541,909 raw records), the system enforces strict chronological partitioning across training, validation, and locked prospective test horizons to guarantee zero lookahead data leakage. A bounded candidate generation universe (top 1,000 frequent products) is evaluated with an explicit candidate recall audit. Sixteen leakage-safe features spanning customer RFM behavior, product demand/pricing characteristics, and customer-item pair affinities are extracted and utilized to train a tuned Random Forest ranking classifier. The system is comprehensively benchmarked against a non-personalized Popularity Baseline and an Item-Item Collaborative Filtering model using listwise ranking metrics (Precision@K, Recall@K, HitRate@K, NDCG@K, MAP@K), 95% bootstrap confidence intervals, systematic feature ablation, and a five-case qualitative error audit.

---

### Key Empirical Results

| Metric (@ K=10) | Popularity Baseline | Item-Item CF Benchmark | Random Forest Ranker | Relative Gain vs Popularity |
| :--- | :--- | :--- | :--- | :--- |
| **Precision@10** | 0.0723 | 0.2080 | **0.2756** | **+281.2%** |
| **Recall@10** | 0.0315 | 0.1335 | **0.1724** | **+447.3%** |
| **HitRate@10** | 44.16% | 70.86% | **83.31%** | **+88.7%** |
| **NDCG@10** | 0.0822 | 0.2450 | **0.3252** | **+295.6%** |
| **Pair PR-AUC** | 0.0389 | N/A | **0.2232** | **+473.8%** |
| **Pair ROC-AUC** | 0.6331 | N/A | **0.8524** | **+34.6%** |

*Note: 95% user-level bootstrap confidence intervals for Random Forest Recall@10 are [0.1604, 0.1847] and for NDCG@10 are [0.3114, 0.3395], demonstrating non-overlapping statistical separation from both baselines at p < 0.001.*

---

### Repository File Structure

```text
├── DA6.ipynb                           # Main executed Jupyter notebook (zero code comments)
├── DA6.ipymb                           # Duplicate notebook copy matching requested naming
├── README.md                           # Comprehensive reproduction guide and student attribution
├── report.md                           # Formal academic report following required subheadings
├── data/
│   └── online_retail.csv               # Cleaned cached UCI Online Retail transaction dataset
├── models/
│   └── random_forest.joblib            # Serialized tuned Random Forest ranking model
├── artifacts/
│   ├── candidate_policy.json           # Formal candidate catalog boundaries and parameters
│   ├── feature_schema.json             # Feature dictionary across customer, item, and pair levels
│   └── split_manifest.json             # Chronological cutoff timestamps and split row counts
├── output/
│   ├── fig1_transaction_volume.png     # Weekly transaction order volume and chronological cutoffs
│   ├── fig2_top15_items.png            # Top 15 products by unique invoice frequency
│   ├── fig3_cust_purchase_frequency.png# Customer purchase frequency distribution
│   ├── fig4_rfm_distributions.png      # Customer Recency, Frequency, and Monetary (RFM) distributions
│   ├── fig5_class_balance.png          # Class balance in training and validation folds
│   ├── fig6_feature_importance.png     # Gini feature importance chart for tuned Random Forest
│   ├── fig7_precision_recall_k.png     # Precision@K vs Recall@K trade-off curves (K = 5, 10, 20)
│   ├── fig8_baseline_vs_rf_ranking.png # Comparative bar chart of ranking metrics across models
│   ├── fig9_score_distribution.png     # Score separation density for positive vs negative pairs
│   ├── fig10_catalog_coverage.png      # Lorenz curve of catalog coverage across test users
│   └── analysis/
│       ├── advanced_uncertainty.csv    # Bootstrap means, standard deviations, and 95% CIs
│       ├── candidate_recall.csv        # Validation and test candidate recall audit
│       ├── error_analysis.csv          # Qualitative audit of 5 distinct recommendation cases
│       ├── feature_ablation.csv        # Systematic feature subset ablation performance
│       ├── ranking_metrics.csv         # Full listwise metrics across K in {5, 10, 20}
│       ├── recommendations.csv         # Exported Top-5 recommendations for sample customers
│       └── validation_tuning.csv       # Hyperparameter grid evaluation on validation fold
```

---

### Environment Specifications and Dependencies

The implementation requires Python 3.10+ (tested on Python 3.14.0 Windows 64-bit). The necessary Python packages include:

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn joblib nbformat openpyxl
```

Recommended versions:
- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- scipy >= 1.10.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- joblib >= 1.2.0
- nbformat >= 5.9.0

---

### How to Run and Reproduce

#### Option 1: Run the Complete Jupyter Notebook
1. Open the repository directory in your preferred environment (VS Code, JupyterLab, or Jupyter Notebook).
2. Open `DA6.ipynb` (or `DA6.ipymb`).
3. Execute all cells sequentially from top to bottom (`Kernel` -> `Restart & Run All`).
4. The notebook will automatically:
   - Verify/download the dataset via public repository archives into `data/online_retail.csv`.
   - Clean cancellations and missing identifiers.
   - Enforce chronological partitions (`artifacts/split_manifest.json`).
   - Define candidate policies and run candidate recall audits (`output/analysis/candidate_recall.csv`).
   - Extract 16 time-valid features (`artifacts/feature_schema.json`).
   - Tune and train Random Forest models (`models/random_forest.joblib`).
   - Fit Popularity Baseline and Item-Item Collaborative Filtering benchmarks.
   - Evaluate ranking metrics and generate bootstrap 95% confidence intervals (`output/analysis/ranking_metrics.csv`, `output/analysis/advanced_uncertainty.csv`).
   - Run feature ablation experiments (`output/analysis/feature_ablation.csv`).
   - Conduct the qualitative five-case error audit (`output/analysis/error_analysis.csv`).
   - Generate all 10 diagnostic figures in high resolution to `output/*.png`.
   - Validate Appendix C acceptance assertions.

#### Option 2: Programmatic Execution via Command Line
To execute the automated pipeline from the terminal:
```bash
python execute_and_save_notebook.py
```
This script executes every cell, populates all outputs within `DA6.ipynb` and `DA6.ipymb`, and confirms all acceptance test assertions.

---

### Acceptance Tests and Verification (Appendix C)
The notebook includes automated programmatic assertions verifying:
1. `train_hist.InvoiceDate.max() < t1` (Strict temporal isolation of training history).
2. `val_future.InvoiceDate.min() >= t1` and `test_future.InvoiceDate.min() >= t2` (Strict future boundary enforcement).
3. `test_future` indices are disjoint from `train_hist`.
4. All feature columns are present and contain zero NaN values.
5. Target labels are strictly binary (`{0, 1}`).
6. The candidate catalog size satisfies `500 <= len(candidate_items) <= 2000` (Bounded at exactly 1,000 items).
7. Candidate recall lies within the valid unit interval `[0.0, 1.0]`.
8. Serialized model predictions from `models/random_forest.joblib` match in-memory model predictions identically.

All acceptance assertions pass with 100% compliance.
