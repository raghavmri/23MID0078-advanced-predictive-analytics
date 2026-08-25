## Laboratory Experiment 05: Product and Brand Sentiment Prediction from Tweet Data

## 1. Project Overview & Analytical Scope

This repository provides a complete, leak-proof, and fully reproducible natural language processing (NLP) workflow for **Experiment 05: Product and Brand Sentiment Prediction from Tweet Data**. In this study, we model tweet sentiment as a supervised 3-class classification problem (Negative, Neutral, Positive) targeting consumer feedback directed toward major commercial airlines.

The primary dataset used throughout the core experiments is the **Twitter US Airline Sentiment** corpus ($N=14,640$ annotated tweets across six US air carriers), accessed programmatically via the Hugging Face `datasets` public API (`osanseviero/twitter-airline-sentiment`). To assess cross-domain generalization, we also benchmark the pipeline against the standardized **TweetEval Sentiment** dataset ($N=45,615$ training tweets) using `cardiffnlp/tweet_eval`.

Our methodology emphasizes transparent feature representations and rigorous model selection. We evaluate baseline strategies (`DummyClassifier` and rule-based `VADER`) alongside classical term frequency-inverse document frequency (TF-IDF) feature pipelines paired with `MultinomialNB`, `LinearSVC`, and `LogisticRegression`. All feature extraction statistics are fitted strictly inside 5-Fold Stratified Cross-Validation folds to ensure zero data leakage before final single-pass evaluation on an untouched test partition.

---

## 2. Directory Layout & Artifact Organization

The repository is structured to separate source code notebooks, final report documentation, trained models, and generated experimental outputs:

```
.
├── DA5.ipynb                              # Primary executable Jupyter Notebook (code cells with 1-line section headers)
├── 23MID0078_Lab05_Report.md              # Comprehensive academic report with embedded figure placeholders
├── report.md                               # Markdown report copy
├── README.md                               # Project guide and environment documentation
├── models/
│   └── selected_pipeline.joblib           # Canonical trained scikit-learn model pipeline artifact
└── output/
    ├── figures/                           # High-resolution output plots generated during execution
    │   ├── class_distribution.png
    │   ├── tweet_length_distribution.png
    │   ├── top_ngrams_by_class.png
    │   ├── cv_model_comparison.png
    │   ├── confusion_matrices.png
    │   ├── per_class_metrics.png
    │   ├── entity_sentiment_distribution.png
    │   ├── entity_error_rates.png
    │   ├── cleaning_ablation_comparison.png
    │   └── tweeteval_benchmark_comparison.png
    └── analysis/                          # Quantitative results, predictions, and error tables (CSV format)
        ├── cv_results.csv
        ├── 23MID0078_Lab05_CV_Results.csv
        ├── test_predictions.csv
        ├── 23MID0078_Lab05_Test_Predictions.csv
        ├── error_analysis.csv
        ├── 23MID0078_Lab05_Error_Analysis.csv
        └── entity_sentiment_distribution.csv
```

---

## 3. Environment & Installation Requirements

### Software Stack
- **Python Version:** Python 3.10 or higher (Tested on Python 3.12 / 3.14)
- **Execution Platform:** Local Jupyter Notebook, JupyterLab, or Google Colab

### Package Dependencies
To replicate the environment, install the required dependencies using `pip`:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn vaderSentiment datasets joblib
```

---

## 4. Execution & Replication Guide

### Step 1: Running the Main Notebook
Execute the primary Jupyter Notebook `DA5.ipynb`:

```bash
jupyter notebook DA5.ipynb
```

From the Jupyter interface, select **Cell -> Run All**. The notebook will execute the complete pipeline top-to-bottom:
1. Programmatically fetch dataset partitions via Hugging Face public APIs.
2. Apply minimal tweet normalization (`<URL>` and `<USER>` substitution).
3. Compute 5-Fold Stratified Cross-Validation metrics for all candidate models.
4. Export evaluation metrics, locked test set predictions, and error cases to `output/analysis/`.
5. Render and save all 10 visual plots into `output/figures/`.
6. Serialize the optimal trained classifier pipeline to `models/selected_pipeline.joblib`.

---

## 5. Summary of Experimental Results

### 5-Fold Cross-Validation & Test Set Evaluation

| Model Architecture | Representation | CV Macro F1 (Mean ± SD) | CV Weighted F1 | Test Macro F1 | Test Accuracy | Mean Fit Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **DummyClassifier** | Majority Class | 0.2569 ± 0.0000 | 0.4704 | 0.2569 | 0.6270 | 0.08 s |
| **VADER Lexicon** | Sentiment Rules | 0.5126 ± 0.0000 | 0.5740 | 0.5126 | 0.5495 | 0.05 s |
| **MultinomialNB** | TF-IDF (1,2-gram) | 0.5858 ± 0.0103 | 0.7363 | 0.5858 | 0.7363 | 0.41 s |
| **LinearSVC** | TF-IDF (Balanced) | 0.7395 ± 0.0076 | 0.8006 | 0.7395 | 0.8006 | 0.53 s |
| **Logistic Regression** | **TF-IDF (Balanced)** | **0.7444 ± 0.0084** | **0.7928** | **0.7442** | **0.7917** | **0.82 s** |

*Key Takeaway:* **Multinomial Logistic Regression with sublinear TF-IDF word/bigram features** achieved the highest cross-validation stability (CV Macro F1 of **0.7444**) and locked test set Macro F1 (**0.7442**), while providing well-calibrated class probability estimates.

---

## 6. Data Governance & Ethical Considerations

- **Leakage Prevention:** Annotation confidence metadata (`airline_sentiment_confidence`, `negativereason_confidence`) and post-label categories (`negativereason`) were explicitly stripped prior to training to prevent target leakage.
- **Anonymization:** Tweet IDs, personal user handles (`<USER>`), URLs (`<URL>`), and geographic coordinates were excluded to protect user privacy.
- **Scope & Limitations:** Social media sentiment metrics reflect vocal public chatter during specific operational periods (e.g., winter weather flight disruptions) and should not be interpreted as unbiased customer satisfaction or market share measurements.
