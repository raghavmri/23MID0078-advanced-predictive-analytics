# Advanced Predictive Analytics – Digital Assignment 3 (DA 3)

**Author:** Raghav Mrituanjaya (23MID0078)

An end-to-end NLP pipeline for email intent classification and spam detection using scikit-learn baseline models, TF-IDF representations, and deep learning (BiLSTM / MLP).

---

## 📌 Overview

This project explores text classification workflows across three benchmark datasets:
1. **Business Email Intent**: Multi-class classification (request, meeting, complaint, information, urgent action, spam).
2. **Enron Spam**: Binary spam/ham classification.
3. **SpamAssassin**: Binary spam/ham classification.

Key focus areas include model benchmarking, cross-validation evaluation, text length analysis, and domain transferability (evaluating models trained on one spam dataset directly on another).

---

## 📁 Project Structure

```text
DA 3/
├── DA 3.ipynb           # Main interactive notebook with full code, plots, and commentary
├── create_da3.py        # Python script that programmatically builds and updates the notebook
├── requirements.txt     # Python package requirements
├── data/                # Dataset CSV files (generated/loaded during execution)
└── outputs/             # Exported performance charts, metrics CSVs, and split manifests
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Running the Code
- **Interactive Notebook**: Open [`DA 3.ipynb`](file:///c:/Users/mribl/Documents/GitHub/College/Sem%207/Advance%20Predictive%20Analysis/Lab/DA%203/DA%203.ipynb) in VS Code or Jupyter Notebook and run all cells.
- **Regenerate Notebook**: If you want to rebuild the notebook programmatically from `create_da3.py`, run:
  ```bash
  python create_da3.py
  ```

---

## 🔬 Models & Analysis Included

- **Baseline & Linear Classifiers**: Dummy Classifier, Multinomial & Complement Naive Bayes, Logistic Regression, Linear SVC.
- **Deep Learning**: BiLSTM (with Embedding layer and Dropout) using Keras / TensorFlow (with MLP fallback if TensorFlow isn't present).
- **Domain Generalization**: Cross-dataset evaluation (training on Enron, evaluating on SpamAssassin and vice versa).
- **Artifacts Saved**: Cross-validation macro F1 summaries, confusion matrices, training loss/accuracy curves, and class distribution plots in `outputs/`.
