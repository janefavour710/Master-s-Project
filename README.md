# SpamShield

Supervised ML pipeline that classifies SMS messages as **spam** or **ham**, with a Streamlit prediction interface.

**COMP11117 MSc Information Technology · University of the West of Scotland**  
Student: Favour Nnenna Ogbonnaya-John · Supervisor: Dr. N.O. Salau

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (downloads data, trains all models, saves reports)
python main.py

# 3. Launch the web demo
streamlit run app.py
```

---

## Project Structure

```
SpamShield/
├── data/                    # Raw dataset (auto-downloaded)
│   └── spam.csv
├── models/                  # Trained classifiers + TF-IDF vectoriser
│   ├── logistic_regression.joblib
│   ├── svm.joblib
│   ├── xgboost.joblib
│   ├── tfidf.joblib
│   └── best_model.txt       # Name of best F1 model
├── notebooks/
│   └── spamshield_analysis.ipynb  # EDA + full pipeline walkthrough
├── reports/                 # Evaluation outputs
│   ├── metrics.csv
│   ├── model_comparison.png
│   ├── confusion_matrices.png
│   └── roc_curves.png
├── src/
│   ├── preprocess.py        # Text cleaning & lemmatisation
│   ├── features.py          # TF-IDF vectorisation + SMOTE
│   ├── train.py             # LR / SVM / XGBoost training
│   ├── evaluate.py          # 6-metric evaluation + plots
│   └── predict.py           # Single-message inference
├── app.py                   # Streamlit UI
├── main.py                  # End-to-end pipeline runner
├── download_data.py         # UCI dataset downloader
└── requirements.txt
```

---

## Pipeline

| Step | Description |
|------|-------------|
| 1 | Load UCI SMS Spam Collection (5 574 messages, 87:13 ham:spam) |
| 2 | Lowercase · remove URLs / emails / phone numbers · lemmatise · remove stopwords |
| 3 | Stratified 80/20 train-test split (`random_state=42`) |
| 4 | TF-IDF (unigrams + bigrams, 5 000 features) fitted on train only |
| 5 | SMOTE applied to training features only — no leakage into test set |
| 6 | Train Logistic Regression, LinearSVC, XGBoost |
| 7 | Evaluate on held-out test set: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC |
| 8 | Save models, metrics CSV, and plots |

---

## Models

| Model | Notes |
|-------|-------|
| Logistic Regression | Fast, interpretable linear baseline |
| SVM (LinearSVC) | Strong margin-based classifier for text; decision_function used for AUC |
| XGBoost | Gradient-boosted ensemble |

Primary evaluation metrics: **F1-score** and **ROC-AUC** (robust to class imbalance).

---

## Dataset

UCI SMS Spam Collection — 5 574 English SMS messages labelled `spam` / `ham`.  
Downloaded automatically from the UCI ML Repository on first run.

---

## Tech Stack

Python 3.10+ · scikit-learn · XGBoost · imbalanced-learn · NLTK · pandas · NumPy · Matplotlib · Seaborn · Streamlit · joblib
