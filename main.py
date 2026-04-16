import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocess import preprocess_series
from src.features import build_features
from src.train import train_all
from src.evaluate import evaluate_all, save_plots, save_metrics_csv, best_model_name

DATA_PATH = "data/spam.csv"


def main():
    # ── 1. Data ────────────────────────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print("Dataset not found — downloading from UCI...")
        from download_data import download
        download()

    print("Loading data...")
    df = pd.read_csv(DATA_PATH, encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "message"]
    df["label"] = (df["label"] == "spam").astype(int)
    spam_count = df["label"].sum()
    ham_count = (df["label"] == 0).sum()
    print(f"  Loaded {len(df):,} messages — {ham_count:,} ham, {spam_count:,} spam")

    # ── 2. Clean ───────────────────────────────────────────────────────────────
    print("Preprocessing messages (lowercasing, stripping URLs, lemmatising)...")
    df["clean"] = preprocess_series(df["message"])

    # 80% for training, 20% for testing; stratify keeps the ham/spam ratio equal
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    print(f"  Train: {len(X_train):,} messages  |  Test: {len(X_test):,} messages")

    # ── 3. Vectorise ───────────────────────────────────────────────────────────
    print("Vectorising (TF-IDF) and resampling (SMOTE)...")
    X_train_res, y_train_res, X_test_tfidf, _ = build_features(X_train, X_test, y_train)

    # ── 4. Train ───────────────────────────────────────────────────────────────
    print("Training classifiers...")
    trained = train_all(X_train_res, y_train_res)

    # ── 5. Evaluate ────────────────────────────────────────────────────────────
    print("\nEvaluation results (held-out test set):")
    results = evaluate_all(trained, X_test_tfidf, y_test)
    save_plots(results, y_test)
    save_metrics_csv(results)
    print("  Charts and metrics saved to reports/")

    # ── 6. Record best ─────────────────────────────────────────────────────────
    best = best_model_name(results)
    os.makedirs("models", exist_ok=True)
    with open(os.path.join("models", "best_model.txt"), "w") as f:
        f.write(best)

    print(f"\nBest model by F1: {best}")
    print("Run  streamlit run app.py  to open the web interface.")


if __name__ == "__main__":
    main()
