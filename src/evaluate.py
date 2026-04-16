import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve,
)

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
METRICS_CSV = os.path.join(REPORTS_DIR, "metrics.csv")


def evaluate_model(name, clf, X_test, y_test):
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X_test)[:, 1]
    else:
        y_score = clf.decision_function(X_test)

    y_pred = clf.predict(X_test)

    return {
        "name": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_score),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_score": y_score,
    }


def evaluate_all(trained_models, X_test, y_test):
    results = []
    for name, clf in trained_models.items():
        r = evaluate_model(name, clf, X_test, y_test)
        results.append(r)
        print(f"  {name:25s}  Acc={r['accuracy']:.4f}  F1={r['f1']:.4f}  ROC-AUC={r['roc_auc']:.4f}")
    return results


def save_metrics_csv(results):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    rows = [{
        "model": r["name"],
        "accuracy": round(r["accuracy"], 4),
        "precision": round(r["precision"], 4),
        "recall": round(r["recall"], 4),
        "f1": round(r["f1"], 4),
        "roc_auc": round(r["roc_auc"], 4),
    } for r in results]
    pd.DataFrame(rows).to_csv(METRICS_CSV, index=False)


def save_plots(results, y_test):
    os.makedirs(REPORTS_DIR, exist_ok=True)

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        sns.heatmap(r["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
        ax.set_title(r["name"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrices.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["y_score"])
        plt.plot(fpr, tpr, label=f"{r['name']} (AUC={r['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "roc_curves.png"), dpi=150)
    plt.close()

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = range(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, r in enumerate(results):
        values = [r[m] for m in metrics]
        offset = (i - len(results) / 2 + 0.5) * width
        bars = ax.bar([xi + offset for xi in x], values, width, label=r["name"])
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels([m.replace("_", " ").upper() for m in metrics])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "model_comparison.png"), dpi=150)
    plt.close()


def best_model_name(results):
    return max(results, key=lambda r: r["f1"])["name"]
