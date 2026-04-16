import os
import joblib
from functools import lru_cache
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

CLASSIFIERS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "svm": LinearSVC(max_iter=2000, random_state=42),
    "xgboost": XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=42),
}


def train_all(X_train, y_train):
    """Fit every classifier and save each one to the models/ directory."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    trained = {}
    for name, clf in CLASSIFIERS.items():
        print(f"  [{name}] fitting...")
        clf.fit(X_train, y_train)
        joblib.dump(clf, os.path.join(MODELS_DIR, f"{name}.joblib"))
        print(f"  [{name}] saved.")
        trained[name] = clf
    return trained


@lru_cache(maxsize=8)
def load_model(name):
    return joblib.load(os.path.join(MODELS_DIR, f"{name}.joblib"))
