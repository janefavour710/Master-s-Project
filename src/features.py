import os
import joblib
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE

VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "tfidf.joblib")


def build_features(X_train, X_test, y_train, max_features=5000):
    """Convert cleaned text into numeric feature matrices ready for training.

    Steps
    -----
    1. TF-IDF turns each message into a 5,000-number vector.
       Unigrams and bigrams are both used (e.g. "win" and "win prize").
       sublinear_tf=True dampens very frequent terms so they don't dominate.
    2. SMOTE synthetically over-samples the minority class (spam) so the
       classifier sees a balanced 50/50 split during training.  Without this,
       models can cheat by predicting "ham" almost every time and still look
       accurate on the raw 87:13 imbalanced dataset.
    3. The fitted vectorizer is saved to disk so predict.py can reuse it
       without retraining.
    """
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features, sublinear_tf=True)

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)  # transform only — never fit on test data

    X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train_tfidf, y_train)

    os.makedirs(os.path.dirname(VECTORIZER_PATH), exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    return X_train_res, y_train_res, X_test_tfidf, vectorizer


@lru_cache(maxsize=1)
def load_vectorizer():
    return joblib.load(VECTORIZER_PATH)
