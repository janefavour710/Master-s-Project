from typing import Optional, Tuple
import numpy as np
from src.preprocess import clean_text
from src.features import load_vectorizer
from src.train import load_model


def predict(message, model_name) -> Tuple[str, Optional[float]]:
    """Classify a single raw SMS message.

    Returns
    -------
    label       : "Spam" or "Ham"
    confidence  : probability that the label is correct (0–1), or None if the
                  model has no probability output.

    Note: LinearSVC has no predict_proba, so we convert its decision score
    to a rough probability using the sigmoid function.
    """
    vectorizer = load_vectorizer()
    clf = load_model(model_name)

    features = vectorizer.transform([clean_text(message)])
    label_int = clf.predict(features)[0]
    label = "Spam" if label_int == 1 else "Ham"

    if hasattr(clf, "predict_proba"):
        confidence = float(clf.predict_proba(features)[0][label_int])
    elif hasattr(clf, "decision_function"):
        # Sigmoid converts an unbounded score into a 0–1 probability estimate
        score = clf.decision_function(features)[0]
        confidence = float(1 / (1 + np.exp(-score)))
    else:
        confidence = None

    return label, confidence
