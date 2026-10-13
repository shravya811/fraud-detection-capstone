"""One place that turns raw transactions into labels and fraud probabilities."""
import numpy as np
import pandas as pd

from src.config import RAW_FEATURES


def predict_dataframe(model, df):
    """Returns (labels, fraud_probabilities) for a DataFrame with the 7 raw fields."""
    X = pd.DataFrame(df)[RAW_FEATURES]
    labels = model.predict(X)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
    else:
        # Linear SVM has no probabilities: squash its decision score to 0-1 (not calibrated).
        probabilities = 1.0 / (1.0 + np.exp(-model.decision_function(X)))
    return labels.astype(int), probabilities.astype(float)
