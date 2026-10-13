"""Loads the saved pipeline."""
import os
from pathlib import Path

import joblib

from src.config import FINAL_MODEL_PATH


def get_model_path():
    return Path(os.getenv("FRAUD_MODEL_PATH", str(FINAL_MODEL_PATH)))


def load_model():
    """Returns the fitted pipeline, or None if the file does not exist."""
    path = get_model_path()
    if not path.exists():
        return None
    return joblib.load(path)
