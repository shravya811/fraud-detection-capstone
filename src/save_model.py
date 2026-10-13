"""Step 11 - save the chosen pipeline as a .pkl, reload it and check the predictions match.

Run from the project root:  python -m src.save_model
Output: models/fraud_pipeline.pkl, models/model_metadata.json
"""
import json
import platform
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from src import config
from src.data_utils import load_clean_data, split_xy, time_split
from src.predict import predict_dataframe


def main():
    best = json.loads((config.REPORTS_DIR / "best_model.json").read_text())
    tuning = json.loads((config.REPORTS_DIR / "tuning_results.json").read_text())
    name = best["best_model"]
    fitted = joblib.load(config.CANDIDATES_DIR / f"{name}.pkl")

    # The resampler only matters while fitting. Remove it so the saved pipeline stays lean
    # (it then only needs scikit-learn, pandas and numpy to run).
    final = Pipeline([(n, step) for n, step in fitted.steps if n != "sampler"])

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, config.FINAL_MODEL_PATH)
    size_mb = config.FINAL_MODEL_PATH.stat().st_size / 1e6
    print(f"Saved {config.FINAL_MODEL_PATH} ({size_mb:.1f} MB)")

    # Reload and compare predictions on the full test set
    reloaded = joblib.load(config.FINAL_MODEL_PATH)
    df = load_clean_data()
    _, test_df, cutoff = time_split(df, config.TEST_ROW_FRACTION)
    X_test, _ = split_xy(test_df)
    labels_a, probs_a = predict_dataframe(final, X_test)
    labels_b, probs_b = predict_dataframe(reloaded, X_test)
    same_labels = bool(np.array_equal(labels_a, labels_b))
    same_scores = bool(np.allclose(probs_a, probs_b))
    print(f"Predictions identical after reload - labels: {same_labels}, "
          f"probabilities: {same_scores}")
    if not (same_labels and same_scores):
        raise SystemExit("Reloaded model does not match the original. Do not use this file.")

    metadata = {
        "model_name": name,
        "best_params": tuning[name]["best_params"],
        "imbalance_strategy": tuning[name]["imbalance_strategy"],
        "cv_average_precision": tuning[name]["cv_average_precision"],
        "test_metrics": best["test_metrics"],
        "test_cutoff_step": cutoff,
        "decision_threshold": "model default (probability 0.5 / SVM score 0)",
        "input_fields": config.RAW_FEATURES,
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "load_with": "joblib.load('models/fraud_pipeline.pkl')  # run from the project root",
        "python_executable": sys.executable,
    }
    config.MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"Saved {config.MODEL_METADATA_PATH}")


if __name__ == "__main__":
    main()
