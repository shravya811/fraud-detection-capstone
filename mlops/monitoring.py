"""Step 20 - simple data-drift and performance monitoring plus a retraining rule.

    python -m mlops.monitoring --build-reference            # once, after training
    python -m mlops.monitoring --check recent.csv           # on new transactions

recent.csv needs the 7 request fields. If it also has an isFraud column (labels that became
known later), recall and precision are checked against the training-time test results.

Drift measure: Population Stability Index (PSI) per feature, using bins fixed at training time.
    PSI < 0.10 stable | 0.10-0.25 moderate shift | > 0.25 significant shift
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from src import config
from src.data_utils import load_clean_data, time_split
from src.metrics import compute_metrics, get_scores

MONITORED = config.MONEY_COLUMNS
PSI_ALERT = 0.25
RECALL_DROP_ALERT = 0.10


def _proportions(values, inner_edges):
    bins = np.concatenate([[-np.inf], inner_edges, [np.inf]])
    counts, _ = np.histogram(values, bins=bins)
    return (counts / max(counts.sum(), 1)).tolist()


def build_reference():
    df = load_clean_data()
    train_df, _, _ = time_split(df, config.TEST_ROW_FRACTION)
    reference = {}
    for col in MONITORED:
        inner = np.unique(np.quantile(train_df[col], np.linspace(0, 1, 11)))[1:-1]
        reference[col] = {"edges": inner.tolist(),
                          "proportions": _proportions(train_df[col], inner)}
    reference["type_share"] = train_df["type"].astype(str).value_counts(normalize=True).to_dict()
    config.REFERENCE_STATS_PATH.write_text(json.dumps(reference, indent=2))
    print("Saved", config.REFERENCE_STATS_PATH)


def psi(expected, actual, eps=1e-4):
    e = np.clip(np.asarray(expected), eps, None)
    a = np.clip(np.asarray(actual), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def check(recent_path):
    reference = json.loads(config.REFERENCE_STATS_PATH.read_text())
    recent = pd.read_csv(recent_path)
    retrain_reasons = []

    print("Data drift (PSI per feature):")
    for col in MONITORED:
        edges = np.array(reference[col]["edges"])
        value = psi(reference[col]["proportions"], _proportions(recent[col], edges))
        status = "STABLE" if value < 0.10 else "MODERATE" if value < PSI_ALERT else "SIGNIFICANT"
        print(f"  {col:16s} PSI={value:.3f}  {status}")
        if value >= PSI_ALERT:
            retrain_reasons.append(f"{col} drifted (PSI {value:.2f})")

    if config.TARGET in recent.columns and config.FINAL_MODEL_PATH.exists():
        model = joblib.load(config.FINAL_MODEL_PATH)
        X = recent[config.RAW_FEATURES]
        result = compute_metrics(recent[config.TARGET], model.predict(X), get_scores(model, X))
        baseline = json.loads(config.MODEL_METADATA_PATH.read_text())["test_metrics"]["recall"]
        print(f"\nPerformance on recent labelled data: recall={result['recall']:.3f} "
              f"precision={result['precision']:.3f} pr_auc={result['pr_auc']:.3f} "
              f"(baseline recall {baseline:.3f})")
        if baseline - result["recall"] > RECALL_DROP_ALERT:
            retrain_reasons.append("recall dropped by more than 0.10")

    print("\nRETRAIN RECOMMENDED: " + "; ".join(retrain_reasons) if retrain_reasons
          else "\nNo retraining trigger fired. (Also retrain on a schedule, e.g. monthly.)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-reference", action="store_true")
    parser.add_argument("--check", type=str, help="CSV of recent transactions")
    args = parser.parse_args()
    if args.build_reference:
        build_reference()
    elif args.check:
        check(args.check)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
