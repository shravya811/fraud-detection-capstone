"""Step 5 - compare ways of handling the class imbalance.

Every strategy is fitted on the EARLY part of the training data and scored on the LATER
part of the training data. The final test set is not touched. Resampling happens inside the
pipeline, so it only ever sees training rows.

Run from the project root:  python -m src.compare_imbalance
Output: reports/imbalance_comparison.csv and reports/imbalance_best.json
"""
import json
import time

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.data_utils import class_counts, load_clean_data, split_xy, time_split
from src.metrics import compute_metrics, get_scores
from src.pipeline import IMBALANCE_STRATEGIES, build_pipeline


def main():
    df = load_clean_data()
    train_df, _, test_cutoff = time_split(df, config.TEST_ROW_FRACTION)
    fit_df, val_df, val_cutoff = time_split(train_df, config.VALIDATION_ROW_FRACTION)
    X_fit, y_fit = split_xy(fit_df)
    X_val, y_val = split_xy(val_df)
    print(f"Fit rows: {len(fit_df):,} {class_counts(y_fit)}  (steps <= {val_cutoff})")
    print(f"Validation rows: {len(val_df):,} {class_counts(y_val)}  (steps {val_cutoff + 1}"
          f"-{test_cutoff})")

    rows = []
    for strategy in IMBALANCE_STRATEGIES:
        # A fixed, fast Decision Tree keeps the comparison fair and quick
        tree = DecisionTreeClassifier(
            max_depth=10, min_samples_leaf=20, random_state=config.RANDOM_STATE,
            class_weight="balanced" if strategy == "class_weight" else None,
        )
        pipeline = build_pipeline("decision_tree", strategy, estimator=tree)
        start = time.time()
        pipeline.fit(X_fit, y_fit)
        seconds = time.time() - start
        scores = get_scores(pipeline, X_val)
        result = compute_metrics(y_val, pipeline.predict(X_val), scores)
        result.update({"strategy": strategy, "fit_seconds": round(seconds, 1)})
        rows.append(result)
        print(f"{strategy:13s} PR-AUC={result['pr_auc']:.4f} recall={result['recall']:.4f} "
              f"precision={result['precision']:.4f} f1={result['f1']:.4f} ({seconds:.0f}s)")

    table = pd.DataFrame(rows).set_index("strategy")
    table = table[["pr_auc", "roc_auc", "recall", "precision", "f1", "accuracy",
                   "false_negatives", "false_positives", "fit_seconds"]]
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.REPORTS_DIR / "imbalance_comparison.csv")
    best = table["pr_auc"].idxmax()
    with open(config.REPORTS_DIR / "imbalance_best.json", "w") as f:
        json.dump({"best_strategy": best, "chosen_by": "highest PR-AUC on validation"}, f, indent=2)
    print("\n", table.round(4).to_string())
    print(f"\nBest strategy by PR-AUC: {best}  (train_models.py will use it)")


if __name__ == "__main__":
    main()
