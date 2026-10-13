"""Steps 7-8 - train and tune the five models on the training set (cross-validated).

Run from the project root:
    python -m src.train_models                       (all five models)
    python -m src.train_models --models random_forest svm   (only some)

Each tuned pipeline is saved to models/candidates/<name>.pkl and its settings and
cross-validation score are saved to reports/tuning_results.json.
"""
import argparse
import json
import time

import joblib
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV, StratifiedKFold

from src import config
from src.data_utils import class_counts, load_clean_data, split_xy, time_split
from src.pipeline import PARAM_GRIDS, SEARCH_N_ITER, build_pipeline

RESULTS_PATH = config.REPORTS_DIR / "tuning_results.json"


def read_strategy():
    path = config.REPORTS_DIR / "imbalance_best.json"
    if path.exists():
        return json.loads(path.read_text())["best_strategy"]
    print("No imbalance_best.json found - using class weights. "
          "Run python -m src.compare_imbalance first for a data-driven choice.")
    return "class_weight"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=config.MODEL_NAMES,
                        choices=config.MODEL_NAMES)
    args = parser.parse_args()

    df = load_clean_data()
    train_df, _, cutoff = time_split(df, config.TEST_ROW_FRACTION)
    X_train, y_train = split_xy(train_df)
    strategy = read_strategy()
    print(f"Training rows: {len(train_df):,} (steps <= {cutoff}) {class_counts(y_train)}")
    print(f"Imbalance strategy: {strategy}")

    config.CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    for name in args.models:
        grid = PARAM_GRIDS[name]
        n_iter = min(SEARCH_N_ITER[name], len(ParameterGrid(grid)))
        print(f"\n=== {name}: trying {n_iter} settings x {config.CV_FOLDS} folds ===")
        search = RandomizedSearchCV(
            build_pipeline(name, strategy), grid, n_iter=n_iter, scoring="average_precision",
            cv=cv, refit=True, n_jobs=1, random_state=config.RANDOM_STATE, verbose=2,
        )
        start = time.time()
        search.fit(X_train, y_train)
        seconds = time.time() - start

        joblib.dump(search.best_estimator_, config.CANDIDATES_DIR / f"{name}.pkl")
        results[name] = {
            "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
            "cv_average_precision": float(search.best_score_),
            "imbalance_strategy": strategy,
            "train_seconds": round(seconds, 1),
            "test_cutoff_step": cutoff,
        }
        print(f"{name}: best CV average precision = {search.best_score_:.4f} "
              f"| params = {results[name]['best_params']} | {seconds / 60:.1f} min")
        RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))

    print(f"\nSaved tuning results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
