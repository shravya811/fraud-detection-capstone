"""Step 17 - log the five trained candidate models to MLflow (one run per model).

It re-uses the saved candidates and the shared evaluation code, so nothing is trained twice.

    python -m mlops.log_to_mlflow            # logs to your Azure ML workspace
    python -m mlops.log_to_mlflow --local    # dry run into a local ./mlruns folder
"""
import argparse
import json

import joblib
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from mlops.azure_config import EXPERIMENT_NAME, get_tracking_uri
from src import config
from src.data_utils import load_clean_data, split_xy, time_split
from src.metrics import (
    compute_metrics,
    feature_importance_table,
    get_scores,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pr_curve,
    plot_roc_curve,
)


def main():
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="log to ./mlruns instead of Azure")
    args = parser.parse_args()

    if args.local:
        mlflow.set_tracking_uri((config.ROOT / "mlruns").as_uri())
    else:
        mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    print("Tracking URI:", mlflow.get_tracking_uri())

    df = load_clean_data()
    _, test_df, cutoff = time_split(df, config.TEST_ROW_FRACTION)
    X_test, y_test = split_xy(test_df)
    tuning = json.loads((config.REPORTS_DIR / "tuning_results.json").read_text())

    run_ids = {}
    for name in config.MODEL_NAMES:
        if name not in tuning:
            print(f"Skipping {name}: no saved candidate")
            continue
        model = joblib.load(config.CANDIDATES_DIR / f"{name}.pkl")
        scores = get_scores(model, X_test)
        predictions = model.predict(X_test)
        metrics = compute_metrics(y_test, predictions, scores)

        with mlflow.start_run(run_name=f"{name}-tuned") as run:
            mlflow.set_tag("model_family", name)
            mlflow.log_params({
                "model": name,
                "imbalance_strategy": tuning[name]["imbalance_strategy"],
                "test_cutoff_step": cutoff,
                "cv_folds": config.CV_FOLDS,
                **tuning[name]["best_params"],
            })
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            mlflow.log_metric("cv_pr_auc", tuning[name]["cv_average_precision"])
            mlflow.log_metric("train_seconds", tuning[name]["train_seconds"])

            mlflow.log_figure(plot_confusion_matrix(y_test, predictions, name),
                              "plots/confusion_matrix.png")
            mlflow.log_figure(plot_roc_curve(y_test, scores, name), "plots/roc_curve.png")
            mlflow.log_figure(plot_pr_curve(y_test, scores, name), "plots/pr_curve.png")
            importance = feature_importance_table(model, X_test, y_test)
            mlflow.log_figure(plot_feature_importance(importance, f"Feature importance - {name}"),
                              "plots/feature_importance.png")
            plt.close("all")

            # code_paths ships src/ with the model so it can be unpickled anywhere
            mlflow.sklearn.log_model(model, artifact_path="model",
                                     code_paths=[str(config.ROOT / "src")])
            run_ids[name] = run.info.run_id
            print(f"Logged {name}: run_id={run.info.run_id} pr_auc={metrics['pr_auc']:.4f}")

    (config.REPORTS_DIR / "mlflow_runs.json").write_text(json.dumps(run_ids, indent=2))
    print("Open the MLflow UI from Azure ML Studio -> Jobs -> fraud-detection-capstone")


if __name__ == "__main__":
    main()
