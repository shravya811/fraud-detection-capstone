"""Step 18 - pick the best run from MLflow and register it in the model registry.

    python -m mlops.register_best_model            # registers the best run as a new version
    python -m mlops.register_best_model --promote 1   # mark version 1 as Production

The best run is chosen by cross-validated PR-AUC (cv_pr_auc), not by test-set results.
Version labels use model-version TAGS, which Azure ML supports; an alias is also attempted.
"""
import argparse

import mlflow
from mlflow.tracking import MlflowClient

from mlops.azure_config import EXPERIMENT_NAME, REGISTERED_MODEL_NAME, get_tracking_uri


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--promote", type=str, help="model version to mark as Production")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()

    if args.local:
        from src.config import ROOT
        mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    else:
        mlflow.set_tracking_uri(get_tracking_uri())
    client = MlflowClient()

    if args.promote:
        client.set_model_version_tag(REGISTERED_MODEL_NAME, args.promote, "stage", "Production")
        try:
            client.set_registered_model_alias(REGISTERED_MODEL_NAME, "production", args.promote)
        except Exception as error:  # aliases may not be available in every workspace
            print("Alias not set (tag was set):", type(error).__name__)
        print(f"Version {args.promote} is now tagged stage=Production")
        return

    runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME],
                              order_by=["metrics.cv_pr_auc DESC"], max_results=10)
    if runs.empty:
        raise SystemExit("No runs found. Run python -m mlops.log_to_mlflow first.")
    best = runs.iloc[0]
    print(f"Best run: {best['tags.mlflow.runName']} (cv_pr_auc={best['metrics.cv_pr_auc']:.4f})")

    version = mlflow.register_model(model_uri=f"runs:/{best['run_id']}/model",
                                    name=REGISTERED_MODEL_NAME)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, "stage", "Staging")
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, "source_run_id",
                                 best["run_id"])
    print(f"Registered '{REGISTERED_MODEL_NAME}' version {version.version} with stage=Staging")


if __name__ == "__main__":
    main()
