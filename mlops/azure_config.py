"""Shared Azure helpers. Settings come from environment variables - no secrets in code.

Set these in your terminal before running any mlops script (PowerShell):
    $env:AZURE_SUBSCRIPTION_ID = "<your subscription id>"
    $env:AZURE_RESOURCE_GROUP  = "rg-fraud-capstone"
    $env:AZURE_WORKSPACE_NAME  = "mlw-fraud-capstone"
and sign in once with:  az login
"""
import os

EXPERIMENT_NAME = "fraud-detection-capstone"
REGISTERED_MODEL_NAME = "fraud-detection-model"
ENDPOINT_NAME_ENV = "AZURE_ENDPOINT_NAME"


def get_ml_client():
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    missing = [v for v in ("AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_WORKSPACE_NAME")
               if not os.getenv(v)]
    if missing:
        raise SystemExit(f"Set these environment variables first: {', '.join(missing)}")
    return MLClient(
        DefaultAzureCredential(),
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_WORKSPACE_NAME"],
    )


def get_tracking_uri():
    """MLflow tracking URI of the Azure ML workspace."""
    client = get_ml_client()
    return client.workspaces.get(client.workspace_name).mlflow_tracking_uri
