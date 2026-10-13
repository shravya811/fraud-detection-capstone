"""Lets the FastAPI service call the deployed Azure ML endpoint as a client.

The endpoint URL and key come from environment variables - never from the code:
    AZURE_ENDPOINT_URL   (the scoring URI of the endpoint)
    AZURE_ENDPOINT_KEY   (the endpoint key)
    AZURE_DEPLOYMENT_NAME (optional, send the request to one specific deployment)
"""
import os

import httpx


class AzureNotConfiguredError(RuntimeError):
    pass


def score_with_azure(record):
    url = os.getenv("AZURE_ENDPOINT_URL")
    key = os.getenv("AZURE_ENDPOINT_KEY")
    if not url or not key:
        raise AzureNotConfiguredError("AZURE_ENDPOINT_URL and AZURE_ENDPOINT_KEY are not set")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    deployment = os.getenv("AZURE_DEPLOYMENT_NAME")
    if deployment:
        headers["azureml-model-deployment"] = deployment
    response = httpx.post(url, json={"data": [record]}, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()["predictions"][0]
