"""Step 19 - deploy the registered model to a managed online endpoint (blue/green ready).

    python -m mlops.deploy_endpoint --endpoint-name fraud-ep-<yourname> --model-version 1 \
        --deployment blue

COST NOTE: a managed online deployment bills for its VM the whole time it exists.
Delete it when finished:  python -m mlops.deploy_endpoint --endpoint-name <name> --delete

The scoring environment is generated from the package versions installed on YOUR machine,
so the pickled model is loaded with the same scikit-learn / numpy / pandas versions.
"""
import argparse
import importlib.metadata as md
import shutil

from azure.ai.ml.entities import (
    CodeConfiguration,
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
)

from mlops.azure_config import REGISTERED_MODEL_NAME, get_ml_client
from src.config import ROOT

BASE_IMAGE = "mcr.microsoft.com/azureml/openmpi4.1.2-ubuntu20.04:latest"


def build_code_folder():
    """Copies only what scoring needs into build/score_code (never the 490 MB dataset)."""
    folder = ROOT / "build" / "score_code"
    if folder.exists():
        shutil.rmtree(folder)
    (folder / "src").mkdir(parents=True)
    shutil.copy(ROOT / "mlops" / "score.py", folder / "score.py")
    shutil.copy(ROOT / "src" / "__init__.py", folder / "src" / "__init__.py")
    shutil.copy(ROOT / "src" / "features.py", folder / "src" / "features.py")
    return folder


def build_conda_file():
    pins = [f"{p}=={md.version(p)}" for p in ("numpy", "pandas", "scikit-learn", "joblib")]
    lines = ["name: fraud-scoring", "channels:", "  - conda-forge", "dependencies:",
             "  - python=3.11", "  - pip", "  - pip:"]
    lines += [f"      - {p}" for p in pins + ["azureml-inference-server-http"]]
    path = ROOT / "build" / "conda.yml"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--model-version", default="1")
    parser.add_argument("--deployment", default="blue", help="blue or green")
    parser.add_argument("--instance-type", default="Standard_DS3_v2")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--delete", action="store_true", help="delete the whole endpoint")
    args = parser.parse_args()

    client = get_ml_client()
    if args.delete:
        client.online_endpoints.begin_delete(name=args.endpoint_name).result()
        print("Endpoint deleted.")
        return

    endpoint = ManagedOnlineEndpoint(name=args.endpoint_name, auth_mode="key",
                                     description="Online fraud detection")
    client.online_endpoints.begin_create_or_update(endpoint).result()

    model = client.models.get(name=REGISTERED_MODEL_NAME, version=args.model_version)
    environment = Environment(name="fraud-sklearn-env", image=BASE_IMAGE,
                              conda_file=str(build_conda_file()))
    deployment = ManagedOnlineDeployment(
        name=args.deployment,
        endpoint_name=args.endpoint_name,
        model=model,
        environment=environment,
        code_configuration=CodeConfiguration(code=str(build_code_folder()),
                                             scoring_script="score.py"),
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        app_insights_enabled=True,  # request volume, latency and errors in Application Insights
    )
    print(f"Creating deployment '{args.deployment}' (this takes several minutes) ...")
    client.online_deployments.begin_create_or_update(deployment).result()

    endpoint = client.online_endpoints.get(args.endpoint_name)
    if not endpoint.traffic:  # first deployment takes all traffic; later ones start at 0%
        endpoint.traffic = {args.deployment: 100}
    else:
        endpoint.traffic = {**endpoint.traffic, args.deployment: endpoint.traffic.get(
            args.deployment, 0)}
    client.online_endpoints.begin_create_or_update(endpoint).result()
    print("Traffic:", client.online_endpoints.get(args.endpoint_name).traffic)


if __name__ == "__main__":
    main()
