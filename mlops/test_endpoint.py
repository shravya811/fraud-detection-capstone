"""Step 20 - call the deployed endpoint with its key (the key is fetched, never hard-coded).

    python -m mlops.test_endpoint --endpoint-name <name>
"""
import argparse
import json

import httpx

from mlops.azure_config import get_ml_client

SAMPLE = {"step": 1, "type": "TRANSFER", "amount": 181.0, "oldbalanceOrg": 181.0,
          "newbalanceOrig": 0.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-name", required=True)
    args = parser.parse_args()

    client = get_ml_client()
    endpoint = client.online_endpoints.get(args.endpoint_name)
    key = client.online_endpoints.get_keys(args.endpoint_name).primary_key
    response = httpx.post(
        endpoint.scoring_uri,
        json={"data": [SAMPLE]},
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=60,
    )
    print("Status:", response.status_code)
    print(json.dumps(response.json(), indent=2))
    print("\nTo let FastAPI call this endpoint, set (do not commit these):")
    print(f"  AZURE_ENDPOINT_URL = {endpoint.scoring_uri}")
    print("  AZURE_ENDPOINT_KEY = <the endpoint key from Azure ML Studio -> Endpoints -> Consume>")


if __name__ == "__main__":
    main()
