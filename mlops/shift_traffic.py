"""Blue/green rollout and rollback.

    python -m mlops.shift_traffic --endpoint-name <name> --blue 90 --green 10   # canary
    python -m mlops.shift_traffic --endpoint-name <name> --blue 0 --green 100   # full switch
    python -m mlops.shift_traffic --endpoint-name <name> --blue 100 --green 0   # ROLLBACK
"""
import argparse

from mlops.azure_config import get_ml_client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--blue", type=int, required=True)
    parser.add_argument("--green", type=int, required=True)
    args = parser.parse_args()
    if args.blue + args.green != 100:
        raise SystemExit("Percentages must add up to 100")

    client = get_ml_client()
    endpoint = client.online_endpoints.get(args.endpoint_name)
    endpoint.traffic = {"blue": args.blue, "green": args.green}
    client.online_endpoints.begin_create_or_update(endpoint).result()
    print("New traffic split:", client.online_endpoints.get(args.endpoint_name).traffic)


if __name__ == "__main__":
    main()
