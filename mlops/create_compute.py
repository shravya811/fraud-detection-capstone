"""Creates the compute cluster and compute instance in the Azure ML workspace.

    python -m mlops.create_compute --instance-name fraud-ci-<yourname>

COST NOTE: a compute instance bills while it is running - stop it when you are not using it.
The cluster scales to 0 nodes when idle, so it costs nothing while idle.
"""
import argparse

from azure.ai.ml.entities import AmlCompute, ComputeInstance

from mlops.azure_config import get_ml_client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-name", required=True,
                        help="3-24 chars, letters/digits/hyphens, unique in the Azure region")
    parser.add_argument("--vm-size", default="Standard_DS3_v2")
    args = parser.parse_args()

    client = get_ml_client()
    cluster = AmlCompute(name="cpu-cluster", size=args.vm_size, min_instances=0,
                         max_instances=1, idle_time_before_scale_down=120)
    print("Creating compute cluster 'cpu-cluster' ...")
    client.compute.begin_create_or_update(cluster).result()

    instance = ComputeInstance(name=args.instance_name, size=args.vm_size)
    print(f"Creating compute instance '{args.instance_name}' ...")
    client.compute.begin_create_or_update(instance).result()
    print("Done. Remember to STOP the compute instance in Azure ML Studio when finished.")


if __name__ == "__main__":
    main()
