"""Step 2 - clean the data and print how many rows each decision affected.

Run from the project root:  python -m src.prepare_data
Output: reports/cleaning_report.json and data/processed/transactions_clean.pkl
"""
import json

from src import config
from src.data_utils import load_clean_data


def main():
    df = load_clean_data(force_rebuild=True)
    with open(config.REPORTS_DIR / "cleaning_report.json") as f:
        report = json.load(f)
    print(json.dumps(report, indent=2))
    print(f"\nClean table: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Saved to {config.CLEAN_DATA_CACHE}")


if __name__ == "__main__":
    main()
