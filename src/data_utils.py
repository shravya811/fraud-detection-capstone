"""Loading, cleaning and splitting the data."""
import json

import numpy as np
import pandas as pd

from src import config

RAW_DTYPES = {
    "step": "int16",
    "type": "category",
    "amount": "float64",
    "nameOrig": "str",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "nameDest": "str",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}


def load_raw_data(path=None):
    """Reads the full CSV (every row, all 11 columns) with memory-friendly data types."""
    path = config.resolve_raw_data_path() if path is None else path
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found. Checked {config.RAW_DATA_PATH} and "
            f"{config.NESTED_RAW_DATA_PATH}. Put onlinefraud.csv in data/raw/."
        )
    return pd.read_csv(path, dtype=RAW_DTYPES)


def clean_data(raw):
    """Cleans the raw data and returns (clean_df, report). The raw file is never modified."""
    report = {"rows_raw": int(len(raw)), "fraud_raw": int(raw[config.TARGET].sum())}

    # 1. Missing values: drop rows that have any (the real file has none)
    missing_rows = int(raw.isna().any(axis=1).sum())
    report["missing_values_total"] = int(raw.isna().sum().sum())
    df = raw.dropna() if missing_rows else raw

    # 2. Duplicates: only rows identical in ALL 11 columns (including account names)
    duplicates = int(df.duplicated().sum())
    report["duplicate_rows_removed"] = duplicates
    if duplicates:
        df = df.drop_duplicates()

    # 3. Impossible values: negative money. Dropped as invalid (the real file has none).
    negative_mask = (df[config.MONEY_COLUMNS] < 0).any(axis=1)
    report["negative_money_rows_removed"] = int(negative_mask.sum())
    df = df[~negative_mask]

    # 4. Zero-amount rows: kept, because they may be real fraud (reported, not removed)
    zero_amount = df["amount"] == 0
    report["zero_amount_rows_kept"] = int(zero_amount.sum())
    report["zero_amount_rows_that_are_fraud"] = int(df.loc[zero_amount, config.TARGET].sum())

    # 5. Outliers: counted and kept. Extreme amounts can be genuine fraud signals.
    outliers = {}
    for col in config.MONEY_COLUMNS:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        mask = df[col] > q3 + 3 * (q3 - q1)
        outliers[col] = {
            "rows_above_3xIQR_fence": int(mask.sum()),
            "fraud_rate_in_those_rows": round(float(df.loc[mask, config.TARGET].mean()), 5)
            if mask.any() else None,
            "decision": "kept; handled with a signed log transform instead of removal",
        }
    report["outliers"] = outliers

    # 6. Columns not used for modelling
    report["columns_dropped"] = {
        "nameOrig, nameDest": "account identifiers: almost every value is unique, so they "
                              "cannot generalise to new accounts",
        "isFlaggedFraud": "output of the system's own rule (only 16 rows are 1), not an input "
                          "that exists before scoring",
    }
    keep = config.RAW_FEATURES + [config.TARGET]
    df = df[keep].reset_index(drop=True)
    report["rows_clean"] = int(len(df))
    report["fraud_clean"] = int(df[config.TARGET].sum())
    return df, report


def load_clean_data(force_rebuild=False):
    """Returns the cleaned modelling table. Built once, then cached in data/processed/."""
    if config.CLEAN_DATA_CACHE.exists() and not force_rebuild:
        return pd.read_pickle(config.CLEAN_DATA_CACHE)
    raw = load_raw_data()
    df, report = clean_data(raw)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(config.CLEAN_DATA_CACHE)
    with open(config.REPORTS_DIR / "cleaning_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return df


def time_split(df, test_fraction):
    """Time-based split: the LAST rows (highest step values) become the test set.

    Returns (train_df, test_df, cutoff_step). Rows with step <= cutoff_step are training rows.
    """
    rows_per_step = df.groupby("step", observed=True).size().sort_index()
    cumulative_share = rows_per_step.cumsum() / rows_per_step.sum()
    cutoff_step = int(cumulative_share[cumulative_share >= 1 - test_fraction].index[0])
    train_df = df[df["step"] <= cutoff_step]
    test_df = df[df["step"] > cutoff_step]
    return train_df, test_df, cutoff_step


def split_xy(df):
    """Feature matrix X (the 7 raw fields) and target y."""
    return df[config.RAW_FEATURES], df[config.TARGET].to_numpy()


def class_counts(y):
    y = np.asarray(y)
    return {"legitimate": int((y == 0).sum()), "fraud": int((y == 1).sum())}
