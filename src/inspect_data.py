"""Step 1 - load the full raw CSV and report what it contains.

Run from the project root:  python -m src.inspect_data
Output: printed on screen and saved to reports/data_inspection.txt
"""
from src import config
from src.data_utils import clean_data, load_raw_data


def main():
    lines = []

    def show(text=""):
        print(text)
        lines.append(str(text))

    raw = load_raw_data()
    show("=== SHAPE ===")
    show(f"{raw.shape[0]:,} rows x {raw.shape[1]} columns")
    show(f"Memory used: {raw.memory_usage(deep=True).sum() / 1e6:.0f} MB")
    show("\n=== FIRST 5 ROWS ===")
    show(raw.head().to_string())
    show("\n=== DATA TYPES ===")
    show(raw.dtypes.to_string())
    show("\n=== SUMMARY STATISTICS (numeric columns) ===")
    show(raw.describe().T.to_string())
    show("\n=== MISSING VALUES PER COLUMN ===")
    show(raw.isna().sum().to_string())
    show(f"\n=== DUPLICATE ROWS (identical in all columns) === {int(raw.duplicated().sum())}")

    show("\n=== TARGET: isFraud ===")
    counts = raw[config.TARGET].value_counts()
    show(counts.to_string())
    show(f"Fraud share: {raw[config.TARGET].mean() * 100:.4f}%")
    show("\n=== FRAUD BY TRANSACTION TYPE ===")
    by_type = raw.groupby("type", observed=True)[config.TARGET].agg(["count", "sum", "mean"])
    show(by_type.to_string())

    show("\n=== IDENTIFIER COLUMNS ===")
    for col in config.IDENTIFIER_COLUMNS:
        show(f"{col}: {raw[col].nunique():,} unique values in {len(raw):,} rows")
    flagged = raw[config.RULE_FLAG_COLUMN]
    show(f"\nisFlaggedFraud = 1 in {int(flagged.sum())} rows; "
         f"of those, fraud = {int(raw.loc[flagged == 1, config.TARGET].sum())}")
    show(f"step ranges from {raw['step'].min()} to {raw['step'].max()} (one step = one hour)")

    show("\n=== LEAKAGE CHECK: share of rows where the sender's new balance is exactly 0 ===")
    show(raw.groupby(config.TARGET)["newbalanceOrig"].apply(lambda s: (s == 0).mean()).to_string())

    _, report = clean_data(raw)
    show("\n=== WHAT THE CLEANING STEP WOULD DO ===")
    for key, value in report.items():
        if key not in ("outliers", "columns_dropped"):
            show(f"{key}: {value}")

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.REPORTS_DIR / "data_inspection.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
