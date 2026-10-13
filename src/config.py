"""Central settings for the whole project. Every other file imports from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Folders and files
RAW_DATA_DIR = ROOT / "data" / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "onlinefraud.csv"
NESTED_RAW_DATA_PATH = RAW_DATA_DIR / "onlinefraud" / "onlinefraud.csv"


def resolve_raw_data_path():
    """Prefer the documented location, while supporting the downloaded nested layout."""
    for path in (RAW_DATA_PATH, NESTED_RAW_DATA_PATH):
        if path.exists():
            return path
    return RAW_DATA_PATH


PROCESSED_DIR = ROOT / "data" / "processed"
CLEAN_DATA_CACHE = PROCESSED_DIR / "transactions_clean.pkl"
MODELS_DIR = ROOT / "models"
CANDIDATES_DIR = MODELS_DIR / "candidates"
FINAL_MODEL_PATH = MODELS_DIR / "fraud_pipeline.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
REFERENCE_STATS_PATH = MODELS_DIR / "reference_stats.json"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Column names in the real dataset
TARGET = "isFraud"
IDENTIFIER_COLUMNS = ["nameOrig", "nameDest"]
RULE_FLAG_COLUMN = "isFlaggedFraud"
# The 7 raw fields a prediction request must contain (also the API schema)
RAW_FEATURES = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
MONEY_COLUMNS = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]

# Reproducibility and validation
RANDOM_STATE = 42
TEST_ROW_FRACTION = 0.20        # last ~20% of rows (by time step) are the final test set
VALIDATION_ROW_FRACTION = 0.15  # used only by the imbalance comparison, taken from training data
CV_FOLDS = 3

# Ranges taken from the real data (step 1-743 hours, largest amounts below 1e9)
STEP_MIN, STEP_MAX = 1, 743
MAX_MONEY = 1e10

MODEL_NAMES = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "svm",
]
