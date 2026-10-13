"""Prints the versions of key packages and confirms the dataset file is present."""
import importlib.metadata as md
import sys

from src import config

packages = [
    "numpy", "pandas", "scikit-learn", "imbalanced-learn", "matplotlib",
    "seaborn", "joblib", "fastapi", "uvicorn", "pydantic", "pytest", "ruff",
    "langchain", "langchain-core", "langchain-community",
]

print("Python:", sys.version.split()[0])
for name in packages:
    try:
        print(f"{name}: {md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name}: NOT INSTALLED")

csv_path = config.resolve_raw_data_path()
if csv_path.exists():
    print(f"Dataset found: {csv_path} ({csv_path.stat().st_size / 1e6:.1f} MB)")
else:
    print("Dataset NOT found. Checked:")
    print(" -", config.RAW_DATA_PATH)
    print(" -", config.NESTED_RAW_DATA_PATH)

installed = {d.metadata["Name"].lower() for d in md.distributions()}
hf = [p for p in ("transformers", "huggingface-hub", "datasets", "sentence-transformers")
      if p in installed]
print("Hugging Face packages installed:", hf if hf else "none")
