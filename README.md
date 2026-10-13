# Online Fraud Transaction Detection - End-to-End ML, FastAPI, CI and Azure MLflow

Capstone Project 1, **Option A**. Predicts whether an online transaction is fraudulent
(target column `isFraud`), serves the model through FastAPI, runs lint + tests in GitHub
Actions, and tracks/registers/deploys the model with MLflow on Azure Machine Learning.

## Folder structure
```
capstone-project/
├── data/raw/            put onlinefraud.csv here (NOT committed: 493 MB; nested downloads are also detected)
├── data/processed/      cleaned cache created by the code (not committed)
├── notebooks/           01_inspection_and_eda.ipynb
├── src/                 config, data_utils, features, pipeline, metrics, and the step scripts
├── models/              fraud_pipeline.pkl + model_metadata.json (the .pkl is not committed)
├── api/                 FastAPI app, Pydantic schemas, model loader, Azure client
├── mlops/               MLflow logging, registry, Azure compute, deployment, monitoring
├── tests/               pytest + FastAPI TestClient
├── reports/             generated tables, charts and logs for your report
├── docs/                AZURE_GUIDE.md, REPORT_TEMPLATE.md
├── .github/workflows/ci.yml
├── requirements.txt / requirements-azure.txt
└── README.md
```

## Setup (Windows PowerShell, from the project folder)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip check
python -m src.check_env
```
Always run scripts **from the project root** with `python -m ...` (the saved model refers to `src.features`).

## Run order - Section 1 (ML)
| Step | Command | Produces |
|---|---|---|
| 1 Inspect | `python -m src.inspect_data` | `reports/data_inspection.txt` |
| 2 Clean | `python -m src.prepare_data` | `reports/cleaning_report.json` |
| 3 EDA | `python -m src.eda` | `reports/figures/01..07_*.png` |
| 5 Imbalance | `python -m src.compare_imbalance` | `reports/imbalance_comparison.csv` |
| 7-8 Train+tune | `python -m src.train_models` | `models/candidates/*.pkl`, `reports/tuning_results.json` |
| 9-10 Evaluate | `python -m src.evaluate` | `reports/model_comparison.csv`, ROC/PR/confusion charts |
| 11 Save | `python -m src.save_model` | `models/fraud_pipeline.pkl` |

Training tip: run one model at a time, e.g. `python -m src.train_models --models logistic_regression`.

## Section 2 (API, tests, CI)
```powershell
pytest -v
ruff check .
uvicorn api.main:app --reload        # then open http://127.0.0.1:8000/docs
```
Example request / response:
```json
{"step": 1, "type": "TRANSFER", "amount": 181.0, "oldbalanceOrg": 181.0,
 "newbalanceOrig": 0.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0}
```
```json
{"prediction": 1, "label": "fraud", "fraud_probability": 0.98}
```
(The response above only shows the format. Real values come from your trained model.)

## Section 3 (Azure)
See `docs/AZURE_GUIDE.md`.

## Design decisions
- **Split:** time-based on `step` (last ~20% of rows are the test set), stratified 3-fold CV inside training.
- **Imbalance:** class weights, random under/over-sampling and SMOTE compared on a later slice of the training data.
- **Models:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting (`HistGradientBoostingClassifier`) and a linear SVM (`LinearSVC`).
  A kernel SVM and the classic `GradientBoostingClassifier` are too slow on 6.3M rows.
- **Model choice:** by cross-validated PR-AUC on the training set; test metrics are only reported.
- **Dropped columns:** `nameOrig`, `nameDest` (identifiers), `isFlaggedFraud` (the system's own rule output).

## Limitations
This is a capstone, not a production fraud system. The balance columns are recorded after the
transaction and fraud rows almost always empty the sender's account, so metrics can look
unrealistically high. See `docs/REPORT_TEMPLATE.md`.
