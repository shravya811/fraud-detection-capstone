# Report skeleton (fill ONLY from real outputs)

Mark anything not yet done as **PENDING**. Never paste numbers you did not produce.

1. Title page, abstract, problem statement (Track: Option A - Online Fraud Detection) - PENDING
2. Dataset description and preprocessing - source: `reports/data_inspection.txt`, `reports/cleaning_report.json`
3. EDA and insights - source: `reports/figures/01..07_*.png`
4. Feature engineering, imbalance handling, validation strategy - source: `src/features.py`,
   `reports/imbalance_comparison.csv` (time-based split, stratified 3-fold CV)
5. Model building, tuning, evaluation, selection - source: `reports/tuning_results.json`,
   `reports/model_comparison.csv`, `reports/figures/confusion_matrices.png`, `roc_curves.png`, `pr_curves.png`
6. Model serialisation and validation - source: terminal output of `python -m src.save_model`,
   `models/model_metadata.json`
7. FastAPI design, endpoints, testing, CI - source: Swagger UI, `pytest -v` output, GitHub Actions run
8. Azure MLflow: tracking, registry, deployment, monitoring - source: Azure ML Studio screenshots
9. Results, limitations, ethics (false positives/negatives, bias, privacy), future work
10. References and appendix

## Limitations you must discuss (they are real properties of this dataset)
- Balance columns are recorded after the transaction; fraud rows almost always empty the sender's
  account (98% have a new balance of 0), so scores can look unrealistically high.
- The data is simulated, with one month of history; hour-of-day patterns may not match real customers.
- Nearly every account appears once, so per-account velocity features are not possible.
- Capstone vs production: no real-time feature store, no human review queue, no fairness audit, no
  adversarial testing, no latency or cost budget.

## Screenshot checklist (save in `reports/screenshots/`)
1. `python -m src.inspect_data` output  2. cleaning report  3. each EDA chart
4. `python -m src.compare_imbalance` table  5. `python -m src.train_models` best scores
6. `python -m src.evaluate` table + confusion matrix, ROC, PR images  7. `save_model` output
   and `models/fraud_pipeline.pkl` in the file explorer  8. Swagger UI `/docs` with a /predict call
9. `pytest -v` passing  10. GitHub Actions green run  11-14. Azure screenshots (see AZURE_GUIDE.md)
