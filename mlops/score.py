"""Scoring script that runs INSIDE the Azure ML managed online endpoint.

Azure calls init() once when the container starts and run() for every request.
Request body:  {"data": [ {step, type, amount, oldbalanceOrg, newbalanceOrig,
                           oldbalanceDest, newbalanceDest}, ... ]}
Response body: {"predictions": [ {"prediction": 0/1, "fraud_probability": 0-1}, ... ]}
"""
import glob
import json
import logging
import os
import pickle

import numpy as np
import pandas as pd

FEATURES = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest"]
model = None


def init():
    global model
    model_dir = os.environ["AZUREML_MODEL_DIR"]
    paths = glob.glob(os.path.join(model_dir, "**", "model.pkl"), recursive=True)
    if not paths:
        raise FileNotFoundError(f"No model.pkl found under {model_dir}")
    with open(paths[0], "rb") as f:
        model = pickle.load(f)
    logging.info("Fraud model loaded from %s", paths[0])


def run(raw_data):
    records = json.loads(raw_data)["data"]
    df = pd.DataFrame(records)[FEATURES]
    labels = model.predict(df)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(df)[:, 1]
    else:
        probabilities = 1.0 / (1.0 + np.exp(-model.decision_function(df)))
    return {"predictions": [
        {"prediction": int(label), "fraud_probability": float(prob)}
        for label, prob in zip(labels, probabilities)
    ]}
