"""FastAPI app.   Run from the project root:  uvicorn api.main:app --reload
Swagger UI: http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.azure_client import AzureNotConfiguredError, score_with_azure
from api.model_loader import load_model
from api.schemas import HealthResponse, PredictionResponse, TransactionRequest
from src.predict import predict_dataframe


@asynccontextmanager
async def lifespan(app):
    app.state.model = load_model()  # loaded ONCE at startup
    yield


app = FastAPI(title="Online Fraud Detection API", version="1.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Clear messages per field, without echoing the submitted values back."""
    errors = [
        {"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


def _to_response(label, probability):
    return PredictionResponse(
        prediction=int(label),
        label="fraud" if int(label) == 1 else "legitimate",
        fraud_probability=float(min(max(probability, 0.0), 1.0)),
    )


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    loaded = request.app.state.model is not None
    return HealthResponse(status="ok" if loaded else "model_not_loaded", model_loaded=loaded)


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest, request: Request):
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    labels, probabilities = predict_dataframe(model, pd.DataFrame([transaction.model_dump()]))
    return _to_response(labels[0], probabilities[0])


@app.post("/predict/azure", response_model=PredictionResponse)
def predict_with_azure(transaction: TransactionRequest):
    """Same request, but scored by the model deployed on the Azure ML managed online endpoint."""
    try:
        result = score_with_azure(transaction.model_dump())
    except AzureNotConfiguredError:
        raise HTTPException(status_code=503, detail="Azure endpoint is not configured")
    except httpx.HTTPError:
        raise HTTPException(status_code=502, detail="Azure endpoint call failed")
    return _to_response(result["prediction"], result["fraud_probability"])
