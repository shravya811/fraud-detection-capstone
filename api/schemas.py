"""Request and response models for the API. Ranges come from the real dataset."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.config import MAX_MONEY, STEP_MAX, STEP_MIN


class TransactionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "step": 1,
                "type": "TRANSFER",
                "amount": 181.0,
                "oldbalanceOrg": 181.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
            }
        },
    )

    step: int = Field(..., ge=STEP_MIN, le=STEP_MAX, description="Hour since start (1-743)")
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    amount: float = Field(..., ge=0, le=MAX_MONEY)
    oldbalanceOrg: float = Field(..., ge=0, le=MAX_MONEY, description="Sender balance before")
    newbalanceOrig: float = Field(..., ge=0, le=MAX_MONEY, description="Sender balance after")
    oldbalanceDest: float = Field(..., ge=0, le=MAX_MONEY, description="Receiver balance before")
    newbalanceDest: float = Field(..., ge=0, le=MAX_MONEY, description="Receiver balance after")


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = fraud, 0 = legitimate")
    label: str
    fraud_probability: float = Field(..., ge=0, le=1)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
