"""Feature engineering. This file is self-contained on purpose: the saved model refers to
these classes by name, so the API and the Azure scoring script must be able to import it."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

MONEY_COLUMNS = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
ENGINEERED_MONEY = ["errorBalanceOrig", "errorBalanceDest", "amount_to_origin_balance"]
LOG_SCALED_COLUMNS = MONEY_COLUMNS + ENGINEERED_MONEY
FLAG_COLUMNS = ["is_night", "origin_emptied", "dest_zero_balances"]


def signed_log1p(values):
    """log(1 + |x|) keeping the sign. Tames the huge range of money values (0 to 90 million)."""
    values = np.asarray(values, dtype="float64")
    return np.sign(values) * np.log1p(np.abs(values))


class FraudFeatureBuilder(BaseEstimator, TransformerMixin):
    """Turns the 7 raw transaction fields into the model's input features.

    Engineered features and why they exist:
    - errorBalanceOrig: oldbalanceOrg - amount - newbalanceOrig. Should be 0 for a normal debit.
    - errorBalanceDest: oldbalanceDest + amount - newbalanceDest. Should be 0 for a normal credit.
    - amount_to_origin_balance: amount relative to the sender's balance (the dataset has no
      repeated accounts, so an 'account average' cannot be computed).
    - hour_of_day: (step - 1) % 24. Assumes step 1 is the first hour of a day (an assumption).
    - is_night: 1 when hour_of_day is 0-5.
    - origin_emptied: 1 when the sender's balance goes from positive to exactly 0.
    - dest_zero_balances: 1 when the receiver's balance is 0 before and after.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).reset_index(drop=True)
        out = pd.DataFrame(index=df.index)
        out["type"] = df["type"].astype(str)
        for col in MONEY_COLUMNS:
            out[col] = df[col].astype("float64")

        out["errorBalanceOrig"] = out["oldbalanceOrg"] - out["amount"] - out["newbalanceOrig"]
        out["errorBalanceDest"] = out["oldbalanceDest"] + out["amount"] - out["newbalanceDest"]
        out["amount_to_origin_balance"] = out["amount"] / (out["oldbalanceOrg"] + 1.0)

        hour = (df["step"].astype("int64") - 1) % 24
        out["hour_of_day"] = hour.astype("float64")
        out["is_night"] = (hour < 6).astype("int8")
        out["origin_emptied"] = (
            (out["oldbalanceOrg"] > 0) & (out["newbalanceOrig"] == 0)
        ).astype("int8")
        out["dest_zero_balances"] = (
            (out["oldbalanceDest"] == 0) & (out["newbalanceDest"] == 0)
        ).astype("int8")
        return out
