"""
FASTAPI SERVING APPLICATION

Nigerian Loan Default Risk Predictor

The API receives only applicant-facing loan information.

Customer identity and historical customer lookup are not
required for the public prediction interface.

The inference layer internally creates the historical
features for a new applicant.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal

from src.serving.inference import predict
from src.upstream.upstream_identity import (get_current_customer_id)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Nigerian Loan Default Risk Predictor API",
    description=(
        "API for predicting Nigerian loan default risk "
        "and applying loan business rules."
    ),
    version="1.0.0"
)


# ============================================================
# HEALTH CHECK ENDPOINT
# ============================================================

@app.get("/")
def root():
    """
    Health-check endpoint.

    Confirms that the FastAPI application is running.
    """

    return {
        "status": "ok",
        "message": (
            "Nigerian Loan Default Risk Predictor API "
            "is running."
        )
    }


# ============================================================
# REQUEST DATA SCHEMA
# ============================================================

class LoanApplication(BaseModel):
    """
    Applicant-facing loan application.

    The applicant provides only the information required
    to apply for the loan.

    Customer identity is NOT requested.
    """

    age: float

    loanamount: float

    termdays: int

    bank_account_type: Literal[
        "None",
        "Other",
        "Savings",
        "Current"
    ]


# ============================================================
# MAIN PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def get_prediction(data: LoanApplication):
    """
    Receive the loan application and run the complete
    inference pipeline.

    The applicant does not provide:

        - customer ID
        - application user ID
        - previous default
        - days overdue
        - number of previous loans
        - current loan number
        - total due

    These values are handled internally by the inference
    pipeline for the new-applicant prediction scenario.
    """
    
    
    try:
        input_data = data.model_dump()
        customerid = get_current_customer_id()

        # ----------------------------------------------------
        # CALL THE COMPLETE INFERENCE PIPELINE
        # ----------------------------------------------------

        result = predict(
            customerid=customerid,
            age=data.age,
            loanamount=data.loanamount,
            termdays=data.termdays,
            bank_account_type=data.bank_account_type
        )

        # ----------------------------------------------------
        # RETURN COMPLETE RISK ASSESSMENT
        # ----------------------------------------------------

        return result

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "An internal error occurred while "
                "processing the loan application."
            )
        )