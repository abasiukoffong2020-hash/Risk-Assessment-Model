"""
SIMULATED UPSTREAM CUSTOMER IDENTITY SYSTEM

This module simulates an upstream system that has already
identified the current applicant before the loan-risk API
processes the application.

The applicant does NOT provide a customer ID.

The simulator uses the complete customer registry containing
all 4,368 registered customers.

For testing/demo purposes, the current applicant can be
selected externally without modifying this Python file.
"""

import os
import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# CUSTOMER REGISTRY PATH
# ============================================================

CUSTOMER_REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "lookup"
    / "customer_registry.csv"
)


# ============================================================
# CHECK CUSTOMER REGISTRY
# ============================================================

if not CUSTOMER_REGISTRY_PATH.exists():
    raise FileNotFoundError(
        f"Customer registry was not found at: "
        f"{CUSTOMER_REGISTRY_PATH}"
    )


# ============================================================
# LOAD CUSTOMER REGISTRY
# ============================================================

try:
    customer_registry = pd.read_csv(
        CUSTOMER_REGISTRY_PATH
    )

except Exception as e:
    raise RuntimeError(
        f"Failed to load customer registry: {e}"
    ) from e


# ============================================================
# VALIDATE CUSTOMER REGISTRY
# ============================================================

required_columns = [
    "application_user_id",
    "customerid"
]

missing_columns = [
    col
    for col in required_columns
    if col not in customer_registry.columns
]

if missing_columns:
    raise ValueError(
        "Customer registry is missing required columns: "
        f"{missing_columns}"
    )


# ============================================================
# VALIDATE REGISTRY CONTENT
# ============================================================

if customer_registry["application_user_id"].duplicated().any():
    raise ValueError(
        "Customer registry contains duplicate "
        "application user IDs."
    )


if customer_registry["customerid"].duplicated().any():
    raise ValueError(
        "Customer registry contains duplicate "
        "customer IDs."
    )


# ============================================================
# SIMULATED CURRENT APPLICANT
# ============================================================

"""
For demonstration purposes, the current applicant can be
selected through an environment variable.

Example:

Windows PowerShell:

$env:SIMULATED_APPLICATION_USER_ID="USER000001"

The applicant-facing API does NOT receive this value.

It represents identity information that an upstream
system has already established internally.
"""

SIMULATED_APPLICATION_USER_ID = os.getenv(
    "SIMULATED_APPLICATION_USER_ID"
)


# ============================================================
# UPSTREAM IDENTITY RESOLUTION
# ============================================================

def get_current_customer_id():
    """
    Simulate the upstream identity system identifying
    the current applicant.

    The applicant does not provide the identity.

    The simulator receives the selected application user ID
    internally and resolves it against the complete registry.

    Returns:
        customerid for an existing registered customer
        None for a new/unregistered applicant
    """

    # --------------------------------------------------------
    # NEW APPLICANT
    # --------------------------------------------------------

    if SIMULATED_APPLICATION_USER_ID == "NEW":
        return None


    # --------------------------------------------------------
    # NO SIMULATED ID PROVIDED
    # --------------------------------------------------------

    if (
        SIMULATED_APPLICATION_USER_ID is None
        or str(SIMULATED_APPLICATION_USER_ID).strip() == ""
    ):
        raise ValueError(
            "No simulated applicant identity was provided. "
            "Set SIMULATED_APPLICATION_USER_ID before "
            "starting the API."
        )


    # --------------------------------------------------------
    # EXISTING CUSTOMER LOOKUP
    # --------------------------------------------------------

    customer = customer_registry[
        customer_registry["application_user_id"]
        == SIMULATED_APPLICATION_USER_ID
    ]


    # --------------------------------------------------------
    # IDENTITY NOT FOUND
    # --------------------------------------------------------

    if customer.empty:
        raise ValueError(
            "The simulated application user ID was not "
            "found in the customer registry."
        )


    # --------------------------------------------------------
    # RETURN INTERNAL CUSTOMER ID
    # --------------------------------------------------------

    return customer.iloc[0]["customerid"]