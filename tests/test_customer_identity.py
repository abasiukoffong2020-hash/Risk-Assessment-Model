# this test answers Can my simulated upstream identification system correctly resolve an application user 
# to the internal customer ID and retrieve that customer's historical information?"
import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "lookup"
    / "customer_registry.csv"
)

REFERENCE_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "lookup"
    / "trainperf_merged_reference.csv"
)


# ============================================================
# TEST APPLICATION USER
# ============================================================

application_user_id = "USER000001"


# ============================================================
# LOAD CUSTOMER REGISTRY
# ============================================================

registry = pd.read_csv(REGISTRY_PATH)


# ============================================================
# STEP 1 — APPLICATION ID → CUSTOMER ID
# ============================================================

customer = registry[
    registry["application_user_id"] == application_user_id
]

if customer.empty:
    raise ValueError(
        f"Application user '{application_user_id}' "
        "was not found in the customer registry."
    )

customerid = customer.iloc[0]["customerid"]

print("=" * 60)
print("STEP 1: APPLICATION ID → CUSTOMER ID")
print("=" * 60)

print(f"Application user ID : {application_user_id}")
print(f"Customer ID         : {customerid}")


# ============================================================
# LOAD REFERENCE DATA
# ============================================================

reference_data = pd.read_csv(
    REFERENCE_DATA_PATH
)


# ============================================================
# STEP 2 — CUSTOMER ID → HISTORICAL FEATURES
# ============================================================

customer_history = reference_data[
    reference_data["customerid"] == customerid
]

if customer_history.empty:
    raise ValueError(
        f"Customer ID '{customerid}' was not found "
        "in the reference dataset."
    )

if len(customer_history) > 1:
    raise ValueError(
        f"Customer ID '{customerid}' has multiple "
        "reference records."
    )

customer_record = customer_history.iloc[0]


# ============================================================
# RETRIEVE HISTORICAL FEATURES
# ============================================================

previous_default = float(
    customer_record["previous_default"]
)

days_overdue = float(
    customer_record["days_overdue"]
)

no_of_previous_loans = float(
    customer_record["no_of_previous_loans"]
)


# ============================================================
# DISPLAY COMPLETE CHAIN
# ============================================================

print()
print("=" * 60)
print("STEP 2: CUSTOMER ID → HISTORICAL FEATURES")
print("=" * 60)

print(f"Customer ID              : {customerid}")
print(f"Previous default         : {previous_default}")
print(f"Days overdue             : {days_overdue}")
print(f"No. previous loans      : {no_of_previous_loans}")


# ============================================================
# FINAL VERIFICATION
# ============================================================

print()
print("=" * 60)
print("FINAL VERIFICATION")
print("=" * 60)

print(
    "Application identity successfully resolved "
    "to customer ID."
)

print(
    "Customer ID successfully resolved "
    "to historical features."
)

print("✅ Complete identity → history lookup succeeded.")