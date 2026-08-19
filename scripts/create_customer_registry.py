import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "lookup"
    / "trainperf_merged_reference.csv"
)

REGISTRY_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "lookup"
    / "customer_registry.csv"
)


# ============================================================
# LOAD REFERENCE DATA
# ============================================================

if not REFERENCE_DATA_PATH.exists():
    raise FileNotFoundError(
        f"Reference dataset was not found at: "
        f"{REFERENCE_DATA_PATH}"
    )

reference_data = pd.read_csv(REFERENCE_DATA_PATH)


# ============================================================
# VALIDATE CUSTOMER ID COLUMN
# ============================================================

if "customerid" not in reference_data.columns:
    raise ValueError(
        "Reference dataset does not contain "
        "'customerid' column."
    )


# ============================================================
# VALIDATE CUSTOMER UNIQUENESS
# ============================================================

if reference_data["customerid"].duplicated().any():
    raise ValueError(
        "Duplicate customer IDs detected. "
        "Registry generation stopped."
    )


# ============================================================
# CREATE SIMULATED APPLICATION IDENTITIES
# ============================================================

customer_registry = pd.DataFrame({
    "application_user_id": [
        f"USER{i:06d}"
        for i in range(1, len(reference_data) + 1)
    ],
    "customerid": reference_data["customerid"].astype(str)
})


# ============================================================
# SAVE CUSTOMER REGISTRY
# ============================================================

customer_registry.to_csv(
    REGISTRY_DATA_PATH,
    index=False
)


# ============================================================
# VALIDATION AFTER CREATION
# ============================================================

print("✅ Customer registry created successfully.")
print()
print(f"Registry path: {REGISTRY_DATA_PATH}")
print(f"Total customers: {len(customer_registry)}")
print(
    "Unique application identities:",
    customer_registry["application_user_id"].nunique()
)
print(
    "Unique customer IDs:",
    customer_registry["customerid"].nunique()
)
print(
    "Duplicate customer IDs:",
    customer_registry["customerid"].duplicated().sum()
)


# ============================================================
# CHECK YOUR SPECIFIC CUSTOMER
# ============================================================

target_customer_id = (
    "8a2a81a74ce8c05d014cfb32a0da1049"
)

target = customer_registry[
    customer_registry["customerid"] == target_customer_id
]

print()
print("Target customer mapping:")

if target.empty:
    print(
        "❌ Target customer ID was not found "
        "in the generated registry."
    )
else:
    print(target.to_string(index=False))


    


