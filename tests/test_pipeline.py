import sys
from pathlib import Path

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Add src to Python path
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))


# ============================================================
# IMPORT PROJECT FUNCTIONS
# ============================================================

from src.data.data_loader import load_data
from src.data.great_expectation.data_quality import validate_loan_data
from src.data.preprocessing import merge_datasets, preprocess_data
from src.features.features import build_features
from src.model.train import train_model
from src.model.evaluate import evaluate_model


# ============================================================
# DATA PATHS
# ============================================================

TRAINPERF_PATH = PROJECT_ROOT / "data" / "trainperf.csv"
CUSTOMER_PREV_PATH = PROJECT_ROOT / "data" / "customer_prevcleaned.csv"
DEMOGRAPHICS_PATH = PROJECT_ROOT / "data" / "traindemographics.csv"


# ============================================================
# EXPECTED MODEL FEATURES
# ============================================================

EXPECTED_FEATURES = [
    "current_loan_number",
    "loanamount",
    "totaldue",
    "termdays",
    "previous_default",
    "no_of_previous_loans",
    "days_overdue",
    "age",
    "bank_account_type_None",
    "bank_account_type_Other",
    "bank_account_type_Savings"
]


# ============================================================
# INTEGRATION TEST
# ============================================================

def test_pipeline():

    print("\n" + "=" * 60)
    print("STARTING INTEGRATION TEST")
    print("=" * 60)


    # ========================================================
    # STEP 1: LOAD DATA
    # ========================================================

    print("\n[1/7] Testing data loading...")

    trainperf = load_data(TRAINPERF_PATH)
    customer_prevcleaned = load_data(CUSTOMER_PREV_PATH)
    traindemographics = load_data(DEMOGRAPHICS_PATH)

    assert not trainperf.empty
    assert not customer_prevcleaned.empty
    assert not traindemographics.empty

    print("   ✅ Data loading passed")


    # ========================================================
    # STEP 2: MERGE DATASETS
    # ========================================================

    print("\n[2/7] Testing dataset merging...")

    merged_data = merge_datasets(
        trainperf,
        customer_prevcleaned,
        traindemographics
    )

    assert not merged_data.empty
    assert "customerid" in merged_data.columns
    assert "bank_account_type" in merged_data.columns
    assert "previous_default" in merged_data.columns

    print(
        f"   ✅ Dataset merging passed "
        f"({merged_data.shape[0]} rows, "
        f"{merged_data.shape[1]} columns)"
    )


    # ========================================================
    # STEP 3: PREPROCESS DATA
    # ========================================================

    print("\n[3/7] Testing preprocessing...")

    preprocessed_data = preprocess_data(
        merged_data
    )

    # Required preprocessing outputs
    assert "default" in preprocessed_data.columns
    assert "age" in preprocessed_data.columns
    assert "bank_account_type" in preprocessed_data.columns

    # Target must be binary
    assert set(
        preprocessed_data["default"].dropna().unique()
    ).issubset({0, 1})

    # Missing categorical values should have been replaced
    assert preprocessed_data["bank_account_type"].isna().sum() == 0

    # Numerical missing values should have been filled
    numeric_columns = [
        "previous_default",
        "no_of_previous_loans",
        "days_overdue",
        "age"
    ]

    for column in numeric_columns:
        assert preprocessed_data[column].isna().sum() == 0

    print("   ✅ Preprocessing passed")


    # ========================================================
    # STEP 4: FEATURE ENGINEERING
    # ========================================================

    print("\n[4/7] Testing feature engineering...")

    feature_data = build_features(
        preprocessed_data
    )

    # Check expected encoded columns
    for column in [
        "bank_account_type_None",
        "bank_account_type_Other",
        "bank_account_type_Savings"
    ]:
        assert column in feature_data.columns

    # Check that original categorical column is gone
    assert "bank_account_type" not in feature_data.columns

    # Check Boolean dummy columns were converted to integers
    for column in [
        "bank_account_type_None",
        "bank_account_type_Other",
        "bank_account_type_Savings"
    ]:
        assert feature_data[column].dtype == "int64"

    # Check that all expected model predictors exist
    for column in EXPECTED_FEATURES:
        assert column in feature_data.columns

    print("   ✅ Feature engineering passed")


    # ========================================================
    # STEP 5: DATA QUALITY VALIDATION
    # ========================================================

    print("\n[5/7] Testing data quality validation...")

    validation_passed, failed_checks = validate_loan_data(
        feature_data
    )

    assert validation_passed is True
    assert failed_checks == []

    print("   ✅ Data quality validation passed")


    # ========================================================
    # STEP 6: MODEL TRAINING
    # ========================================================

    print("\n[6/7] Testing model training...")

    model, scaler, X_test, y_test, y_pred = train_model(
        feature_data,
        target_col="default"
    )

    assert model is not None
    assert scaler is not None
    assert X_test is not None
    assert y_test is not None
    assert y_pred is not None

    print("   ✅ Model training passed")


    # ========================================================
    # STEP 7: MODEL EVALUATION
    # ========================================================

    print("\n[7/7] Testing model evaluation...")

    # X_test returned by train_model() is already scaled
    evaluate_model(
        model,
        X_test,
        y_test,
        threshold=0.40
    )

    print("   ✅ Model evaluation passed")


    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n" + "=" * 60)
    print("✅ INTEGRATION TEST PASSED")
    print("=" * 60)

    print("\nComplete pipeline verified:")
    print("   Data loading")
    print("        ↓")
    print("   Dataset merging")
    print("        ↓")
    print("   Preprocessing")
    print("        ↓")
    print("   Feature engineering")
    print("        ↓")
    print("   Data quality validation")
    print("        ↓")
    print("   Model training")
    print("        ↓")
    print("   Model evaluation")


# ============================================================
# RUN TEST
# ============================================================

if __name__ == "__main__":
    test_pipeline()