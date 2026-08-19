import great_expectations as ge
from typing import Tuple, List


def validate_loan_data(df) -> Tuple[bool, List[str]]:
    """
    Validate the cleaned Nigerian loan default dataset
    using Great Expectations.

    The validation checks:
    - Required columns
    - Missing values
    - Business rules
    - Numeric ranges
    - Data consistency
    """

    print("🔍 Starting data validation with Great Expectations...")

    # Convert pandas DataFrame to Great Expectations Dataset
    ge_df = ge.dataset.PandasDataset(df)

    # ============================================================
    # SCHEMA VALIDATION - ESSENTIAL COLUMNS
    # ============================================================

    print("   📋 Validating schema and required columns...")

    # Customer identifier
    ge_df.expect_column_to_exist("customerid")
    ge_df.expect_column_values_to_not_be_null("customerid")

    # Loan information
    ge_df.expect_column_to_exist("current_loan_number")
    ge_df.expect_column_to_exist("loanamount")
    ge_df.expect_column_to_exist("totaldue")
    ge_df.expect_column_to_exist("termdays")

    # Historical loan information
    ge_df.expect_column_to_exist("previous_default")
    ge_df.expect_column_to_exist("no_of_previous_loans")
    ge_df.expect_column_to_exist("days_overdue")

    # Customer demographic information
    ge_df.expect_column_to_exist("age")

    # Encoded bank account information
    ge_df.expect_column_to_exist("bank_account_type_None")
    ge_df.expect_column_to_exist("bank_account_type_Other")
    ge_df.expect_column_to_exist("bank_account_type_Savings")

    # Target variable
    ge_df.expect_column_to_exist("default")
    ge_df.expect_column_values_to_not_be_null("default")

    # ============================================================
    # BUSINESS LOGIC VALIDATION
    # ============================================================

    print("   💼 Validating business logic constraints...")

    # Default must be binary:
    # 0 = no default
    # 1 = default
    ge_df.expect_column_values_to_be_in_set(
        "default",
        [0, 1]
    )

    # Previous default must also be binary
    ge_df.expect_column_values_to_be_in_set(
        "previous_default",
        [0, 1]
    )

    # Loan amount cannot be negative
    ge_df.expect_column_values_to_be_between(
        "loanamount",
        min_value=0
    )

    # Total amount due cannot be negative
    ge_df.expect_column_values_to_be_between(
        "totaldue",
        min_value=0
    )

    # Loan term cannot be negative
    ge_df.expect_column_values_to_be_between(
        "termdays",
        min_value=0
    )

    # Number of previous loans cannot be negative
    ge_df.expect_column_values_to_be_between(
        "no_of_previous_loans",
        min_value=0
    )

    # Days overdue cannot be negative
    ge_df.expect_column_values_to_be_between(
        "days_overdue",
        min_value=0
    )

    # Age cannot be negative
    ge_df.expect_column_values_to_be_between(
        "age",
        min_value=0
    )

    # ============================================================
    # NUMERIC RANGE VALIDATION
    # ============================================================

    print("   📊 Validating numeric ranges...")

    # Loan amount should be greater than or equal to zero
    ge_df.expect_column_values_to_be_between(
        "loanamount",
        min_value=0
    )

    # Total due should be greater than or equal to zero
    ge_df.expect_column_values_to_be_between(
        "totaldue",
        min_value=0
    )

    # Loan term should be greater than or equal to zero
    ge_df.expect_column_values_to_be_between(
        "termdays",
        min_value=0
    )

    # Age should be greater than or equal to zero
    ge_df.expect_column_values_to_be_between(
        "age",
        min_value=0
    )

    # Days overdue should be greater than or equal to zero
    ge_df.expect_column_values_to_be_between(
        "days_overdue",
        min_value=0
    )

    # ============================================================
    # MISSING VALUE VALIDATION
    # ============================================================

    print("   🧹 Validating missing values...")

    ge_df.expect_column_values_to_not_be_null("loanamount")
    ge_df.expect_column_values_to_not_be_null("totaldue")
    ge_df.expect_column_values_to_not_be_null("termdays")
    ge_df.expect_column_values_to_not_be_null("previous_default")
    ge_df.expect_column_values_to_not_be_null("no_of_previous_loans")
    ge_df.expect_column_values_to_not_be_null("days_overdue")
    ge_df.expect_column_values_to_not_be_null("age")

    # ============================================================
    # DATA CONSISTENCY CHECKS
    # ============================================================

    print("   🔄 Validating data consistency...")

    # Total amount due must be greater than or equal to
    # the original loan amount.
    ge_df.expect_column_pair_values_A_to_be_greater_than_B(
        column_A="totaldue",
        column_B="loanamount",
        or_equal=True
    )
    
   

    # ============================================================
    # RUN VALIDATION SUITE
    # ============================================================

    print("   ⚙️ Running complete validation suite...")

    results = ge_df.validate()

    # ============================================================
    # PROCESS RESULTS
    # ============================================================

    failed_expectations = []

    for result in results["results"]:

        if not result["success"]:

            expectation_type = (
                result["expectation_config"]["expectation_type"]
            )

            failed_expectations.append(expectation_type)

    # ============================================================
    # VALIDATION SUMMARY
    # ============================================================

    total_checks = len(results["results"])

    passed_checks = sum(
        1
        for result in results["results"]
        if result["success"]
    )

    failed_checks = total_checks - passed_checks

    if results["success"]:

        print(
            f"✅ Data validation PASSED: "
            f"{passed_checks}/{total_checks} checks successful"
        )

    else:

        print(
            f"❌ Data validation FAILED: "
            f"{failed_checks}/{total_checks} checks failed"
        )

        print(
            f"   Failed expectations: "
            f"{failed_expectations}"
        )

    return results["success"], failed_expectations
