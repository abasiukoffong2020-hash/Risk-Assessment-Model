import sys
import os

# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, PROJECT_ROOT)

# IMPORT EXISTING INFERENCE FUNCTIONS
from src.serving.inference import (
    reference_data,
    customer_registry,
    identify_customer,
    lookup_customer,
    calculate_total_due,
    validate_applicant_input,
    validate_new_customer_loan,
    build_feature_row,
    predict_default_probability,
    assess_loan_risk,
    predict
)

# REFERENCE DATA LOADING TEST
print("\n" + "=" * 60)
print("REFERENCE DATA TEST")
print("=" * 60)
assert not reference_data.empty
print(f"✅ Reference dataset loaded: " f"{len(reference_data)} customers.")

# TOTAL DUE BUSINESS RULE TEST
print("\n" + "=" * 60)
print("TOTAL DUE BUSINESS RULE TEST")
print("=" * 60)
loan_amount = 5000
total_due = calculate_total_due(loan_amount)
print(f"Loan Amount: ₦{loan_amount:,.2f}")
print(f"Total Due:   ₦{total_due:,.2f}")
assert total_due == 6000
print("✅ Total due calculation passed.")

# APPLICATION IDENTITY → CUSTOMER ID TEST
print("\n" + "=" * 60)
print("APPLICATION IDENTITY → CUSTOMER ID TEST")
print("=" * 60)
application_user_id = "USER000001"
expected_customer_id = "8a2a81a74ce8c05d014cfb32a0da1049"
resolved_customer_id = identify_customer(application_user_id)
print(f"Application User ID : {application_user_id}")
print(f"Customer ID         : {resolved_customer_id}")
assert resolved_customer_id == expected_customer_id
print("✅ Application identity successfully resolved to customer ID.")

#EXISTING CUSTOMER LOOKUP TEST
print("\n" + "=" * 60)
print("EXISTING CUSTOMER LOOKUP TEST")
print("=" * 60)
existing_customer_id = "8a2a81a74ce8c05d014cfb32a0da1049"
customer_history = lookup_customer(existing_customer_id)
print(customer_history)
assert customer_history["existing_customer"] is True
assert customer_history["previous_default"] >= 0
assert customer_history["days_overdue"] >= 0
assert customer_history["no_of_previous_loans"] >= 0
expected_current_loan = (customer_history["no_of_previous_loans"] + 1)
assert (
    customer_history["current_loan_number"]
    == expected_current_loan
)
print("✅ Existing customer lookup passed.")
print(f"Previous Default: " f"{customer_history['previous_default']}")
print(f"Days Overdue: " f"{customer_history['days_overdue']}")
print(f"No. Previous Loans: " f"{customer_history['no_of_previous_loans']}")
print(f"Current Loan Number: " f"{customer_history['current_loan_number']}")

#EXISTING CUSTOMER COMPLETE PREDICTION TEST
print("\n" + "=" * 60)
print("EXISTING CUSTOMER PREDICTION TEST")
print("=" * 60)
existing_customer_result = predict(
    application_user_id="USER000001",
    age=32,
    loanamount=5000,
    termdays=30,
    bank_account_type="Savings",
    threshold=0.40
)
print("\nLoan Risk Assessment:")
print(f"Loan Amount: " f"₦{5000:,.2f}")
print(f"Total Due: " f"₦{existing_customer_result['totaldue']:,.2f}")
print(f"Default Probability: " f"{existing_customer_result['default_probability']:.2%}")
print(f"Risk Score: " f"{existing_customer_result['risk_score']:.2f} / 100")
print(f"Risk Category: " f"{existing_customer_result['risk_category']}")
print(f"Model Decision: " f"{existing_customer_result['model_decision']}")
print(f"Business Decision: " f"{existing_customer_result['business_decision']}")
print(f"Existing Customer: " f"{existing_customer_result['existing_customer']}")
print(f"Current Loan Number: " f"{existing_customer_result['current_loan_number']}")

# Verify total due
assert existing_customer_result["totaldue"] == 6000
# Verify customer was recognized
assert existing_customer_result["existing_customer"] is True
assert (existing_customer_result["application_user_id"] == "USER000001")
# Verify current loan number
assert (existing_customer_result["current_loan_number"] == customer_history["current_loan_number"])
# Verify probability is valid
assert 0 <= existing_customer_result["default_probability"] <= 1
# Verify risk score is valid
assert 0 <= existing_customer_result["risk_score"] <= 100
print("✅ Existing customer prediction passed.")


# NEW CUSTOMER IDENTITY TEST
print("\n" + "=" * 60)
print("NEW CUSTOMER IDENTITY TEST")
print("=" * 60)
unknown_application_user_id = "USER999999"
new_customer_customer_id = identify_customer(unknown_application_user_id)
print(f"Application User ID : " f"{unknown_application_user_id}")
print(f"Resolved Customer ID: " f"{new_customer_customer_id}")
assert new_customer_customer_id is None
print("✅ Unknown application identity correctly identified as new customer.")


# NEW CUSTOMER LOOKUP TEST
print("\n" + "=" * 60)
print("NEW CUSTOMER LOOKUP TEST")
print("=" * 60)
new_customer_history = lookup_customer(new_customer_customer_id)
print(new_customer_history)
assert new_customer_history["existing_customer"] is False
assert new_customer_history["previous_default"] == 0
assert new_customer_history["days_overdue"] == 0
assert new_customer_history["no_of_previous_loans"] == 0
expected_current_loan = (new_customer_history["no_of_previous_loans"] + 1)
assert (new_customer_history["current_loan_number"] == expected_current_loan)
assert new_customer_history["current_loan_number"] == 1
print("✅ New customer lookup passed.")
print(f"Previous Default: " f"{new_customer_history['previous_default']}")
print(f"Days Overdue: " f"{new_customer_history['days_overdue']}")
print(f"No. Previous Loans: " f"{new_customer_history['no_of_previous_loans']}")
print(f"Current Loan Number: " f"{new_customer_history['current_loan_number']}")

# NEW CUSTOMER ₦10,000 BUSINESS RULE
print("\n" + "=" * 60)
print("NEW CUSTOMER ₦10,000 LIMIT TEST")
print("=" * 60)
validate_new_customer_loan(loanamount=10000, existing_customer=False)
print("✅ ₦10,000 first loan is allowed.")

# NEW CUSTOMER ABOVE ₦10,000 LIMIT
print("\n" + "=" * 60)
print("NEW CUSTOMER ABOVE ₦10,000 LIMIT TEST")
print("=" * 60)
try:
    validate_new_customer_loan(loanamount=20000, existing_customer=False)
    raise AssertionError("❌ Business rule failed: " "₦20,000 should have been rejected.")
except ValueError as e:
    expected_message = ("New customers cannot apply for a first loan " "greater than ₦10,000.")
    assert str(e) == expected_message
    print("✅ Business rule correctly rejected ₦20,000.")
    print(f"Message: {e}")

# NEW CUSTOMER COMPLETE PREDICTION
print("\n" + "=" * 60)
print("NEW CUSTOMER PREDICTION TEST")
print("=" * 60)
new_customer_result = predict(
    application_user_id="USER999999",
    age=25,
    loanamount=5000,
    termdays=30,
    bank_account_type="Savings",
    threshold=0.40
)
print("\nLoan Risk Assessment:")
print(f"Loan Amount: " f"₦{5000:,.2f}")
print(f"Total Due: " f"₦{new_customer_result['totaldue']:,.2f}")
print(f"Default Probability: " f"{new_customer_result['default_probability']:.2%}")
print(f"Risk Score: " f"{new_customer_result['risk_score']:.2f} / 100")
print(f"Risk Category: " f"{new_customer_result['risk_category']}")
print(f"Model Decision: " f"{new_customer_result['model_decision']}")
print(f"Business Decision: " f"{new_customer_result['business_decision']}")
print(f"Existing Customer: " f"{new_customer_result['existing_customer']}")
print(f"Current Loan Number: " f"{new_customer_result['current_loan_number']}")
assert new_customer_result["existing_customer"] is False
assert new_customer_result["current_loan_number"] == 1
assert new_customer_result["totaldue"] == 6000
assert new_customer_result["business_decision"] == "Allowed"
assert 0 <= new_customer_result["default_probability"] <= 1
assert 0 <= new_customer_result["risk_score"] <= 100
print("✅ New customer prediction passed.")

# BANK ACCOUNT TYPE VALIDATION
print("\n" + "=" * 60)
print("BANK ACCOUNT TYPE VALIDATION TEST")
print("=" * 60)
for account_type in [
    "None",
    "Other",
    "Savings",
    "Current"
]:
    validate_applicant_input(        
        age=30,
        loanamount=50000,
        termdays=30,
        bank_account_type=account_type
    )
    print(
        f"✅ Bank account type accepted: "
        f"{account_type}"
    )

# FINAL MESSAGE
print("\n" + "=" * 60)
print("ALL INFERENCE TESTS PASSED")
print("=" * 60)