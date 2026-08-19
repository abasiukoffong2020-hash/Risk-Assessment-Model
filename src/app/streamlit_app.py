"""
STREAMLIT LOAN RISK PREDICTION APPLICATION

Nigerian Loan Default Risk Predictor

The applicant-facing interface collects only:

- Age
- Loan amount
- Term days
- Bank account type

Customer identity is handled internally through a simulated
upstream identity context.

For demonstration purposes, the user can select:

- Existing customer
- New customer

The customer ID itself is never entered by the applicant.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import streamlit as st
import pandas as pd

PROFILE_IMAGE = (
    PROJECT_ROOT
    / "src"
    / "app"
    / "assests"
    / "Abasiuko.JPEG"
)

from src.serving.inference import predict
from src.upstream.upstream_identity import customer_registry


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Nigerian Loan Default Risk Predictor",
    page_icon="💰",
    layout="centered"
)

st.image(str(PROFILE_IMAGE), width=150)

st.markdown("### Abasiuko Edet Offong")
st.caption("AI & ML Engineer")
st.caption("Loan Default Risk Predictor")

st.divider()
# ============================================================
# APPLICATION TITLE
# ============================================================

st.title("Nigerian Loan Default Risk Predictor")

st.write(
    "Enter the loan application details below to receive "
    "an automated loan-risk assessment."
)


# ============================================================
# SIMULATED UPSTREAM IDENTITY CONTEXT
# ============================================================

st.subheader("Demo Identity Context")

st.info(
    "This section simulates an upstream customer-management "
    "system that has already identified the applicant. "
    "The applicant does not enter a customer ID."
)


applicant_type = st.radio(
    "Applicant type",
    options=[
        "Existing customer",
        "New customer"
    ],
    horizontal=True
)


# ============================================================
# EXISTING CUSTOMER IDENTITY CONTEXT
# ============================================================

customerid = None

if applicant_type == "Existing customer":

    # --------------------------------------------------------
    # Select one customer from the registry.
    #
    # This is a DEMO identity control, not an applicant
    # customer-ID input field.
    # --------------------------------------------------------

    selected_application_user_id = st.selectbox(
        "Demo customer",
        options=customer_registry["application_user_id"].tolist(),
        help=(
            "For demonstration purposes, select one of the "
            "registered customers. The internal customer ID "
            "is resolved automatically."
        )
    )

    # --------------------------------------------------------
    # Resolve the selected application identity internally.
    # --------------------------------------------------------

    selected_customer = customer_registry[
        customer_registry["application_user_id"]
        == selected_application_user_id
    ]

    # --------------------------------------------------------
    # Retrieve the internal customer ID.
    #
    # This value is NOT displayed to the applicant.
    # --------------------------------------------------------

    customerid = selected_customer.iloc[0]["customerid"]

else:

    # --------------------------------------------------------
    # New applicant has no existing customer identity.
    # --------------------------------------------------------

    customerid = None

    st.info(
        "New applicant selected. No existing customer history "
        "will be used."
    )


# ============================================================
# LOAN APPLICATION FORM
# ============================================================

st.subheader("Loan Application")

age = st.number_input(
    "Age",
    min_value=18.0,
    max_value=100.0,
    value=30.0,
    step=1.0
)


loanamount = st.number_input(
    "Loan amount (₦)",
    min_value=1.0,
    max_value=1_000_000.0,
    value=5_000.0,
    step=500.0
)


termdays = st.number_input(
    "Term days",
    min_value=1,
    max_value=365,
    value=30,
    step=1
)


bank_account_type = st.selectbox(
    "Bank account type",
    options=[
        "None",
        "Other",
        "Savings",
        "Current"
    ]
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "Assess Loan Risk",
    type="primary",
    use_container_width=True
):

    try:

        # ----------------------------------------------------
        # CALL THE EXISTING INFERENCE PIPELINE
        #
        # The applicant never supplies customerid.
        # Streamlit supplies it internally from the simulated
        # upstream identity context.
        # ----------------------------------------------------

        result = predict(
            customerid=customerid,
            age=age,
            loanamount=loanamount,
            termdays=termdays,
            bank_account_type=bank_account_type
        )


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.subheader("Loan Risk Assessment")


        # ----------------------------------------------------
        # PRIMARY RISK RESULTS
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Default Probability",
                f"{result['default_probability']:.2%}"
            )

        with col2:

            st.metric(
                "Risk Score",
                f"{result['risk_score']:.2f}%"
            )


        st.write(
            "**Risk Category:**",
            result["risk_category"]
        )

        st.write(
            "**Model Decision:**",
            result["model_decision"]
        )

        st.write(
            "**Business Decision:**",
            result["business_decision"]
        )


        # ====================================================
        # CUSTOMER INFORMATION
        # ====================================================

        st.subheader("Customer Information")

        st.write(
            "**Existing Customer:**",
            result["existing_customer"]
        )

        st.write(
            "**Current Loan Number:**",
            int(result["current_loan_number"])
        )


        # ====================================================
        # HISTORICAL CUSTOMER FEATURES
        # ====================================================

        st.subheader("Historical Customer Information")

        st.write(
            "**Previous Default:**",
            result["previous_default"]
        )

        st.write(
            "**Number of Previous Loans:**",
            result["no_of_previous_loans"]
        )

        st.write(
            "**Days Overdue:**",
            result["days_overdue"]
        )


        # ====================================================
        # CALCULATED LOAN INFORMATION
        # ====================================================

        st.subheader("Calculated Loan Information")

        st.write(
            "**Total Due:**",
            f"₦{result['totaldue']:,.2f}"
        )


    except ValueError as e:

        st.error(str(e))


    except Exception as e:

        st.error(
            "An unexpected error occurred while "
            "processing the loan application."
        )