import joblib
import json
import pandas as pd
from pathlib import Path
from src.model.risk_scoring import calculate_risk_score


# PROJECT PATHS
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_model.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "artifacts" / "feature_columns.json"

REFERENCE_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "lookup"
    / "trainperf_merged_reference.csv"
)

CUSTOMER_REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "lookup"
    / "customer_registry.csv"
)

# LOAD MODEL, SCALER AND REFERENCE DATA SAFELY, CHECK MODEL FILE
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Trained model was not found at: {MODEL_PATH}")

# CHECK SCALER FILE
if not SCALER_PATH.exists():
    raise FileNotFoundError(f"Scaler was not found at: {SCALER_PATH}")
    
# CHECK FEATURE METADATA FILE
if not FEATURE_COLUMNS_PATH.exists():
    raise FileNotFoundError(f"Feature columns file was not found at: " f"{FEATURE_COLUMNS_PATH}")

# CHECK REFERENCE DATA
if not REFERENCE_DATA_PATH.exists():
    raise FileNotFoundError(f"Reference dataset was not found at: " f"{REFERENCE_DATA_PATH}")

# CHECK CUSTOMER REGISTRY
if not CUSTOMER_REGISTRY_PATH.exists():
    raise FileNotFoundError(f"Customer registry was not found at: " f"{CUSTOMER_REGISTRY_PATH}")

# LOAD MODEL
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load trained model: {e}") from e

# LOAD SCALER
try:
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load scaler: {e}") from e

# LOAD REFERENCE DATA
try:
    reference_data = pd.read_csv(REFERENCE_DATA_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load reference dataset: {e}") from e

# LOAD CUSTOMER REGISTRY
try:
    customer_registry = pd.read_csv(CUSTOMER_REGISTRY_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load customer registry: {e}") from e
print("✅ Model loaded successfully.")
print("✅ Scaler loaded successfully.")
print("✅ Reference data loaded successfully.")
print("✅ Customer registry loaded successfully.")


# VALIDATE REFERENCE DATA
required_reference_columns = ["customerid", "previous_default", "days_overdue", "no_of_previous_loans"]
missing_reference_columns = [col for col in required_reference_columns if col not in reference_data.columns]
if missing_reference_columns:
    raise ValueError("Reference dataset is missing required columns: " f"{missing_reference_columns}")

# Each customer should have one reference record
if reference_data["customerid"].duplicated().any():
    raise ValueError("Reference dataset contains duplicate customer IDs. " "Expected one reference record per customer.")
print(f"✅ Reference data validated: " f"{len(reference_data)} unique customers.")

# VALIDATE CUSTOMER REGISTRY
required_registry_columns = ["application_user_id", "customerid"]
missing_registry_columns = [col for col in required_registry_columns if col not in customer_registry.columns]
if missing_registry_columns:
    raise ValueError("Customer registry is missing required columns: " f"{missing_registry_columns}")
if customer_registry["application_user_id"].duplicated().any():
    raise ValueError("Customer registry contains duplicate " "application user IDs.")
if customer_registry["customerid"].duplicated().any():
    raise ValueError("Customer registry contains duplicate " "customer IDs.")
print(f"✅ Customer registry validated: " f"{len(customer_registry)} users.")

# CUSTOMER HISTORY LOOKUP
def lookup_customer(customerid):
    # new customer
    if customerid is None:
        no_of_previous_loans = 0
        current_loan_number = (no_of_previous_loans + 1)
        return {
            "previous_default": 0,
            "days_overdue": 0,
            "no_of_previous_loans": no_of_previous_loans,
            "current_loan_number": current_loan_number,
            "existing_customer": False
        }
  

    
    # EXISTING CUSTOMER
    customer = reference_data[reference_data["customerid"] == customerid]    
    if not customer.empty:
        # Retrieve the customer's single reference record.        
        latest_customer = customer.iloc[0]

        #Look at this customer's historical record and retrieve their previous-default value.
        previous_default = float(
            latest_customer["previous_default"]
        )

        days_overdue = float(
            latest_customer["days_overdue"]
        )

        no_of_previous_loans = float(
            latest_customer["no_of_previous_loans"]
        )

        current_loan_number = (
            no_of_previous_loans + 1
        )

        return {
            "previous_default": previous_default,
            "days_overdue": days_overdue,
            "no_of_previous_loans": no_of_previous_loans,
            "current_loan_number": current_loan_number,
            "existing_customer": True
        }

    
    # NEW CUSTOMER i.e customer id not found in historical data
    no_of_previous_loans = 0
    current_loan_number = (no_of_previous_loans + 1)  
    return {
        "previous_default": 0,
        "days_overdue": 0,
        "no_of_previous_loans": no_of_previous_loans,
        "current_loan_number": current_loan_number,
        "existing_customer": False
    }


# LOAD TRAINING FEATURE ORDER
try:
    with open(
        FEATURE_COLUMNS_PATH,
        "r"
    ) as f:
        feature_columns = json.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load feature_columns.json: {e}") from e

# Ensure the feature metadata is a list
if not isinstance(feature_columns, list):
    raise ValueError("feature_columns.json must contain a list " "of feature names.")

# Ensure the model expects the correct number of features
if len(feature_columns) != 11:
    raise ValueError(
        f"Expected 11 model features, "
        f"but feature_columns.json contains "
        f"{len(feature_columns)}."
    )
print(f"✅ Loaded {len(feature_columns)} " f"model feature columns.")


# APPLICATION IDENTITY → CUSTOMER ID
def identify_customer(application_user_id):
    """
    Resolve the application user's identity to the internal customer ID.
    In this project, customer_registry.csv simulates the upstream customer-identification system.
    Existing customer: application_user_id → customerid
    Unknown application identity: returns None so the inference pipeline can simulate a new customer.
    """
    if (application_user_id is None or str(application_user_id).strip() == ""):
        raise ValueError("Application user ID cannot be empty.")
    customer = customer_registry[customer_registry["application_user_id"] == application_user_id]

    # EXISTING CUSTOMER
    if not customer.empty: return customer.iloc[0]["customerid"]
    # NEW CUSTOMER: In this simulation, an application identity that  does not yet have a historical customer record  represents a new customer.
    return None


# INPUT VALIDATION
def validate_applicant_input(age, loanamount, termdays, bank_account_type):
    """
    Validate the applicant information before running the prediction pipeline.
    This prevents invalid data from reaching the model.
    """
    
    # AGE
    try:
        age = float(age)
    except (TypeError, ValueError):
        raise ValueError("Age must be a valid number.")

    if age <= 0:
        raise ValueError("Age must be greater than 0.")

    # LOAN AMOUNT
    try:
        loanamount = float(loanamount)
    except (TypeError, ValueError):
        raise ValueError("Loan amount must be a valid number.")
    if loanamount <= 0:
        raise ValueError(
            "Loan amount must be greater than ₦0."
        )

    # TERM DAYS
    try:
        termdays = int(termdays)
    except (TypeError, ValueError):
        raise ValueError(
            "Term days must be a valid whole number."
        )

    if termdays <= 0:
        raise ValueError(
            "Term days must be greater than 0."
        )

    # BANK ACCOUNT TYPE
    valid_account_types = {"None", "Other", "Savings", "Current"}
    if bank_account_type not in valid_account_types:
        raise ValueError(
            "Invalid bank account type. "
            "Choose None, Other, Savings, or Current."
        )


# NEW CUSTOMER BUSINESS RULE
def validate_new_customer_loan(
    loanamount,
    existing_customer
):
    """
    Apply the first-loan limit for new customers.
    New customers cannot receive a first loan
    greater than ₦10,000.
    """

    # Existing customers are not subject
    # to the new-customer first-loan limit.
    if existing_customer:
        return

    # Apply the ₦10,000 first-loan limit.
    if loanamount > 10000:
        raise ValueError("New customers cannot apply for a first loan " "greater than ₦10,000.")


# BUSINESS RULE 2: CALCULATE TOTAL DUE
def calculate_total_due(loanamount):
    """
    Calculate the amount the applicant must repay.
    Business rule:
        totaldue = loanamount × 1.20
    This represents a 20% charge on the original
    loan amount.
    """

    totaldue = loanamount * 1.20
    return totaldue 

#Build the complete applicant feature row
def build_feature_row(
    age,
    loanamount,
    totaldue,
    termdays,
    bank_account_type,
    previous_default,
    no_of_previous_loans,
    days_overdue,
    current_loan_number
):
    """
    Construct the complete 11-feature input
    expected by the trained model.
    """

    feature_row = {
        "current_loan_number": current_loan_number,
        "loanamount": loanamount,
        "totaldue": totaldue,
        "termdays": termdays,
        "previous_default": previous_default,
        "no_of_previous_loans": no_of_previous_loans,
        "days_overdue": days_overdue,
        "age": age,

        "bank_account_type_None": 0,
        "bank_account_type_Other": 0,
        "bank_account_type_Savings": 0
    }

    #reproduce the same one hot encoding used during model training
    if bank_account_type == "None":
        feature_row["bank_account_type_None"] = 1

    elif bank_account_type == "Other":
        feature_row["bank_account_type_Other"] = 1

    elif bank_account_type == "Savings":
        feature_row["bank_account_type_Savings"] = 1

    elif bank_account_type == "Current":
        pass

    else:
        raise ValueError(
            f"Invalid bank account type:"
            f" {bank_account_type}"
        )

    #Convert the dictionary into a DataFrame
    feature_df = pd.DataFrame([feature_row])
    #Enforce the exact training order
    feature_df = feature_df[feature_columns]

    if feature_df.columns.tolist() != feature_columns:
        raise ValueError(
            "❌ Serving features do not match "
            "the training feature order."
        )

    # Verify there are no missing values
    if feature_df.isnull().any().any():
        raise ValueError("❌ Applicant feature row contains " "missing values.")
    return feature_df


# MODEL PREDICTION
def predict_default_probability(feature_df):
    """
    Scale the applicant's features and generate
    the probability of loan default.
    """
    # VERIFY FEATURE COUNT
    if feature_df.shape[1] != len(feature_columns):
        raise ValueError(f"Model expected {len(feature_columns)} features, " f"but received {feature_df.shape[1]}.")

    # SCALE FEATURES
    try:
        scaled_features = scaler.transform(feature_df)
    except Exception as e:
        raise RuntimeError(
            f"Feature scaling failed: {e}"
        ) from e

    # GENERATE DEFAULT PROBABILITY
    try:
        probabilities = model.predict_proba(scaled_features)
    except Exception as e:
        raise RuntimeError(
            f"Model prediction failed: {e}"
        ) from e

    # Class 1 = Default
    default_probability = probabilities[0, 1]

    # SAFETY CHECK
    if not 0 <= default_probability <= 1:
        raise ValueError("Model returned an invalid default probability.")
    return float(default_probability)


# RISK SCORE + RISK CATEGORY + DECISIONS
def assess_loan_risk(default_probability, current_loan_number, loanamount, threshold=0.40):
    """
    Convert the model's default probability into:

    1. Risk score
    2. Risk category
    3. Model classification
    4. Business decision

    The ML model is NOT changed.

    Special business-rule handling:
        - Current loan number == 1 means first-time/new customer.
        - For a new customer, the ₦10,000 first-loan rule determines business eligibility.
        - The ML risk assessment is still calculated and returned.
    """

    
    # Calculate the risk score and risk category
    # This calls the existing calculate_risk_score() function. 
    risk_score, risk_category = calculate_risk_score(default_probability)

    
    # Preserve the existing ML model classification it is still based entirely on the trained model,
    # probability and the 0.40 classification threshold.
    if default_probability >= threshold:
        model_decision = "Default"
    else:
        model_decision = "No Default"

    
    # Determine the business decision:  A current loan number of 1 means this is a customer's first loan application.
    if current_loan_number == 1:

        # New-customer business rule:
        # First loan up to and including ₦10,000 is allowed.
        if loanamount <= 10000:
            business_decision = "Allowed"

        else:
            # This had been caught earlier by
            # validate_new_customer_loan(), but keep this
            # check here as an additional safety layer.
            business_decision = "Rejected"

    else:

        # Existing customers are not subject to the
        # ₦10,000 first-loan restriction.
        #
        # Therefore, their business decision follows
        # the existing ML classification.
        if model_decision == "Default":
            business_decision = "Rejected"
        else:
            business_decision = "Allowed"

    
    # Return the complete risk assessment
    
    return {
        "default_probability": default_probability,
        "risk_score": risk_score,
        "risk_category": risk_category,
        "model_decision": model_decision,
        "business_decision": business_decision
    }


# COMPLETE LOAN RISK PREDICTION
def predict(
    customerid,        
    age,
    loanamount,
    termdays,
    bank_account_type,
    threshold=0.40
):
    """
    Run the complete loan risk assessment.

    The function:
    validate applicant input
    Looks up the customer's history.
    Applies the new-customer loan business rule.
    Builds the complete 11-feature model input.
    Scales the features.
    Generates the default probability.
    Calculates the risk score and category.
    Applies the classification threshold.
    """
    
    #validate applicant input
    validate_applicant_input(age=age, loanamount=loanamount, termdays=termdays, bank_account_type=bank_account_type)

    # lookup customer history
    customer_history = lookup_customer(customerid)
    
    # APPLY NEW-CUSTOMER BUSINESS RULE   
    validate_new_customer_loan(loanamount=loanamount, existing_customer=customer_history["existing_customer"])


    #  CALCULATE TOTAL DUE:The applicant does not enter totaldue. The system calculates it automatically using
    # the 20% charge business rule.
    totaldue = calculate_total_due(loanamount)

    
    #  BUILD COMPLETE 11-FEATURE INPUT
    feature_df = build_feature_row(
        age=age,
        loanamount=loanamount,
        totaldue=totaldue,
        termdays=termdays,
        bank_account_type=bank_account_type,
        previous_default=customer_history["previous_default"],
        no_of_previous_loans=customer_history["no_of_previous_loans"],
        days_overdue=customer_history["days_overdue"],
        current_loan_number=customer_history["current_loan_number"]
    )


    # GENERATE DEFAULT PROBABILITY
    default_probability = predict_default_probability(feature_df)

    # CALCULATE RISK ASSESSMENT
    risk_assessment = assess_loan_risk(
        default_probability=default_probability,
        current_loan_number=customer_history["current_loan_number"],
        loanamount=loanamount,
        threshold=threshold
    )

    #  ADD CUSTOMER INFORMATION
    risk_assessment["existing_customer"] = (customer_history["existing_customer"])
    risk_assessment["current_loan_number"] = (customer_history["current_loan_number"])
    # Display historical information retrieved by lookup_customer()
    risk_assessment["previous_default"] = (customer_history["previous_default"])
    risk_assessment["no_of_previous_loans"] = (customer_history["no_of_previous_loans"])
    risk_assessment["days_overdue"] = (customer_history["days_overdue"])
    # Add the automatically calculated repayment amount
    risk_assessment["totaldue"] = totaldue
    

    return risk_assessment