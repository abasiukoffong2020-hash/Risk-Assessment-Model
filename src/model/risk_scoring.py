# Calculate model risk score based on default probability

def calculate_risk_score(default_probability):
    risk_score = (1 - default_probability) * 100

    if default_probability < 0.20:
        risk_category = "Low Risk"
    elif default_probability < 0.40:
        risk_category = "Medium Risk"
    else:
        risk_category = "High Risk"

    return risk_score, risk_category