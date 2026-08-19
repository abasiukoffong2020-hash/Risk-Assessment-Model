# Nigerian Loan Default Risk Predictor

## AI-04 — Loan Default Risk Predictor

A machine-learning solution for predicting the likelihood of loan default for Nigerian micro-lending and POS environments.
The project combines customer and loan data, historical repayment behaviour, feature engineering, a trained classification model, risk scoring, business rules, MLflow experiment tracking, and a Streamlit web application for manual loan-risk assessment.

---

## Purpose

Micro-lenders and POS agents need a fast and consistent way to assess repayment risk before approving a loan.
This project builds a machine-learning model that predicts the likelihood that a loan applicant will default.

The system is designed to:
- Accept applicant loan information.
- Retrieve historical customer information internally when available.
- Generate the features required by the trained model.
- Predict the probability of default.
- Convert the probability into a risk score and risk category.
- Apply a business decision based on the model result and loan rules.
- Present the result through a simple Streamlit web interface.
- Preserve model evaluation results and MLflow experiment artifacts for reproducibility.

---

# Problem Solved & Benefits

### Faster loan-risk decisions
The system provides an automated risk assessment instead of requiring a lender or POS agent to manually inspect historical repayment information.
### Data-driven lending
The model uses historical loan behaviour and applicant characteristics to estimate the probability of default.

### Customer-history awareness
Existing customers can be evaluated using historical repayment information
New customers are treated as having no historical loan record.

### Explainable risk output
The application does not return only a binary prediction. It provides:
- Default probability
- Risk score
- Risk category
- Model decision
- Business decision
- Customer history information

### Reproducible machine learning
Training outputs, preprocessing information, feature metadata, evaluation results, and MLflow experiment artifacts are preserved within the project.

### Practical deployment
The trained model is exposed through a Streamlit application so that users can test the system without opening the training notebook.
A FastAPI implementation is also included as the project's API serving layer.

---

# What I Built
Data & Modeling: Feature engineering + Logistic Regression; experiments logged to MLflow.
Model tracking: Runs, metrics, and the serialized model logged under a named MLflow experiment.
Inference service: FastAPI app exposing /predict (POST) and a root health check /.
Web UI: Streamlit interface for quick, accessible, and shareable manual loan-risk assessment.

 Data Preparation
The project uses Nigerian loan datasets containing current loan applications, previous loans, and customer demographic information.
The project also creates processed and lookup datasets used during inference.

Important data-processing tasks include:
- Date conversion
- Historical loan aggregation
- Customer-level feature creation
- Default identification
- Days-overdue calculation
- Customer history lookup
- Demographic feature integration
- Missing-value handling
- Categorical encoding


## dataset source: https://www.kaggle.com/competitions/data-science-nigeria-credit-risk-prediction/data