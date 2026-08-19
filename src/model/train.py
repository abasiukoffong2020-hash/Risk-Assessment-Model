import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score
)


def train_model(
    df: pd.DataFrame,
    target_col: str = "default",
    threshold: float = 0.40,
    test_size: float = 0.20
):
    """
    Trains the Logistic Regression model and logs training
    metrics and artifacts to the active MLflow run_pipeline.

    Args:
        df (pd.DataFrame):
            Feature-engineered loan dataset.

        target_col (str):
            Name of the target column.

    Returns:
        model:
            Trained Logistic Regression model.

        scaler:
            Fitted StandardScaler.

        X_test:
            Test feature data.

        y_test:
            Test target data.

        y_pred:
            Predictions generated using the selected threshold.
    """

    # ============================================================
    # SEPARATE FEATURES AND TARGET
    # ============================================================

    X = df.drop(
        columns=[
            target_col,
            "customerid",
            "approveddate",
            "creationdate"
        ]
    )

    y = df[target_col]

    # ============================================================
    # TRAIN / TEST SPLIT
    # ============================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    # ============================================================
    # FEATURE SCALING
    # ============================================================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ============================================================
    # MODEL CONFIGURATION
    # ============================================================

    model = LogisticRegression(
        random_state=42,
        class_weight="balanced",
        max_iter=1000
    )

    # ============================================================
    # MLflow EXPERIMENT
    # ============================================================ 
        
    # LOG MODEL PARAMETERS       

    mlflow.log_param(
        "model",
        "Logistic Regression"
    )

    mlflow.log_param(
        "random_state",
        42
    )

    mlflow.log_param(
        "class_weight",
        "balanced"
    )

    mlflow.log_param(
        "max_iter",
        1000
    )

    mlflow.log_param(
        "threshold",
        threshold
    )

    
    # TRAIN MODEL

    model.fit(
        X_train_scaled,
         y_train
    )

    
    # TOP 6 KEY FACTORS
    

    feature_coefficients = pd.DataFrame({
    "feature": X_train.columns,
    "coefficient": model.coef_[0]
    })

    # Use absolute coefficient magnitude to rank feature importance
    feature_coefficients["absolute_coefficient"] = (
    feature_coefficients["coefficient"].abs()
    )

    top_6_key_factors = (
        feature_coefficients
        .sort_values(
            by="absolute_coefficient",
            ascending=False
        )
        .head(6)["feature"]
        .tolist()
    )

    

    
    # PREDICTION       

    probabilities = model.predict_proba(
        X_test_scaled
    )[:, 1]

    y_pred = (
        probabilities >= threshold
    ).astype(int)

        
    # CALCULATE METRICS
        

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    roc_auc = roc_auc_score(y_test, probabilities)

        
    # LOG METRICS
        

    mlflow.log_metric("precision", precision)

    mlflow.log_metric("recall", recall)

    mlflow.log_metric("f1_score", f1)

    mlflow.log_metric("accuracy", accuracy)

    mlflow.log_metric("roc_auc", roc_auc)

        
    # LOG TRAINING DATASET
    

    train_ds = mlflow.data.from_pandas(df, source="training_data")

    mlflow.log_input(train_ds, context="training")

        
    # SAVE MODEL AND SCALER WITH JOBLIB
    

    joblib.dump(model, MODEL_DIR / "logistic_model.pkl")

    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

        
    # LOG MODEL TO MLflow
    

    mlflow.sklearn.log_model(model, artifact_path="model")

        
    # SAVE CLASSIFICATION REPORT AS ARTIFACT
    report = classification_report(y_test, y_pred,  digits=3)

    REPORT_PATH = ARTIFACT_DIR / "classification_report.txt"

    with open(REPORT_PATH, "w") as f:f.write(report)

    mlflow.log_artifact(REPORT_PATH)

    # TRAINING SUMMARY
        

    print(
        f"Model trained. "
        f"Accuracy: {accuracy:.4f}, "
        f"Recall: {recall:.4f}"
    )

    
    # FINAL EVALUATION OUTPUT
    

    print(
        classification_report(
            y_test,
            y_pred,
            digits=3
        )
    )

    return (
        model,
        scaler,
        X_test_scaled,
        y_test,
        y_pred,
        top_6_key_factors
    )
