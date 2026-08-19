#!/usr/bin/env python3
"""
Runs the Nigerian loan default prediction pipeline sequentially:
load → merge → preprocess → feature engineering →
validate → train → evaluate
"""

import os
import sys
import time
import argparse
import json
import joblib
from matplotlib import cm
import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.data.data_loader import load_data
from src.data.preprocessing import merge_datasets, preprocess_data
from src.features.features import build_features
from src.data.great_expectation.data_quality import validate_loan_data
from src.model.train import train_model
from sklearn.metrics import (classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score)
from src.model.risk_scoring import calculate_risk_score

def main(args):
    """
    Main training pipeline function that orchestrates the complete
    Nigerian loan default prediction workflow.
    """

    # === MLflow Setup - ESSENTIAL for experiment tracking ===
    # Configure MLflow to use local file-based tracking
    # unless a custom tracking URI is supplied.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mlflow_db_path = os.path.join(project_root, "mlflow.db")
    mlflow_uri = args.mlflow_uri or f"sqlite:///{mlflow_db_path.replace(os.sep, '/')}"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run():

        mlflow.log_param("model", "Logistic Regression")
        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("test_size", args.test_size)


        # ============================================================
        # STAGE 1: DATA LOADING
        # ============================================================

        print("🔄 Loading loan datasets...")

        trainperf = load_data(args.trainperf)
        customer_prevcleaned = load_data(args.customer_prevcleaned)
        traindemographics = load_data(args.demographics)

        print(f"✅ trainperf loaded: "f"{trainperf.shape[0]} rows, "f"{trainperf.shape[1]} columns")

        print(
            f"✅ customer_prevcleaned loaded: "
            f"{customer_prevcleaned.shape[0]} rows, "
            f"{customer_prevcleaned.shape[1]} columns"
        )

        print(
            f"✅ traindemographics loaded: "
            f"{traindemographics.shape[0]} rows, "
            f"{traindemographics.shape[1]} columns"
        )


        # ============================================================
        # STAGE 2: DATASET MERGING
        # ============================================================

        print("🔄 Merging loan datasets...")

        merged_data = merge_datasets(
            trainperf,
            customer_prevcleaned,
            traindemographics
        )

        if merged_data.empty:

            raise ValueError(
                "❌ Dataset merging produced an empty dataframe."
            )

        print(
            f"✅ Datasets merged: "
            f"{merged_data.shape[0]} rows, "
            f"{merged_data.shape[1]} columns"
        )

        # ============================================================
        # STAGE 3: PREPROCESSING
        # ============================================================

        print("🔄 Preprocessing loan data...")
        preprocessed_data = preprocess_data(merged_data)

        if preprocessed_data.empty:

            raise ValueError(
                "❌ Preprocessing produced an empty dataframe."
            )

        print(
            f"✅ Preprocessing completed: "
            f"{preprocessed_data.shape[0]} rows, "
            f"{preprocessed_data.shape[1]} columns"
        )
        # ------------------------------------------------------------
        # Save processed dataset for reproducibility and debugging
        # ------------------------------------------------------------
        processed_path = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "loan_processed.csv"
        )
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        preprocessed_data.to_csv(processed_path, index=False)
        print(
            f"💾 Processed dataset saved to "
            f"{processed_path} | "
            f"Shape: {preprocessed_data.shape}"
        )


        # ============================================================
        # STAGE 4: FEATURE ENGINEERING
        # ============================================================

        print("🔄 Building model features...")

        # Confirm target exists before feature engineering
        if "default" not in preprocessed_data.columns:

            raise ValueError(
                "❌ Target column 'default' "
                "not found after preprocessing."
            )

        feature_data = build_features(preprocessed_data, target_col="default")

        if feature_data.empty:

            raise ValueError(
                "❌ Feature engineering produced "
                "an empty dataframe."
            )

        # Ensure no Boolean columns reach the model
        bool_columns = feature_data.select_dtypes(
            include=["bool"]
        ).columns.tolist()

        if bool_columns:

            feature_data[bool_columns] = (feature_data[bool_columns].astype(int))

            print(
                f"🔄 Converted remaining Boolean columns "
                f"to integers: {bool_columns}"
            )

        print(
            f"✅ Feature engineering completed: "
            f"{feature_data.shape[0]} rows, "
            f"{feature_data.shape[1]} columns"
        )        
                  
        
        # ============================================================
        # CRITICAL: DATA QUALITY VALIDATION
        # ============================================================

        print(
            "🔍 Validating loan data quality "
            "with Great Expectations..."
        )

        is_valid, failed = validate_loan_data(feature_data)

        # Track data-quality status in MLflow
        mlflow.log_metric("data_quality_pass", int(is_valid))
        if not is_valid:

            # Log validation failures to MLflow
            mlflow.log_text(
                json.dumps(
                    failed,
                    indent=2
                ),
                artifact_file="failed_expectations.json"
            )

            raise ValueError(
                f"❌ Data quality validation failed. "
                f"Issues: {failed}"
            )

        else:
            print("✅ Data validation passed. Logged to MLflow.")

        # ============================================================
        # CRITICAL: SAVE FEATURE METADATA FOR SERVING CONSISTENCY
        # ============================================================
        # This ensures that future predictions use the exact same
        # features in the exact same order as the training process.

    

        artifacts_dir = PROJECT_ROOT / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        # ------------------------------------------------------------
        # Get the exact feature columns used by train_model()
        # ------------------------------------------------------------
        # train_model() removes:
        #
        #    default       -> target
        #    customerid    -> identifier, not a model feature
        #    approveddate  -> date field not used by the model
        #    creationdate  -> date field not used by the model
        #
        # Therefore, the remaining columns are the exact predictors
        # supplied to Logistic Regression.
        feature_cols = feature_data.drop(columns=["default", "customerid", "approveddate", "creationdate"]).columns.tolist()
        
        # Save feature columns locally for development/serving 
        feature_columns_path = (artifacts_dir / "feature_columns.json")
        with open(feature_columns_path, "w") as f:
            json.dump(feature_cols, f, indent=4) 
        
        # Log feature columns to MLflow, This creates an MLflow artifact containing the exact feature order used by the trained model.
        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")
        
        # Save preprocessing metadata, This metadata tells the serving pipeline:
        # which features the model expects,  what the target column is called, This helps prevent training/serving feature mismatch.
        preprocessing_artifact = {"feature_columns": feature_cols, "target": "default"}
        preprocessing_path = (artifacts_dir / "preprocessing.pkl")
        joblib.dump(preprocessing_artifact, preprocessing_path)

        # Log preprocessing metadata to MLflow     
        mlflow.log_artifact(preprocessing_path)  
        print(f"✅ Saved {len(feature_cols)} feature columns "f"for serving consistency")
        
        # STAGE 5: TRAIN / TEST SPLIT, CLASS BALANCE AND MODEL TRAINING
        print("📊 Preparing model training...")

        target = args.target
        X_for_balance = feature_data.drop(columns=[target, "customerid", "approveddate", "creationdate"])
        y_for_balance = feature_data[target]

        class_0_count = (y_for_balance == 0).sum()
        class_1_count = (y_for_balance == 1).sum()

        class_imbalance_ratio = class_0_count / class_1_count

        print(f"📈 Class distribution: No Default={class_0_count}, Default={class_1_count}")
        print(f"📈 Class imbalance ratio: {class_imbalance_ratio:.2f}")
        mlflow.log_metric("class_0_count", int(class_0_count))
        mlflow.log_metric("class_1_count", int(class_1_count))
        mlflow.log_metric("class_imbalance_ratio", float(class_imbalance_ratio))

        print("🤖 Training Logistic Regression model...")

        # === Train Model and Track Training Time ===
        t0 = time.time()
        model, scaler, X_test_scaled, y_test, y_pred, top_6_key_factors = train_model(feature_data, target_col=target, threshold=args.threshold, test_size=args.test_size)
        train_time = time.time() - t0
        mlflow.log_metric("train_time", train_time)
        print(f"✅ Model trained in {train_time:.2f} seconds")

        #feature coefficients
        print("\n🔑 Top 6 Key Factors Influencing Loan Default:")
        for i, feature in enumerate(top_6_key_factors, start=1):
            print(f"   {i}. {feature}")

        
        # SAVE TOP 6 FEATURES AS MLFLOW ARTIFACT
        top_factors_text = (
            "Top 6 Key Factors Influencing Loan Default\n\n"
        )

        for i, feature in enumerate(top_6_key_factors, start=1):
            top_factors_text += f"{i}. {feature}\n"

        mlflow.log_text(
            top_factors_text,
            artifact_file="top_6_key_factors.txt"
        )



        
        # ============================================================
        # STAGE 6: MODEL EVALUATION
        # ============================================================

        print("📊 Evaluating model performance...")

        # Generate predictions and track inference time
        t1 = time.perf_counter()

        proba = model.predict_proba(X_test_scaled)[:, 1]

        risk_results = pd.DataFrame({"default_probability": proba})
        risk_results[["risk_score", "risk_category"]] = risk_results["default_probability"].apply(lambda x: pd.Series(calculate_risk_score(x)))
    
        risk_results["model_decision"] = np.where(y_pred == 1, "Default", "No Default")
        print("\n🎯 Loan Risk Assessment:")
        print(risk_results.head(10).to_string(index=False))

        # Apply classification threshold
        # Threshold = 0.40, as used during model training
        # Lower threshold generally increases sensitivity to defaults
        y_pred = (proba >= args.threshold).astype(int)

    
        cm = confusion_matrix(y_test, y_pred)
        cm_text = f"Confusion Matrix:\n\n{cm}\n\nRows = Actual\nColumns = Predicted\n\n                  Predicted\n                  No Default    Default\nActual No Default  {cm[0,0]:<13}{cm[0,1]}\nActual Default     {cm[1,0]:<13}{cm[1,1]}\n"
        mlflow.log_text(cm_text, artifact_file="confusion_matrix.txt")
        print("\n📊 Confusion Matrix:")
        print(cm)

        #confusion metric visual artifact
        plt.figure(figsize=(7, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Default", "Default"],
            yticklabels=["No Default", "Default"]
        )

        plt.title("Loan Default Prediction - Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")

        plt.tight_layout()

        # Save visualization locally
        cm_path = ARTIFACT_DIR / "confusion_matrix.png"

        plt.savefig(
            cm_path,
            dpi=300,
            bbox_inches="tight"
        )

        # Log visualization to MLflow
        mlflow.log_artifact(cm_path)
        plt.close()

        pred_time = time.perf_counter() - t1

        # Log inference time to MLflow
        mlflow.log_metric("pred_time", pred_time)

        
        # EVALUATION METRICS       
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, proba)

        
        # LOG EVALUATION METRICS TO MLFLOW
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)

        
        # DISPLAY MODEL PERFORMANCE
        print("🎯 Logistic Regression Model Performance:")
        print(f"   Precision: {precision:.3f} | Recall: {recall:.3f}")
        print(f"   F1 Score: {f1:.3f} | Accuracy: {accuracy:.3f}")
        print(f"   ROC AUC: {roc_auc:.3f}")
        print(f"   Prediction Time: {pred_time:.4f} seconds")

        # ============================================================
        # STAGE 7: MODEL SERIALIZATION AND LOGGING
        # ============================================================

        print("💾 Saving Logistic Regression model to MLflow...")

        mlflow.sklearn.log_model(model, artifact_path="model")

        print("✅ Logistic Regression model saved to MLflow for serving")

        
        # FINAL PERFORMANCE SUMMARY

        print("\n⏱️ Performance Summary:")
        print(f"   Training time: {train_time:.2f}s")
        print(f"   Inference time: {pred_time:.4f}s")
        print(f"   Samples per second: {len(X_test_scaled) / pred_time:.0f}")

        print("\n📈 Detailed Loan Default Classification Report:")
        print(classification_report(y_test, y_pred, digits=3))

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Run Nigerian loan default prediction pipeline with Logistic Regression + MLflow"
    )
    p.add_argument("--trainperf", type=str, required=True, help="Path to trainperf.csv")
    p.add_argument("--customer_prevcleaned", type=str, required=True, help="Path to customer_prevcleaned.csv")
    p.add_argument("--demographics", type=str, required=True, help="Path to traindemographics.csv")
    p.add_argument("--target", type=str, default="default")
    p.add_argument("--threshold", type=float, default=0.40)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Nigerian Loan Default Prediction")
    p.add_argument("--mlflow_uri", type=str, default=None, help="Override MLflow tracking URI; otherwise uses project_root/mlruns")

    args = p.parse_args()
    main(args)


"""
#   Use this below to run the pipeline

python scripts/run_pipeline.py --trainperf data/trainperf.csv --customer_prevcleaned data/customer_prevcleaned.csv --demographics data/traindemographics.csv


"""
















