from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, X_test_scaled, y_test, threshold=0.40):
    """
    Evaluates the Logistic Regression model on test data.

    Args:
        model:
            Trained Logistic Regression model.

        X_test:
            Test features. These should already be scaled
            using the fitted StandardScaler.

        y_test:
            Actual test labels.

        threshold:
            Probability threshold used to classify a customer
            as default. Default is 0.40.
    """

    # Generate probability of default
    probabilities = model.predict_proba(X_test_scaled)[:, 1]

    # Apply the selected classification threshold
    preds = (probabilities >= threshold).astype(int)

    # Print classification report
    print(
        "Classification Report:\n",
        classification_report(y_test, preds, digits=3)
    )

    # Print confusion matrix
    print(
        "Confusion Matrix:\n",
        confusion_matrix(y_test, preds)
    )
