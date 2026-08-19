import pandas as pd


def build_features(
    df: pd.DataFrame,
    target_col: str = "default"
) -> pd.DataFrame:
    """
    Apply feature engineering to the Nigerian loan default dataset.

    This function reproduces the categorical encoding performed
    during model development in the notebook.

    Feature engineering steps:
        1. One-hot encode bank_account_type.
        2. Use drop_first=True to avoid redundant dummy variables.
        3. Convert Boolean dummy columns to integers (0 and 1).
        4. Keep the target column unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed loan dataset.

    target_col : str, default="default"
        Name of the target variable. The target is not encoded
        or otherwise modified by this function.

    Returns
    -------
    pd.DataFrame
        Feature-engineered dataframe ready for machine learning.
    """

    # Work on a copy so the original dataframe is not modified
    df = df.copy()

    print(
        f"🔧 Starting feature engineering on "
        f"{df.shape[1]} columns..."
    )

    # ============================================================
    # STEP 1: CHECK FOR BANK ACCOUNT TYPE
    # ============================================================

    if "bank_account_type" in df.columns:

        print(
            "   🌟 Applying one-hot encoding to "
            "bank_account_type..."
        )

        # One-hot encode bank account type
        # drop_first=True removes one reference category
        df = pd.get_dummies(
            df,
            columns=["bank_account_type"],
            drop_first=True
        )

        print(
            "      ✅ Created bank account type dummy variables"
        )

    else:

        print(
            "   ℹ️ bank_account_type is already encoded "
            "or is not present."
        )

    # ============================================================
    # STEP 2: CONVERT BOOLEAN FEATURES TO INTEGER
    # ============================================================

    # Identify Boolean columns created by one-hot encoding
    bool_cols = [
        col
        for col in df.columns
        if col.startswith("bank_account_type_")
        and df[col].dtype == "bool"
    ]

    if bool_cols:

        df[bool_cols] = df[bool_cols].astype(int)

        print(
            f"   🔄 Converted Boolean columns to integers: "
            f"{bool_cols}"
        )

    # ============================================================
    # STEP 3: VERIFY TARGET WAS NOT MODIFIED
    # ============================================================

    if target_col in df.columns:

        print(
            f"   🎯 Target column '{target_col}' "
            f"left unchanged."
        )

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    print(
        f"✅ Feature engineering complete: "
        f"{df.shape[1]} columns"
    )

    return df
