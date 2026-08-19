import pandas as pd

"""
Basic cleaning for the loan default prediction project.
- remove duplicate customer records from the traindemographics dataset.
- rename columns
- dataset merging
- date conversions
- customer age calculation
- rename target column and classes, the map
- fill missing values
- drop unnecessary columns

"""


def merge_datasets(
    trainperf: pd.DataFrame,
    customer_prevcleaned: pd.DataFrame,
    traindemographics: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge customer historical and demographic data into the current loan dataset.
    """

    # Remove duplicate customer records
    traindemographics = traindemographics.drop_duplicates(
        subset="customerid",
        keep="first"
    )

    # Rename loan number in customer historical data
    customer_prevcleaned = customer_prevcleaned.rename(
        columns={"loannumber": "no_of_previous_loans"}
    )

    # Rename loan number in current loan data
    trainperf = trainperf.rename(
        columns={"loannumber": "current_loan_number"}
    )

    # Merge customer historical data
    trainperf_merged = trainperf.merge(
        customer_prevcleaned[
            [
                "customerid",
                "previous_default",
                "no_of_previous_loans",
                "days_overdue"
            ]
        ],
        on="customerid",
        how="left"
    )

    # Merge customer demographic data
    trainperf_merged = trainperf_merged.merge(
        traindemographics[
            [
                "customerid",
                "birthdate",
                "bank_account_type"
            ]
        ],
        on="customerid",
        how="left"
    )

    return trainperf_merged


def preprocess_data(trainperf_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data preprocessing for loan default prediction.
    """

    # Convert date columns to datetime
    date_columns = ["birthdate", "approveddate", "creationdate"]
    trainperf_merged[date_columns] = trainperf_merged[date_columns].apply(pd.to_datetime)

    # Calculate customer age
    trainperf_merged["age"] = (
        (pd.Timestamp.today() - trainperf_merged["birthdate"]).dt.days // 365
    )

    # Rename the target column
    trainperf_merged = trainperf_merged.rename(
        columns={"good_bad_flag": "default"}
    )

    # Rename target classes
    trainperf_merged["default"] = trainperf_merged["default"].replace(
        {
            "Good": "No Default",
            "Bad": "Default"
        }
    )

    # map target
    trainperf_merged["default"] = trainperf_merged["default"].map(
        {
            "No Default": 0,
            "Default": 1
        }
    )

    # Fill missing values in numerical columns
    num_cols = ["previous_default", "no_of_previous_loans", "days_overdue", "age"]
    trainperf_merged[num_cols] = trainperf_merged[num_cols].fillna(0)

    # Fill missing values in categorical column
    trainperf_merged["bank_account_type"] = trainperf_merged["bank_account_type"].fillna("None")

    # Drop unnecessary columns
    trainperf_merged = trainperf_merged.drop(
        columns=["birthdate", "referredby", "systemloanid"]
    )

    return trainperf_merged

def save_trainperf_merged(
    trainperf_merged: pd.DataFrame,
    output_path: str
) -> None:
    """
    Save the preprocessed trainperf_merged dataset
    for use as the customer reference database during serving.
    """

    trainperf_merged.to_csv(
        output_path,
        index=False
    )

    print(
        f"✅ Saved trainperf_merged reference data to: "
        f"{output_path}"
    )
    


