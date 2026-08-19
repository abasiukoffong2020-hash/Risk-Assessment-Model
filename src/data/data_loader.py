# this is the data loading script
import pandas as pd
import os                # Import the os module for file and directory operations


def load_data(file_path: str) -> pd.DataFrame:            # Function to load a CSV file into a pandas DataFrame
    """
    Loads a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """

    if not os.path.exists(file_path):                       # Check whether the specified CSV file exists
        raise FileNotFoundError(f"File not found: {file_path}")                     # Raise an error if the file cannot be found

    return pd.read_csv(file_path)                               # Read the CSV file and return it as a pandas DataFrame