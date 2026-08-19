# inference.py takes information about a person applying for a loan, 
# looks up any relevant existing customer information, applies the project's business rules, 
# prepares the information in exactly the same format the model was trained on, 
# asks the model how likely the applicant is to default, 
# and converts that result into an understandable loan-risk assessment.

import os
import sys
from pathlib import Path

# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import load_data
from src.data.preprocessing import (
    merge_datasets,
    preprocess_data,
    save_trainperf_merged
)


# Create lookup directory if it does not exist
lookup_dir = PROJECT_ROOT / "data" / "lookup"
lookup_dir.mkdir(
    parents=True,
    exist_ok=True
)


# Load original datasets
trainperf = load_data(
    PROJECT_ROOT / "data" / "trainperf.csv"
)

customer_prevcleaned = load_data(
    PROJECT_ROOT / "data" / "customer_prevcleaned.csv"
)

traindemographics = load_data(
    PROJECT_ROOT / "data" / "traindemographics.csv"
)


# Recreate trainperf_merged
merged_data = merge_datasets(
    trainperf,
    customer_prevcleaned,
    traindemographics
)


# Apply the same preprocessing used by the project
trainperf_merged = preprocess_data(
    merged_data
)


# Save reference data for serving
reference_path = (
    lookup_dir / "trainperf_merged_reference.csv"
)

save_trainperf_merged(
    trainperf_merged,
    reference_path
)