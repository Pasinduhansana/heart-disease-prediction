import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

COLUMN_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal', 'target'
]

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'Data_set',
    'processed.cleveland.data'
)


def load_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """Load the Cleveland heart disease dataset from a file path."""
    if data_path is None:
        data_path = DEFAULT_DATA_PATH

    df = pd.read_csv(data_path, names=COLUMN_NAMES, na_values='?', encoding='latin-1')
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset, convert types, impute missing values and binarise the target."""
    df = df.copy()

    # Replace string missing values with pandas NaN
    df.replace('?', np.nan, inplace=True)

    # Convert all columns to numeric types
    for column in COLUMN_NAMES:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    # Handle missing values using median imputation to preserve numerical distributions.
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Convert target into binary classification: 0 = no disease, 1 = disease present.
    # The original dataset uses 1-4 to indicate increasing levels of heart disease.
    df['target'] = df['target'].apply(lambda value: 0 if value == 0 else 1)

    return df


def summarize_target(df: pd.DataFrame) -> pd.Series:
    """Return the distribution of the binary target column."""
    return df['target'].value_counts().sort_index()


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features and binary target for model training."""
    X = df.drop(columns=['target'])
    y = df['target']
    return X, y
