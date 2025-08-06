import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
import numpy as np

def load_data(file_path: str) -> pd.DataFrame:
    """Loads data from a CSV file."""
    return pd.read_csv(file_path)

def get_preprocessing_pipeline(numerical_cols, categorical_cols, log_transform_cols):
    """Creates a Scikit-learn pipeline for preprocessing."""

    # Pipeline for numerical features (including log transform)
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline for log-transformed numerical features
    log_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('log_transformer', FunctionTransformer(np.log1p, validate=False)),
        ('scaler', StandardScaler())
    ])

    # Pipeline for categorical features
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, [col for col in numerical_cols if col not in log_transform_cols]),
            ('log_num', log_pipeline, log_transform_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ],
        remainder='passthrough'
    )

    return preprocessor

def split_data(df: pd.DataFrame, target_column: str, test_size: float, random_state: int):
    """Splits data into training and testing sets with stratification."""
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test
