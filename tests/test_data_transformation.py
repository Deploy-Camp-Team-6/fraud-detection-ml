# tests/test_data_transformation.py
import pandas as pd
import pytest
import numpy as np
from sklearn.compose import ColumnTransformer

# Add src to path to allow imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.components.data_transformation import DataTransformation, DataTransformationConfig

@pytest.fixture
def sample_data():
    """Provides a sample DataFrame for testing."""
    data = {
        'step': [1, 1, 2, 2],
        'type': ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'PAYMENT'],
        'amount': [100.0, 200.0, 50.0, 150.0],
        'nameOrig': ['C1', 'C2', 'C3', 'C4'],
        'oldbalanceOrg': [1000.0, 2000.0, 500.0, 1500.0],
        'newbalanceOrig': [900.0, 1800.0, 450.0, 1350.0],
        'nameDest': ['M1', 'M2', 'M3', 'M4'],
        'oldbalanceDest': [0.0, 100.0, 200.0, 300.0],
        'newbalanceDest': [100.0, 300.0, 250.0, 450.0],
        'isFraud': [0, 1, 0, 1],
        'isFlaggedFraud': [0, 0, 0, 0]
    }
    return pd.DataFrame(data)

def test_preprocessor_creation():
    """
    Tests if the get_preprocessor method returns a valid ColumnTransformer object.
    """
    # Arrange
    config = DataTransformationConfig()
    transformer = DataTransformation(config)
    numeric_features = ['amount', 'oldbalanceOrg']
    categorical_features = ['type']

    # Act
    preprocessor = transformer.get_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

    # Assert
    assert isinstance(preprocessor, ColumnTransformer)
    assert len(preprocessor.transformers) == 2
    assert preprocessor.transformers[0][0] == 'numeric'
    assert preprocessor.transformers[1][0] == 'categorical'

def test_data_transformation_logic(sample_data):
    """
    Tests the full data transformation process.
    """
    # Arrange
    config = DataTransformationConfig(target_column='isFraud')
    transformer = DataTransformation(config)

    # Use the same dataframe for train and test for simplicity
    train_df = sample_data
    test_df = sample_data.copy()

    # Act
    X_train_proc, y_train, X_test_proc, y_test, preprocessor, f_names = \
        transformer.initiate_data_transformation(train_df, test_df)

    # Assert
    # Check shapes
    assert X_train_proc.shape[0] == len(train_df)
    assert y_train.shape[0] == len(train_df)
    assert X_test_proc.shape[0] == len(test_df)
    assert y_test.shape[0] == len(test_df)

    # Check that specified columns are dropped
    dropped_cols = ["step", "nameOrig", "nameDest"]
    remaining_cols = [col for col in sample_data.columns if col not in dropped_cols and col != 'isFraud']

    # Check feature names after transformation
    # Number of original numeric features + number of one-hot encoded categories
    num_numeric = sample_data.select_dtypes(include=np.number).drop(columns=['isFraud', 'step']).shape[1]
    num_one_hot = len(sample_data['type'].unique())
    assert X_train_proc.shape[1] == num_numeric + num_one_hot
    assert len(f_names) == num_numeric + num_one_hot

    # Check that imputation has occurred (no NaNs in output)
    assert not np.isnan(X_train_proc).any()
    assert not np.isnan(X_test_proc).any()