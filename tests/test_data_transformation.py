import os

# Add src to path to allow imports
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.components.data_transformation import DataTransformation

# A sample feature configuration for testing
SAMPLE_FEATURE_CONFIG = {
    'target_column': 'label',
    'numerical_cols': ['amount'],
    'categorical_cols': ['merchant_type', 'device_type'],
    'drop_cols': ['transaction_id']
}

SAMPLE_PARAMS = {
    'data_transformation': {
        'numeric_imputer_strategy': 'median',
        'categorical_imputer_strategy': 'most_frequent'
    }
}

def test_preprocessor_creation():
    """
    Tests if the DataTransformation class correctly creates a preprocessor
    based on the provided configuration.
    """
    # Arrange
    transformer = DataTransformation(
        feature_config=SAMPLE_FEATURE_CONFIG,
        params=SAMPLE_PARAMS
    )

    # Act
    preprocessor = transformer.preprocessor

    # Assert
    assert isinstance(preprocessor, ColumnTransformer)
    assert len(preprocessor.transformers) == 2

    # Check the numeric transformer
    numeric_transformer_tuple = preprocessor.transformers[0]
    assert numeric_transformer_tuple[0] == 'numeric'
    assert isinstance(numeric_transformer_tuple[1], Pipeline)
    assert len(numeric_transformer_tuple[1].steps) == 3 # log_transformer, imputer, scaler
    assert numeric_transformer_tuple[1].steps[0][0] == 'log_transformer'

    # Check the categorical transformer
    categorical_transformer_tuple = preprocessor.transformers[1]
    assert categorical_transformer_tuple[0] == 'categorical'
    assert isinstance(categorical_transformer_tuple[1], Pipeline)
    assert len(categorical_transformer_tuple[1].steps) == 2 # imputer, onehot

def test_transformation_on_sample_data():
    """
    Tests if the preprocessor created by DataTransformation can correctly
    transform sample data without errors.
    """
    # Arrange
    transformer = DataTransformation(
        feature_config=SAMPLE_FEATURE_CONFIG,
        params=SAMPLE_PARAMS
    )
    preprocessor = transformer.preprocessor

    sample_data = pd.DataFrame({
        'transaction_id': [1, 2, 3],
        'amount': [100.0, 200.0, 50.0],
        'merchant_type': ['retail', 'travel', 'retail'],
        'device_type': ['mobile', 'desktop', 'mobile'],
        'label': [0, 1, 0]
    })

    X_sample = sample_data.drop(columns=['label', 'transaction_id'])

    # Act
    X_transformed = preprocessor.fit_transform(X_sample)

    # Assert
    assert X_transformed.shape[0] == 3
    # The number of columns in X_transformed will depend on one-hot encoding
    # For this sample data, merchant_type has 2 unique values, device_type has 2.
    # Total columns = 1 (numeric) + 2 (merchant_type) + 2 (device_type) = 5
    assert X_transformed.shape[1] == 5

    # Verify that the numeric column is the log1p of the original amounts
    expected_numeric = np.log1p(X_sample['amount']).values
    scaler = preprocessor.named_transformers_['numeric'].named_steps['scaler']
    numeric_column = scaler.inverse_transform(X_transformed[:, [0]]).ravel()
    assert np.allclose(numeric_column, expected_numeric)

    # Verify that feature names are returned as expected
    expected_features = [
        'amount',
        'merchant_type_retail',
        'merchant_type_travel',
        'device_type_desktop',
        'device_type_mobile',
    ]
    assert transformer.get_feature_names() == expected_features


def test_get_feature_names_without_fit_raises_error():
    transformer = DataTransformation(
        feature_config=SAMPLE_FEATURE_CONFIG,
        params=SAMPLE_PARAMS,
    )
    with pytest.raises(AttributeError):
        transformer.get_feature_names()
