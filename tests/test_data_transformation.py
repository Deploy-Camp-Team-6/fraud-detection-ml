# tests/test_data_transformation.py
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

# Add src to path to allow imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.components.data_transformation import DataTransformation, DataTransformationConfig

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