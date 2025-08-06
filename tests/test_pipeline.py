# tests/test_pipeline.py
import pytest
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

# Add src to path to allow imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.components.model_trainer import ModelTrainer

@pytest.fixture
def base_params():
    """Provides a base parameter dictionary for the trainer."""
    return {
        'train': {'random_state': 42},
        'xgboost': {'n_estimators': 10},
        'lightgbm': {'n_estimators': 10},
        'logistic_regression': {'max_iter': 100}
    }

@pytest.mark.parametrize("model_name, expected_model_type", [
    ("xgboost", XGBClassifier),
    ("lightgbm", LGBMClassifier),
    ("logistic_regression", LogisticRegression),
])
def test_model_trainer_factory_success(model_name, expected_model_type, base_params):
    """
    Tests that the ModelTrainer correctly creates the specified model.
    """
    # Arrange
    trainer = ModelTrainer(model_name=model_name, params=base_params)

    # Act
    model = trainer._get_model()

    # Assert
    assert isinstance(model, expected_model_type)

def test_model_trainer_factory_unsupported(base_params):
    """
    Tests that the ModelTrainer raises a ValueError for an unsupported model.
    """
    # Arrange
    unsupported_model_name = "unsupported_model"
    trainer = ModelTrainer(model_name=unsupported_model_name, params=base_params)

    # Act & Assert
    with pytest.raises(ValueError, match=f"Unsupported model: {unsupported_model_name}"):
        trainer._get_model()
