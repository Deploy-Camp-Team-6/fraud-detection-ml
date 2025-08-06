# src/components/model_trainer.py
import logging
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

class ModelTrainer:
    """A factory for creating and training different models."""
    def __init__(self, model_name: str, params: dict):
        """
        Initializes the ModelTrainer.

        Args:
            model_name (str): The name of the model to train (e.g., 'xgboost').
            params (dict): A dictionary containing all hyperparameters.
        """
        self.model_name = model_name.lower()
        self.params = params
        self.random_state = params['train']['random_state']
        logging.info(f"Initializing ModelTrainer for model: {self.model_name}")

    def _get_model(self):
        """
        Selects and instantiates a model based on the model_name.
        This is the factory part.
        """
        if self.model_name == "xgboost":
            model_params = self.params.get('xgboost', {})
            return XGBClassifier(random_state=self.random_state, **model_params)

        elif self.model_name == "lightgbm":
            model_params = self.params.get('lightgbm', {})
            return LGBMClassifier(random_state=self.random_state, **model_params)

        elif self.model_name == "logistic_regression":
            model_params = self.params.get('logistic_regression', {})
            return LogisticRegression(random_state=self.random_state, **model_params)
        
        else:
            logging.error(f"Unsupported model: {self.model_name}")
            raise ValueError(f"Unsupported model: {self.model_name}")

    def train(self, X_train, y_train):
        """
        Initializes and fits the selected model.

        Args:
            X_train: Training feature data.
            y_train: Training target data.

        Returns:
            A trained scikit-learn compatible model object.
        """
        model = self._get_model()
        logging.info(f"Starting training for {self.model_name}...")
        model.fit(X_train, y_train)
        logging.info(f"Training for {self.model_name} complete.")
        return model