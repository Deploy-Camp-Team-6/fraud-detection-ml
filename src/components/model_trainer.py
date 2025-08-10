import logging
import numpy as np
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
        self.handle_imbalance = params['train'].get('handle_imbalance', False)
        logging.info(f"Initializing ModelTrainer for model: {self.model_name}")
        logging.info(f"Handle class imbalance: {self.handle_imbalance}")

    def _get_model(self, y_train=None):
        """
        Selects and instantiates a model based on the model_name.
        This is the factory part.
        Args:
            y_train: The training target data, needed for calculating scale_pos_weight.
        """
        model_params = self.params.get(self.model_name, {}).copy()

        if self.model_name == "xgboost":
            from xgboost import XGBClassifier

            if self.handle_imbalance and y_train is not None:
                neg_count = np.sum(y_train == 0)
                pos_count = np.sum(y_train == 1)
                if pos_count > 0:
                    scale_pos_weight = neg_count / pos_count
                    model_params['scale_pos_weight'] = scale_pos_weight
                    logging.info(f"XGBoost scale_pos_weight set to: {scale_pos_weight:.2f}")
            return XGBClassifier(random_state=self.random_state, **model_params)

        elif self.model_name == "lightgbm":
            from lightgbm import LGBMClassifier

            if not self.handle_imbalance and 'is_unbalance' in model_params:
                del model_params['is_unbalance']
            return LGBMClassifier(random_state=self.random_state, **model_params)

        elif self.model_name == "logistic_regression":
            if not self.handle_imbalance and 'class_weight' in model_params:
                del model_params['class_weight']
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
        model = self._get_model(y_train=y_train)
        logging.info(f"Starting training for {self.model_name}...")
        model.fit(X_train, y_train)
        logging.info(f"Training for {self.model_name} complete.")
        return model