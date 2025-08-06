import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataTransformation:
    """Handles feature engineering and preprocessing based on configuration."""

    def __init__(self, feature_config: dict):
        """
        Initializes the DataTransformation component.
        Args:
            feature_config (dict): A dictionary containing feature definitions
                                   (e.g., target, numerical, categorical, drop).
        """
        self.config = feature_config
        self.preprocessor = self._create_preprocessor()

    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Creates a scikit-learn ColumnTransformer for preprocessing based on config.
        - Applies log transform, imputation, and scaling to numerical features.
        - Applies imputation and one-hot encoding to categorical features.
        """
        logging.info("Building preprocessing pipeline from configuration.")

        # Pipeline for numerical features: log transform -> impute -> scale
        # We use np.log1p which is log(1+x) to handle zero values in 'amount'.
        numeric_pipeline = Pipeline(steps=[
            ("log_transformer", FunctionTransformer(np.log1p, validate=True)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # Pipeline for categorical features: impute -> one-hot encode
        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, self.config['numerical_cols']),
                ("categorical", categorical_pipeline, self.config['categorical_cols'])
            ],
            remainder='drop' # Drop columns not specified in transformers
        )
        
        logging.info("Preprocessing pipeline built successfully.")
        return preprocessor

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
        """Fits the preprocessor and transforms the data."""
        logging.info("Fitting preprocessor and transforming data.")
        
        # Drop unused columns first
        df_processed = df.drop(columns=self.config.get('drop_cols', []), errors='ignore')

        X = df_processed.drop(columns=[self.config['target_column']])
        y = df_processed[self.config['target_column']]

        X_transformed = self.preprocessor.fit_transform(X)

        return X_transformed, y

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
        """Transforms data using the fitted preprocessor."""
        logging.info("Transforming new data with the existing preprocessor.")

        df_processed = df.drop(columns=self.config.get('drop_cols', []), errors='ignore')

        X = df_processed.drop(columns=[self.config['target_column']])
        y = df_processed[self.config['target_column']]

        X_transformed = self.preprocessor.transform(X)

        return X_transformed, y

    def get_feature_names(self) -> List[str]:
        """Returns the feature names after transformation."""
        try:
            numeric_features = self.config['numerical_cols']
            categorical_features_raw = self.config['categorical_cols']

            # Get feature names from the one-hot encoder
            onehot_transformer = self.preprocessor.named_transformers_['categorical']['onehot']
            categorical_features_encoded = onehot_transformer.get_feature_names_out(categorical_features_raw).tolist()

            return numeric_features + categorical_features_encoded
        except Exception as e:
            logging.error(f"Could not retrieve feature names: {e}")
            return []
