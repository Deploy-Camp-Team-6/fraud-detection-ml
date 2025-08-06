# src/components/data_transformation.py
import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class DataTransformationConfig:
    """Configuration for data transformation."""
    target_column: str = "isFraud"
    # Drop high-cardinality or identifier columns
    drop_columns: List[str] = field(default_factory=lambda: ["step", "nameOrig", "nameDest"])

class DataTransformation:
    """Handles feature engineering and preprocessing."""
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_preprocessor(self, numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
        """
        Creates and returns a scikit-learn ColumnTransformer for preprocessing.
        - Imputes and scales numerical features.
        - Imputes and one-hot encodes categorical features.
        """
        logging.info("Building preprocessing pipeline.")
        
        # Pipeline for numerical features
        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")), # Robust to outliers
            ("scaler", StandardScaler())
        ])

        # Pipeline for categorical features
        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")), # or 'constant', fill_value='missing'
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numeric_features),
                ("categorical", categorical_pipeline, categorical_features)
            ],
            remainder='passthrough' # Keep other columns if any, though we drop them first
        )
        
        logging.info("Preprocessing pipeline built successfully.")
        return preprocessor

    def initiate_data_transformation(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Applies the complete data transformation process.
        - Identifies feature types
        - Builds and fits the preprocessor on training data
        - Transforms both train and test data
        - Returns preprocessor object and transformed data arrays
        """
        logging.info("Starting data transformation process.")
        
        # Drop specified columns
        train_df = train_df.drop(columns=self.config.drop_columns, errors='ignore')
        test_df = test_df.drop(columns=self.config.drop_columns, errors='ignore')

        # Separate features (X) and target (y)
        X_train = train_df.drop(columns=[self.config.target_column])
        y_train = train_df[self.config.target_column]
        X_test = test_df.drop(columns=[self.config.target_column])
        y_test = test_df[self.config.target_column]

        # Identify feature types
        numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

        logging.info(f"Identified {len(numeric_features)} numerical features: {numeric_features}")
        logging.info(f"Identified {len(categorical_features)} categorical features: {categorical_features}")

        # Get and fit the preprocessor
        preprocessor = self.get_preprocessor(numeric_features, categorical_features)
        preprocessor.fit(X_train)

        # Transform the data
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # Get feature names after transformation for logging
        feature_names = numeric_features + \
            preprocessor.named_transformers_['categorical']['onehot'].get_feature_names_out(categorical_features).tolist()
        
        logging.info("Data transformation complete.")
        return X_train_processed, y_train, X_test_processed, y_test, preprocessor, feature_names