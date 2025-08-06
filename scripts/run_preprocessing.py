# scripts/run_preprocessing.py
import os
import sys
import yaml
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Add src to Python path to allow component imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.components.data_transformation import DataTransformation, DataTransformationConfig

def main():
    """
    Main function to execute the preprocessing pipeline.
    """
    # Load configs
    with open("params.yaml", 'r') as f:
        params = yaml.safe_load(f)
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Define paths
    raw_data_path = os.path.join(config['data_source']['raw_data_dir'], config['data_source']['raw_data_filename'])
    processed_dir = config['data_source']['processed_data_dir']
    
    # Create processed directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(raw_data_path)

    # Split data
    train_df, test_df = train_test_split(
        df,
        test_size=params['train']['test_size'],
        random_state=params['train']['random_state'],
        stratify=df[params['train']['target_column']] # Stratify for imbalanced data
    )

    # Initialize and run data transformation
    transformation_config = DataTransformationConfig(target_column=params['train']['target_column'])
    data_transformer = DataTransformation(config=transformation_config)
    
    # We only need the preprocessor object and transformed datasets here
    X_train_processed, y_train, X_test_processed, y_test, preprocessor, _ = \
        data_transformer.initiate_data_transformation(train_df, test_df)

    # Combine processed features and target for saving
    train_arr = pd.concat([pd.DataFrame(X_train_processed), y_train.reset_index(drop=True)], axis=1)
    test_arr = pd.concat([pd.DataFrame(X_test_processed), y_test.reset_index(drop=True)], axis=1)
    
    # Save processed data
    train_path = os.path.join(processed_dir, config['artifacts']['processed_train_name'])
    test_path = os.path.join(processed_dir, config['artifacts']['processed_test_name'])
    train_arr.to_csv(train_path, index=False, header=False)
    test_arr.to_csv(test_path, index=False, header=False)

    # Save preprocessor object
    preprocessor_path = os.path.join(processed_dir, config['artifacts']['preprocessor_name'])
    joblib.dump(preprocessor, preprocessor_path)

    print(f"Preprocessing complete. Processed data saved to {processed_dir}")
    print(f"Preprocessor object saved to {preprocessor_path}")

if __name__ == "__main__":
    main()