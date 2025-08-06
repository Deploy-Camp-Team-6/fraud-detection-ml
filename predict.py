import argparse
import joblib
import pandas as pd
import json
import os

def predict(model_path: str, preprocessor_path: str, data: pd.DataFrame) -> pd.DataFrame:
    """Makes predictions on new data."""

    # Load model and preprocessor
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    # Preprocess data
    data_processed = preprocessor.transform(data)

    # Make predictions
    predictions = model.predict(data_processed)
    probabilities = model.predict_proba(data_processed)[:, 1]

    result_df = pd.DataFrame({
        'prediction': predictions,
        'probability_fraud': probabilities
    })

    return result_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud Detection Prediction Script")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory where the model and preprocessor are saved."
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help='JSON string of the input data. Example: \'[{"amount": 100, "merchant_type": "travel", "device_type": "mobile"}]\'',
    )
    args = parser.parse_args()

    # Construct paths
    model_path = os.path.join(args.model_dir, 'model.joblib') # Assuming a default name
    preprocessor_path = os.path.join(args.model_dir, 'preprocessor.joblib')

    # Load data from JSON
    input_data = pd.DataFrame(json.loads(args.input_json))

    # Make predictions
    results = predict(model_path, preprocessor_path, input_data)

    print("Prediction Results:")
    print(results)
