import argparse
import logging
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency

# Add project root to Python path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_drift_analysis(reference_df: pd.DataFrame, current_df: pd.DataFrame, config: dict):
    """
    Performs drift analysis between a reference and current dataset.
    - Kolmogorov-Smirnov test for numerical features.
    - Chi-Squared test for categorical features.
    """
    logging.info("Starting data drift analysis...")

    numerical_cols = config['features']['numerical_cols']
    categorical_cols = config['features']['categorical_cols']

    drift_report = {}

    # 1. Numerical feature drift (KS test)
    logging.info("Analyzing numerical feature drift...")
    for col in numerical_cols:
        if col in reference_df.columns and col in current_df.columns:
            stat, p_value = ks_2samp(reference_df[col], current_df[col])
            drift_report[col] = {
                'type': 'numerical',
                'statistic': stat,
                'p_value': p_value,
                'is_drifted': p_value < 0.05  # Significance level alpha = 5%
            }
            logging.info(f"  - {col}: p-value={p_value:.4f} {'(Drift Detected)' if p_value < 0.05 else ''}")
        else:
            logging.warning(f"Column '{col}' not found in one of the dataframes. Skipping.")

    # 2. Categorical feature drift (Chi-Squared test)
    logging.info("Analyzing categorical feature drift...")
    for col in categorical_cols:
        if col in reference_df.columns and col in current_df.columns:
            # Create a contingency table
            contingency_table = pd.crosstab(
                pd.concat([reference_df[[col]].assign(source='ref'), current_df[[col]].assign(source='cur')])[col],
                pd.concat([reference_df[[col]].assign(source='ref'), current_df[[col]].assign(source='cur')])['source']
            )

            stat, p_value, _, _ = chi2_contingency(contingency_table)
            drift_report[col] = {
                'type': 'categorical',
                'statistic': stat,
                'p_value': p_value,
                'is_drifted': p_value < 0.05
            }
            logging.info(f"  - {col}: p-value={p_value:.4f} {'(Drift Detected)' if p_value < 0.05 else ''}")
        else:
            logging.warning(f"Column '{col}' not found in one of the dataframes. Skipping.")

    return drift_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor data drift between two datasets.")
    parser.add_argument(
        "--reference_data",
        type=str,
        required=True,
        help="Path to the reference dataset (e.g., training data)."
    )
    parser.add_argument(
        "--current_data",
        type=str,
        required=True,
        help="Path to the current dataset to check for drift."
    )
    args = parser.parse_args()

    # Load project configuration to get feature types
    try:
        config = load_config()
    except FileNotFoundError:
        logging.error("Could not load config.yaml. Make sure you are running from the project root.")
        sys.exit(1)

    # Load datasets
    try:
        ref_df = pd.read_csv(args.reference_data)
        cur_df = pd.read_csv(args.current_data)
    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}")
        sys.exit(1)

    # Run analysis
    report = run_drift_analysis(ref_df, cur_df, config)

    # Print summary
    logging.info("\n--- Drift Analysis Report ---")
    drift_detected = False
    for feature, results in report.items():
        if results['is_drifted']:
            drift_detected = True
            logging.warning(f"Drift DETECTED in '{feature}' (p-value: {results['p_value']:.4f})")

    if not drift_detected:
        logging.info("No significant data drift was detected.")
    logging.info("---------------------------\n")
