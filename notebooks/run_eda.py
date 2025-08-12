import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to Python path to allow importing utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import load_config

def run_analysis():
    """
    Performs and saves an exploratory data analysis of the dataset.
    """
    try:
        # Load configuration to get data path
        config = load_config()
        data_path = os.path.join(
            config['data_source']['raw_data_dir'],
            config['data_source']['raw_data_filename']
        )

        logging.info(f"Loading dataset from: {data_path}")
        df = pd.read_csv(data_path)

        # Create output directory for plots
        output_dir = "analysis_plots"
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Plots will be saved to '{output_dir}' directory.")

        # --- Basic Info ---
        logging.info("Dataset Info:")
        df.info()
        logging.info("\nDescriptive Statistics:")
        print(df.describe())
        logging.info("\nMissing Values:")
        print(df.isnull().sum())

        # --- Target Variable Analysis ---
        plt.figure(figsize=(6, 4))
        sns.countplot(x='label', data=df)
        plt.title('Distribution of Fraud Labels')
        plt.xlabel('Label (0: Not Fraud, 1: Fraud)')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, '01_label_distribution.png'))
        plt.close()
        logging.info("Saved label distribution plot.")

        # --- Numerical Feature Analysis: Amount ---
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(df['amount'], bins=50, kde=True)
        plt.title('Distribution of Transaction Amount')

        plt.subplot(1, 2, 2)
        sns.histplot(np.log1p(df['amount']), bins=50, kde=True)
        plt.title('Distribution of Log-Transformed Amount')
        plt.savefig(os.path.join(output_dir, '02_amount_distribution.png'))
        plt.close()
        logging.info("Saved amount distribution plots.")

        # --- Categorical Feature Analysis ---
        for col in config['features']['categorical_cols']:
            plt.figure(figsize=(10, 6))
            sns.countplot(y=col, data=df, order=df[col].value_counts().index)
            plt.title(f'Distribution of {col}')
            plt.savefig(os.path.join(output_dir, f'03_{col}_distribution.png'))
            plt.close()
            logging.info(f"Saved {col} distribution plot.")

            # vs. Label
            plt.figure(figsize=(10, 6))
            sns.countplot(y=col, hue='label', data=df, order=df[col].value_counts().index)
            plt.title(f'Distribution of {col} by Fraud Label')
            plt.savefig(os.path.join(output_dir, f'04_{col}_vs_label.png'))
            plt.close()
            logging.info(f"Saved {col} vs. label plot.")

        logging.info("Exploratory Data Analysis complete.")

    except Exception as e:
        logging.error(f"An error occurred during EDA: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    run_analysis()
