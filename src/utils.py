# src/utils.py
import yaml
import logging

def load_config(path="config/config.yaml"):
    """Loads a YAML configuration file."""
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {path}")
        raise
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

def load_params(path="params.yaml"):
    """Loads a YAML parameters file."""
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Parameters file not found at {path}")
        raise
    except Exception as e:
        logging.error(f"Error loading parameters: {e}")
        raise