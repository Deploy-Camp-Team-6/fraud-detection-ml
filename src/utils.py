# src/utils.py
import os
import yaml
import logging
import re
from dotenv import load_dotenv

def _replace_env_vars(config_str: str) -> str:
    """Replaces ${VAR} or $VAR in a string with environment variables."""
    # Find all ${VAR} style placeholders
    pattern = re.compile(r'\$\{(\w+)\}')
    return pattern.sub(lambda m: os.getenv(m.group(1), m.group(0)), config_str)

def load_config(path="config/config.yaml"):
    """
    Loads a YAML configuration file and replaces environment variable placeholders.
    It also loads environment variables from a .env file if it exists.
    """
    # Load .env file into environment variables
    load_dotenv()

    try:
        with open(path, 'r') as f:
            config_str = f.read()

        # Replace environment variables
        config_str = _replace_env_vars(config_str)

        return yaml.safe_load(config_str)
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