# src/utils.py
import os
import yaml
import logging
import re
from dotenv import load_dotenv

def _replace_env_vars(config_str: str) -> str:
    """Replaces ``${VAR}`` or ``$VAR`` in a string with environment variables.

    The previous implementation only supported the ``${VAR}`` style. However, the
    configuration files in this project may use either form, so we need to
    capture both.  Any placeholder without a corresponding environment variable
    is left unchanged.
    """

    # Match either ${VAR} or $VAR. ``group(1)`` contains the name for ${VAR}
    # matches while ``group(2)`` contains the name for $VAR matches.
    pattern = re.compile(r"\$(?:\{(\w+)\}|(\w+))")

    def replace(match: re.Match) -> str:
        var_name = match.group(1) or match.group(2)
        return os.getenv(var_name, match.group(0))

    return pattern.sub(replace, config_str)

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