# src/utils.py
import yaml
import logging
import os

def _get_env_var_name(path_keys):
    """Converts a list of dictionary keys to an uppercase, underscore-separated env var name."""
    return "_".join(path_keys).upper()

def _override_with_env_vars(config, path_keys=[]):
    """
    Recursively traverses the config dict and overrides values with
    corresponding environment variables.
    """
    for key, value in config.items():
        current_path_keys = path_keys + [key]
        if isinstance(value, dict):
            _override_with_env_vars(value, current_path_keys)
        else:
            env_var_name = _get_env_var_name(current_path_keys)
            env_var_value = os.getenv(env_var_name)
            if env_var_value:
                # Try to cast env var to the same type as the original value
                try:
                    original_type = type(value)
                    config[key] = original_type(env_var_value)
                    logging.info(f"Overrode '{'.'.join(current_path_keys)}' with env var '{env_var_name}'.")
                except (ValueError, TypeError):
                    config[key] = env_var_value # Fallback to string
                    logging.warning(f"Could not cast env var '{env_var_name}' to type {original_type}. Using as string.")


def load_config(path="config/config.yaml"):
    """
    Loads a YAML configuration file and allows environment variables to override its values.

    For a nested key like `minio_credentials.endpoint_url`, the corresponding
    environment variable would be `MINIO_CREDENTIALS_ENDPOINT_URL`.
    """
    try:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {path}")
        raise
    except Exception as e:
        logging.error(f"Error loading YAML configuration: {e}")
        raise

    # Override with environment variables
    _override_with_env_vars(config)

    return config

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