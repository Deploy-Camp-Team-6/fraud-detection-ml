# Add src to path to allow imports
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import _replace_env_vars


def test_replace_env_vars_handles_both_styles(monkeypatch):
    monkeypatch.setenv('TEST_VAR', 'value')
    original = 'path:${TEST_VAR}/$TEST_VAR'
    replaced = _replace_env_vars(original)
    assert replaced == 'path:value/value'
