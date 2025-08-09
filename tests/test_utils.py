# Add src to path to allow imports
import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import _replace_env_vars, drop_constant_columns


def test_replace_env_vars_handles_both_styles(monkeypatch):
    monkeypatch.setenv('TEST_VAR', 'value')
    original = 'path:${TEST_VAR}/$TEST_VAR'
    replaced = _replace_env_vars(original)
    assert replaced == 'path:value/value'


def test_drop_constant_columns():
    df = pd.DataFrame({'a': [1, 1, 1], 'b': [1, 2, 3]})
    cleaned, dropped = drop_constant_columns(df, ['a', 'b'])
    assert dropped == ['a']
    assert list(cleaned.columns) == ['b']
