"""
pytest configuration: mock Streamlit before any module imports it.

utils.py runs `if "OPENAI_API_KEY" in st.secrets` at module level, which
raises an error outside a running Streamlit server.  Patching sys.modules
here (in pytest_configure, which fires before collection) keeps all tests
independent of a Streamlit runtime.
"""
import sys
from unittest.mock import MagicMock


def pytest_configure(config):
    st_mock = MagicMock()
    st_mock.secrets = {}          # empty dict — no key found, no env var set
    sys.modules["streamlit"] = st_mock
