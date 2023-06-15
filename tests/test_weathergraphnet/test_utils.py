"""Test module ``weathergraphnet/utils.py``."""
# Standard library
import logging

# First-party
from weathergraphnet.utils import count_to_log_level


def test_count_to_log_level():
    assert count_to_log_level(0) == logging.ERROR
    assert count_to_log_level(1) == logging.WARNING
    assert count_to_log_level(2) == logging.INFO
    assert count_to_log_level(3) == logging.DEBUG
