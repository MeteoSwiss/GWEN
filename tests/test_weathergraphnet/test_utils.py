import logging
import unittest

from weathergraphnet.utils import count_to_log_level


class TestUtils(unittest.TestCase):
    """
    Unit tests for the utils.py module
    """

    def test_count_to_log_level(self):
        """
        Test that the count_to_log_level function maps the verbose count to the correct log level
        """
        # Test that count 0 maps to logging.ERROR
        self.assertEqual(count_to_log_level(0), logging.ERROR)

        # Test that count 1 maps to logging.WARNING
        self.assertEqual(count_to_log_level(1), logging.WARNING)

        # Test that count 2 maps to logging.INFO
        self.assertEqual(count_to_log_level(2), logging.INFO)

        # Test that count 3 maps to logging.DEBUG
        self.assertEqual(count_to_log_level(3), logging.DEBUG)

        # Test that count 4 maps to logging.DEBUG
        self.assertEqual(count_to_log_level(4), logging.DEBUG)


if __name__ == '__main__':
    unittest.main()
