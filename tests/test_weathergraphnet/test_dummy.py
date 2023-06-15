"""
This module contains a unit test for the dummy function in the weathergraphnet package.

The TestDummy class inherits from the unittest.TestCase class and contains a single test
method that checks if 1 equals 1.

Usage:
    This module can be run as a standalone script to execute the unit test.

"""
# Standard library
import unittest


class TestDummy(unittest.TestCase):
    def test_dummy(self):
        self.assertEqual(1, 1)


if __name__ == "__main__":
    unittest.main()
