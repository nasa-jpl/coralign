"""
Unit tests for generic YAML loading
"""

import unittest
import os

from coralign.util.loadyaml import loadyaml


class TestOnlyException(Exception):
    """Exception to be used below for the custom_exception input"""
    pass

class TestLoadYAML(unittest.TestCase):
    """
    Test successful and failed loads
    Test default exception behavior
    """

    def test_good_input(self):
        """
        Verify a valid input loads successfully
        """
        localpath = os.path.dirname(os.path.abspath(__file__))
        fn = os.path.join(localpath, 'testdata', 'ut_valid.yaml')
        loadyaml(fn)
        pass


    def test_missing_file(self):
        """
        Fail when input file is missing
        """
        fn = 'does_not_exist'

        with self.assertRaises(TestOnlyException):
            loadyaml(fn, custom_exception=TestOnlyException)
            pass
        pass


if __name__ == '__main__':
    unittest.main()