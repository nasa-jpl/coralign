"""Unit tests for combine_crop.py."""

import unittest

import numpy as np

from coralign.util.get_crop_lower import get_crop_lower


class TestGetCropLower(unittest.TestCase):
    """
    Unit tests for cropping math
    """

    def test_success(self):
        """Good inputs complete without issue"""
        get_crop_lower(5, 5, 3, 3)
        pass


    def test_correct_sizing(self):
        """
        Verify sizing is done correctly.  Use the following as examples to give
        lower_rol/col = 0:
         0 0 X 0 0 (center = 2, width = 5)
         0 0 X 0 (center = 2, width = 4)
        """

        target = (0, 0)

        self.assertTrue(get_crop_lower(2, 2, 5, 5) == target)
        self.assertTrue(get_crop_lower(2, 2, 5, 4) == target)
        self.assertTrue(get_crop_lower(2, 2, 4, 5) == target)
        self.assertTrue(get_crop_lower(2, 2, 4, 4) == target)

        pass


    def test_bound_neg(self):
        """
        Verify bounding is done correctly.  Use center = 1, width = 5 and
        center = 1, width = 4, which should produce (-1, -1).
        """

        # different if bounded or not
        target = (-1, -1)
        btarget = (0, 0)

        self.assertTrue(get_crop_lower(1, 1, 5, 5, bound=False) == target)
        self.assertTrue(get_crop_lower(1, 1, 5, 4, bound=False) == target)
        self.assertTrue(get_crop_lower(1, 1, 4, 5, bound=False) == target)
        self.assertTrue(get_crop_lower(1, 1, 4, 4, bound=False) == target)

        self.assertTrue(get_crop_lower(1, 1, 5, 5, bound=True) == btarget)
        self.assertTrue(get_crop_lower(1, 1, 5, 4, bound=True) == btarget)
        self.assertTrue(get_crop_lower(1, 1, 4, 5, bound=True) == btarget)
        self.assertTrue(get_crop_lower(1, 1, 4, 4, bound=True) == btarget)

        pass


    def test_bound_pos(self):
        """
        Verify bounding is done correctly.  Use center = 3, width = 5 and
        center = 3, width = 4, which should produce (1, 1).
        """

        # bounding does not matter when not at edge
        target = (1, 1)

        self.assertTrue(get_crop_lower(3, 3, 5, 5, bound=False) == target)
        self.assertTrue(get_crop_lower(3, 3, 5, 4, bound=False) == target)
        self.assertTrue(get_crop_lower(3, 3, 4, 5, bound=False) == target)
        self.assertTrue(get_crop_lower(3, 3, 4, 4, bound=False) == target)

        self.assertTrue(get_crop_lower(3, 3, 5, 5, bound=True) == target)
        self.assertTrue(get_crop_lower(3, 3, 5, 4, bound=True) == target)
        self.assertTrue(get_crop_lower(3, 3, 4, 5, bound=True) == target)
        self.assertTrue(get_crop_lower(3, 3, 4, 4, bound=True) == target)

        pass


    def test_bound_default(self):
        """verify bound default is as expected"""
        # center = 1, width = 5 gives lower = -1

        lower_false = get_crop_lower(1, 1, 5, 5, bound=False)
        lower_true = get_crop_lower(1, 1, 5, 5, bound=True)
        lower_default = get_crop_lower(1, 1, 5, 5)

        self.assertTrue(lower_default == lower_false)
        self.assertFalse(lower_default == lower_true)

        pass


    def test_invalid_row_center(self):
        """Invalid inputs caught"""

        xlist = [-1, 1.5, None, 'txt', (5,), np.ones((5,))]

        for x in xlist:
            with self.assertRaises(TypeError):
                get_crop_lower(x, 5, 3, 3)
            pass
        pass


    def test_invalid_col_center(self):
        """Invalid inputs caught"""

        xlist = [-1, 1.5, None, 'txt', (5,), np.ones((5,))]

        for x in xlist:
            with self.assertRaises(TypeError):
                get_crop_lower(5, x, 3, 3)
            pass
        pass


    def test_invalid_row_width(self):
        """Invalid inputs caught"""

        xlist = [-1, 0, 1.5, None, 'txt', (5,), np.ones((5,))]

        for x in xlist:
            with self.assertRaises(TypeError):
                get_crop_lower(5, 5, x, 3)
            pass
        pass


    def test_invalid_col_width(self):
        """Invalid inputs caught"""

        xlist = [-1, 0, 1.5, None, 'txt', (5,), np.ones((5,))]

        for x in xlist:
            with self.assertRaises(TypeError):
                get_crop_lower(5, 5, 3, x)
            pass
        pass


    def test_invalid_bound(self):
        """Invalid inputs caught"""

        xlist = [-1, 0, 1.5, None, 'txt', (5,), np.ones((5,))]

        for x in xlist:
            with self.assertRaises(TypeError):
                get_crop_lower(5, 5, 3, 3, bound=x)
            pass
        pass




if __name__ == '__main__':
    unittest.main()
