"""Unit tests for combine_crop.py."""

import unittest

import numpy as np

from coralign.util.combine_crop import combine, crop


class TestCombine(unittest.TestCase):
    """
    Unit tests for combining frames with bad pixel maps
    """

    def setUp(self):
        self.frame = np.reshape(np.arange(25), (5, 5))
        self.bp = np.zeros((5, 5)).astype('bool')

        inds = np.array([0, 1, 4, 9, 16])
        self.bp.ravel()[inds] = True

        pass


    def test_success(self):
        """good input completes successfully"""
        combine(self.frame, self.bp)
        pass


    def test_success_clean(self):
        """succeeds with clean-frame-sized inputs"""
        clean = 1024
        combine(np.eye(clean), np.ones((clean, clean)).astype('bool'))
        pass


    def test_invalid_frame(self):
        """invalid inputs caught"""

        xlist = [1, 0, -1.5, None, 'txt', (5,)]

        for x in xlist:
            with self.assertRaises(TypeError):
                combine(x, self.bp)
            pass

        # check other size matrices; make sure frame and bp sizes match
        f = np.ones((5,))
        with self.assertRaises(TypeError):
            combine(f, np.zeros_like(f).astype('bool'))
        f = np.ones((5, 5, 2))
        with self.assertRaises(TypeError):
            combine(f, np.zeros_like(f).astype('bool'))

        pass


    def test_invalid_bp(self):
        """invalid inputs caught"""

        xlist = [1, 0, -1.5, None, 'txt', (5,)]

        for x in xlist:
            with self.assertRaises(TypeError):
                combine(self.frame, x)
            pass

        # check other size matrices; make sure frame and bp sizes match
        b = np.ones((5,)).astype('bool')
        with self.assertRaises(TypeError):
            combine(np.zeros_like(b), b)
        b = np.ones((5, 5, 2)).astype('bool')
        with self.assertRaises(TypeError):
            combine(np.zeros_like(b), b)
        pass


    def test_frame_and_bp_not_same_size(self):
        """invalid inputs caught"""
        with self.assertRaises(TypeError):
            combine(self.frame, self.bp[:-1, :-1])
        pass


    def test_bp_not_boolean(self):
        """invalid inputs caught"""
        with self.assertRaises(TypeError):
            combine(self.frame, np.zeros_like(self.frame).astype('float'))
        pass


    def test_exact(self):
        """exact result as expected"""

        frame = np.reshape(np.arange(25), (5, 5))
        bp = np.zeros((5, 5)).astype('bool')

        inds = np.array([0, 1, 4, 9, 16])
        bp.ravel()[inds] = True

        out = combine(frame, bp)

        self.assertTrue(np.isnan(out[bp]).all())
        self.assertTrue((out[~bp] == frame[~bp]).all())

        pass


class TestCrop(unittest.TestCase):
    """
    Unit tests for frame cropping
    """

    def setUp(self):
        self.frame = np.reshape(np.arange(25), (5, 5))
        self.lr = 1
        self.lc = 1
        self.rw = 3
        self.cw = 3
        pass


    def test_success(self):
        """good inputs run without error"""
        crop(self.frame, self.lr, self.lc, self.rw, self.cw)
        pass


    def test_exact(self):
        """exact values recovered as expected"""

        target = np.array([[6, 7, 8],
                           [11, 12, 13],
                           [16, 17, 18]]).astype('float')

        out = crop(self.frame, self.lr, self.lc, self.rw, self.cw)
        self.assertTrue((out == target).all())
        pass


    def test_exact_offset(self):
        """exact valued recovered as expected even with padding"""
        frame = np.reshape(np.arange(25), (5, 5))
        rw = 3
        cw = 3
        lr = 3
        lc = 3

        target = np.array([[18, 19, np.nan],
                           [23, 24, np.nan],
                           [np.nan, np.nan, np.nan]])
        out = crop(frame, lr, lc, rw, cw)

        self.assertTrue((np.isnan(out) == np.isnan(target)).all())
        self.assertTrue((out[~np.isnan(out)] == target[~np.isnan(out)]).all())

        pass


    def test_invalid_frame(self):
        """Invalid inputs caught"""

        xlist = [1, 0, -1.5, None, 'txt', (5,)]

        for x in xlist:
            with self.assertRaises(TypeError):
                crop(x, self.lr, self.lc, self.rw, self.cw)
            pass
        pass


    def test_invalid_lower_row(self):
        """Invalid inputs caught"""

        xlist = [-1, 1.5, None, 'txt', (5,), np.ones((5,))]

        for x in xlist:
            with self.assertRaises(TypeError):
                crop(self.frame, x, self.lc, self.rw, self.cw)
            pass
        pass


    def test_invalid_lower_col(self):
        """Invalid inputs caught"""

        xlist = [-1, 1.5, None, 'txt', (5,), np.ones((5,))]

        for x in xlist:
            with self.assertRaises(TypeError):
                crop(self.frame, self.lr, x, self.rw, self.cw)
            pass
        pass


    def test_invalid_row_width(self):
        """Invalid inputs caught"""

        xlist = [-1, 0, 1.5, None, 'txt', (5,), np.ones((5,))]

        for x in xlist:
            with self.assertRaises(TypeError):
                crop(self.frame, self.lr, self.lc, x, self.cw)
            pass
        pass


    def test_invalid_col_width(self):
        """Invalid inputs caught"""

        xlist = [-1, 0, 1.5, None, 'txt', (5,), np.ones((5,))]

        for x in xlist:
            with self.assertRaises(TypeError):
                crop(self.frame, self.lr, self.lc, self.rw, x)
            pass
        pass


    def test_lower_vals_out_of_range(self):
        """out-of-range values caught"""
        # first failure when corner = shape
        with self.assertRaises(ValueError):
            crop(self.frame, self.frame.shape[0], self.lc, self.rw, self.cw)
        with self.assertRaises(ValueError):
            crop(self.frame, self.lr, self.frame.shape[1], self.rw, self.cw)

        pass








if __name__ == '__main__':
    unittest.main()
