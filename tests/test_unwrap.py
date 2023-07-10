"""
Test class for the function unwrap in unwrap.py
"""

import unittest

import numpy as np
from scipy.ndimage import binary_erosion

from coralign.util.unwrap import unwrap, unwrap_segments


class TestUnwrap(unittest.TestCase):
    """
    test inputs
    """

    def setUp(self):
        """
        Predefine common variable inputs available for all tests
        """
        # create a wrapped phase image and an amplitude mask
        nx = 256
        w0 = 32 # pixels
        x = np.arange(nx) - nx//2
        X, Y = np.meshgrid(x, x)
        R = np.hypot(X, Y)

        # creat amp and a threshold/mask that ampthresh, called in unwrap, is
        # sure to find
        self.amp = np.exp(-(R/w0)**4)
        self.bMask = self.amp >= 0.2
        self.amp[np.logical_not(self.bMask)] = 0.0

        self.phase_unwrapped = -(np.pi + 0.1) + (2.5*np.pi)*np.exp(-(R/w0)**2)
        self.phase = np.angle(np.exp(1j*self.phase_unwrapped))

        self.not_twoD_array = [0, 1.0, 1.0+1j, np.ones((5,))]
        self.not_positive_scalar_integer = [0, -1.2, 1.2, 1j, 'abc',
                                            np.ones((5,))]
        self.not_real_scalar = [1j, 'abc', np.ones((5,))]
        self.not_bool = ['True', 3j, None]

    def test_invalid_amp(self):
        """ invalid input amplitude image raises error """
        for perr in self.not_twoD_array:
            with self.assertRaises(TypeError):
                unwrap(self.phase, perr)

        # test amplitude is real
        with self.assertRaises(TypeError):
            unwrap(self.phase, 1j*self.amp)

        # test amplitude >= 0
        with self.assertRaises(TypeError):
            unwrap(self.phase, self.amp - 1.0)

    def test_invalid_pha(self):
        """ invalid input phase image raises error """
        for perr in self.not_twoD_array:
            with self.assertRaises(TypeError):
                unwrap(perr, self.amp)

        # test phase is real
        atmp = np.zeros(self.phase.shape, dtype='complex')
        atmp[0, 0] = 1.0j
        with self.assertRaises(TypeError):
            unwrap(self.phase+atmp, self.amp)

    def test_invalid_nbin(self):
        """ invalid input nbin raises error """
        for perr in self.not_positive_scalar_integer:
            with self.assertRaises(TypeError):
                unwrap(self.phase, self.amp, nbin=perr)

        # test nbin must be >= 3
        with self.assertRaises(ValueError):
            unwrap(self.phase, self.amp, nbin=2)

    def test_invalid_fill_value(self):
        """ invalid input fill_value raises error """
        for perr in self.not_real_scalar:
            with self.assertRaises(TypeError):
                unwrap(self.phase, self.amp, fill_value=perr)


    def test_invalid_min_size(self):
        """ invalid input min_size raises error """
        for perr in self.not_positive_scalar_integer:
            with self.assertRaises(TypeError):
                unwrap(self.phase, self.amp, min_size=perr)


    def test_invalid_use_mask(self):
        """ invalid input use_mask raises error """
        for perr in self.not_bool:
            with self.assertRaises(TypeError):
                unwrap(self.phase, self.amp, use_mask=perr)


    def test_success(self):
        """ example that works and check unwrapped phase against original """

        phase_ret, bMask_ret = unwrap(self.phase, self.amp)

        self.assertTrue(np.all(bMask_ret == self.bMask))
        self.assertTrue(np.max(np.abs(
            phase_ret[bMask_ret] - self.phase_unwrapped[bMask_ret]
        )) < 1.e-15*np.pi)

class TestUnwrap_segments(unittest.TestCase):
    """
    test inputs
    """

    def setUp(self):
        """
        Predefine common variable inputs available for all tests
        """
        # create a wrapped phase image and an amplitude mask
        nx = 256
        w0 = 32 # pixels
        x = np.arange(nx) - nx//2
        X, Y = np.meshgrid(x, x)
        R = np.hypot(X, Y)

        # creat amp and a threshold/mask that ampthresh, called in unwrap, is
        # sure to find
        self.amp = np.exp(-(R/w0)**4)
        self.bMask = self.amp >= 0.2

        # divide the pupil mask into disjoint segments
        self.bMask[np.logical_and(X > -5, X < 5)] = False

        #
        self.amp[np.logical_not(self.bMask)] = 0.0

        self.phase_unwrapped = self.bMask*(-(np.pi + 0.1) + (2.5*np.pi)*np.exp(-(R/w0)**2))
        self.phase = self.bMask*np.angle(np.exp(1j*self.phase_unwrapped))

        self.not_twoD_array = [0, 1.0, 1.0+1j, np.ones((5,))]
        self.not_positive_scalar_integer = [0, -1.2, 1.2, 1j, 'abc',
                                            np.ones((5,))]
        self.not_real_scalar = [1j, 'abc', np.ones((5,))]
        self.not_bool = ['True', 3j, None]

    def test_invalid_amp(self):
        """ invalid input amplitude image raises error """
        for perr in self.not_twoD_array:
            with self.assertRaises(TypeError):
                unwrap_segments(self.phase, perr)

        # test amplitude is real
        with self.assertRaises(TypeError):
            unwrap_segments(self.phase, 1j*self.amp)

        # test amplitude >= 0
        with self.assertRaises(TypeError):
            unwrap_segments(self.phase, self.amp - 1.0)

    def test_invalid_pha(self):
        """ invalid input phase image raises error """
        for perr in self.not_twoD_array:
            with self.assertRaises(TypeError):
                unwrap_segments(perr, self.amp)

        # test phase is real
        atmp = np.zeros(self.phase.shape, dtype='complex')
        atmp[0, 0] = 1.0j
        with self.assertRaises(TypeError):
            unwrap_segments(self.phase+atmp, self.amp)

    def test_invalid_nbin(self):
        """ invalid input nbin raises error """
        for perr in self.not_positive_scalar_integer:
            with self.assertRaises(TypeError):
                unwrap_segments(self.phase, self.amp, nbin=perr)

        # test nbin must be >= 3
        with self.assertRaises(ValueError):
            unwrap_segments(self.phase, self.amp, nbin=2)

    def test_invalid_fill_value(self):
        """ invalid input fill_value raises error """
        for perr in self.not_real_scalar:
            with self.assertRaises(TypeError):
                unwrap_segments(self.phase, self.amp, fill_value=perr)


    def test_invalid_min_size(self):
        """ invalid input min_size raises error """
        for perr in self.not_positive_scalar_integer:
            with self.assertRaises(TypeError):
                unwrap_segments(self.phase, self.amp, min_size=perr)


    def test_invalid_use_mask(self):
        """ invalid input use_mask raises error """
        for perr in self.not_bool:
            with self.assertRaises(TypeError):
                unwrap_segments(self.phase, self.amp, use_mask=perr)


    def test_success(self):
        """ example that works and check unwrapped phase against original """

        # execute unwrap_segments on test data
        phase_ret, bMask_ret = unwrap_segments(self.phase, self.amp)

        # bMask_ret is not the same as self.bMask because of the erosion
        # test returned mask against erosion version of self.bMask
        bMaskErode = binary_erosion(self.bMask, structure=np.array([
            [0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]
            ]))

        self.assertTrue(np.all(bMask_ret == bMaskErode))

        # test unwrapped phase within the masked segments
        self.assertTrue(np.max(np.abs(
            phase_ret[bMask_ret] - self.phase_unwrapped[bMask_ret]
        )) < 1.e-15*np.pi)


if __name__ == '__main__':
    unittest.main()
