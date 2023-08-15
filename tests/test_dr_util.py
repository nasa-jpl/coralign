"""Unit tests for the DMREG utility functions."""
import unittest
import numpy as np
from math import isclose

from coralign.dmreg.dr_util import remove_piston_tip_tilt


class TestDmregSupportFunctions(unittest.TestCase):
    """Test all the supporting methods of maskgen."""

    def test_remove_piston_tip_tilt(self):
        """Test that piston, tip, and tilt are fully removed."""
        N = 100
        x = np.arange(-N/2, N/2)/N
        [X, Y] = np.meshgrid(x, x)

        arrayIn = 2*X - 3*Y + 0.8
        mask = np.ones_like(X)
        mask[X*X+Y*Y > 0.5**2] = 0

        arrayOut = remove_piston_tip_tilt(arrayIn, mask)

        self.assertTrue(isclose(np.mean(arrayOut), 0, abs_tol=1e-12))


class TestInputFailure(unittest.TestCase):
    """Test suite for valid function inputs."""

    def test_remove_piston_tip_tilt_inputs(self):
        """Test the inputs of remove_piston_tip_tilt."""
        arrayToFit = np.ones((10, 11))
        mask = np.ones((10, 11))

        for arrayToFitBad in (1, 1.1, 1j, np.ones(5), np.ones((3, 3, 3)),
                              'string'):
            with self.assertRaises(ValueError):
                remove_piston_tip_tilt(arrayToFitBad, mask)
        for maskBad in (1, 1.1, 1j, np.ones(5), np.ones((3, 3, 3)), 'string'):
            with self.assertRaises(ValueError):
                remove_piston_tip_tilt(arrayToFit, maskBad)

        # Check shape equality
        with self.assertRaises(ValueError):
            remove_piston_tip_tilt(np.ones((10, 10)), np.ones((11, 12)))


if __name__ == '__main__':
    unittest.main()
