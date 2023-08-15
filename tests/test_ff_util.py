"""Unit test suite for ff_util."""
import unittest
import numpy as np
from math import isclose

import coralign.focalfit.ff_util as ffu


class TestFfutilInputFailure(unittest.TestCase):
    """Test suite for valid function inputs."""

    def test_inputs_of_calc_offset_quadratic(self):
        """Test the inputs of calc_offset_quadratic."""
        fitCoefPow1 = 1.5
        fitCoefPow2 = 2.1
        roiSumRatio = 0.9

        with self.assertRaises(ValueError):
            for fitCoefPow1Bad in (1j, 2-1j, (1,), 'asdf'):
                ffu.calc_offset_quadratic(fitCoefPow1Bad,
                                          fitCoefPow2,
                                          roiSumRatio)
        with self.assertRaises(ValueError):
            for fitCoefPow2Bad in (1j, 2-1j, (1,), 'asdf'):
                ffu.calc_offset_quadratic(fitCoefPow1,
                                          fitCoefPow2Bad,
                                          roiSumRatio)
        with self.assertRaises(ValueError):
            for roiSumRatioBad in (-1, 0, 1j, 2-1j, (1,), 'asdf'):
                ffu.calc_offset_quadratic(fitCoefPow1,
                                          fitCoefPow2,
                                          roiSumRatioBad)

    def test_inputs_of_calc_offset_linear(self):
        """Test the inputs of calc_offset_linear."""
        fitCoefPow1 = 1.5
        roiSumRatio = 0.9

        with self.assertRaises(ValueError):
            for fitCoefPow1Bad in (1j, 2-1j, (1,), 'asdf'):
                ffu.calc_offset_linear(fitCoefPow1Bad, roiSumRatio)
        with self.assertRaises(ValueError):
            for roiSumRatioBad in (-1, 0, 1j, 2-1j, (1,), 'asdf'):
                ffu.calc_offset_linear(fitCoefPow1, roiSumRatioBad)

    def test_inputs_of_bound_value(self):
        """Test the inputs of bound_value."""
        # Check standard inputs do not raise anything first
        inVal = 4.5
        maxVal = 6
        ffu.bound_value(inVal, maxVal)

        with self.assertRaises(ValueError):
            for inValBad in (1j, 2-1j, (1,), 'asdf'):
                ffu.bound_value(inValBad, maxVal)

        with self.assertRaises(ValueError):
            for maxValBad in (-1, 1j, 2-1j, (1,), 'asdf'):
                ffu.bound_value(inVal, maxValBad)


class TestQuadraticOffset(unittest.TestCase):
    """Unit tests for calc_offset_quadratic."""

    def test_positive_offset(self):
        """Test calc_offset_quadratic for a negative offset."""
        # Equations of form y = a*x*x + b*x + 1
        a = 3.
        b = 5.
        roiSumRatio = 23.
        xOffset = ffu.calc_offset_quadratic(b, a, roiSumRatio)

        self.assertTrue(isclose(xOffset, 2.0, abs_tol=2*np.finfo(float).eps))

    def test_negative_offset(self):
        """Test calc_offset_quadratic for a negative offset."""
        a = 3.
        b = 5.
        roiSumRatio = 1/23.
        xOffset = ffu.calc_offset_quadratic(b, a, roiSumRatio)

        self.assertTrue(isclose(xOffset, -2.0, abs_tol=2*np.finfo(float).eps))

    def test_no_offset(self):
        """Test calc_offset_quadratic for no offset."""
        a2 = 3
        a1 = 2
        roiSumRatio = 1.0
        xOffset = ffu.calc_offset_quadratic(a1, a2, roiSumRatio)

        self.assertTrue(isclose(xOffset, 0, abs_tol=2*np.finfo(float).eps))


class TestLinearOffset(unittest.TestCase):
    """Unit tests for calc_offset_linear."""

    def test_positive_offset(self):
        """Test calc_offset_linear for a negative offset."""
        # Equations of form y = slope*x + 1
        slope = 2.0
        roiSumRatio = 7.0
        xOffset = ffu.calc_offset_linear(slope, roiSumRatio)

        self.assertTrue(isclose(xOffset, 3.0, abs_tol=2*np.finfo(float).eps))

    def test_negative_offset(self):
        """Test calc_offset_linear for a negative offset."""
        slope = 2.0
        roiSumRatio = 3/4
        xOffset = ffu.calc_offset_linear(slope, roiSumRatio)

        self.assertTrue(isclose(xOffset, -1/6, abs_tol=2*np.finfo(float).eps))

    def test_no_offset(self):
        """Test calc_offset_linear for no offset."""
        slope = 2.0
        roiSumRatio = 1.0
        xOffset = ffu.calc_offset_linear(slope, roiSumRatio)

        self.assertTrue(isclose(xOffset, 0, abs_tol=2*np.finfo(float).eps))


class TestBoundValue(unittest.TestCase):
    """Unit tests for bound_value."""

    def test_bound_value_no_change(self):
        """Test the functionality of bound_value."""
        inVal = 4.5
        maxVal = 6
        outVal = ffu.bound_value(inVal, maxVal)

        self.assertTrue(outVal == inVal,
                        msg=('Input value should not change.'))

    def test_bound_value_positive(self):
        """Test the functionality of bound_value."""
        inVal = 10.5
        maxVal = 6.1
        outVal = ffu.bound_value(inVal, maxVal)

        self.assertTrue(outVal == maxVal)

    def test_bound_value_negative(self):
        """Test the functionality of bound_value."""
        inVal = -10.5
        maxVal = 6.1
        outVal = ffu.bound_value(inVal, maxVal)

        self.assertTrue(outVal == -maxVal)


if __name__ == '__main__':
    unittest.main()
