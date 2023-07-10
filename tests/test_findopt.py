"""Unit tests for the findopt.py utility functions."""
import unittest
import numpy as np
from math import isclose

from coralign.util.findopt import find_optimum_1d, find_optimum_2d, QuadraticException


class TestFindoptFunctions(unittest.TestCase):
    """Test all normal use cases."""
    
    def test_find_optimum_1d(self):
        """Test that a true parabola's center is found exactly."""
        xTrue = 4.87654
        
        delta = 1.0
        xVec = np.arange(-4, 5, delta)
        arrayToFit = -(xVec - xTrue)**2
        
        xEst = find_optimum_1d(xVec, arrayToFit)
        self.assertTrue(isclose(xTrue, xEst, abs_tol=1e-5))
        
    def test_find_optimum_1d_on_line(self):
        """Test that a line fails to return an optimum."""
        nPoints = 10
        xVec = np.arange(nPoints)
        arrayToFit = xVec
        
        with self.assertRaises(QuadraticException):
            find_optimum_1d(xVec, arrayToFit)
            
    def test_find_optimum_1d_on_wrong_input_sizes(self):
        """Test that an exception is thrown when the input sizes mismatch."""
        nPoints = 10
        xVec = np.arange(nPoints)
        arrayToFit = np.arange(nPoints+1)
        
        with self.assertRaises(ValueError):
            find_optimum_1d(xVec, arrayToFit)
        
    def test_find_optimum_2d(self):
        """Test that a true paraboloid's center is found exactly."""
        xTrue = 4.87654
        yTrue = 3.24689
        
        delta = 1.0
        xVec = np.arange(-4, 5, delta)
        yVec = np.arange(-5, 3, delta)
        X, Y = np.meshgrid(xVec, yVec)
        arrayToFit = (X - xTrue)**2 + (Y - yTrue)**2
        
        # Mask is the whole array
        mask = np.ones_like(arrayToFit)
        xEst, yEst = find_optimum_2d(xVec, yVec, arrayToFit,
                                                    mask)
        self.assertTrue(isclose(xTrue, xEst, abs_tol=1e-5))
        self.assertTrue(isclose(yTrue, yEst, abs_tol=1e-5))
        
        # Mask is subset of the array
        mask = np.zeros_like(arrayToFit, dtype=bool)
        mask[::2, ::2] = True
        xEst, yEst = find_optimum_2d(xVec, yVec, arrayToFit,
                                                    mask)
        self.assertTrue(isclose(xTrue, xEst, abs_tol=1e-5))
        self.assertTrue(isclose(yTrue, yEst, abs_tol=1e-5))

    def test_find_optimum_2d_failure_x(self):
        """Test the function fails if the x-axis lacks a quadratic part."""
        nPoints = 10
        xVec = np.arange(nPoints)
        yVec = np.arange(nPoints)
        X, Y = np.meshgrid(xVec, yVec)
        arrayToFit = Y**2
        mask = np.ones_like(arrayToFit)
        
        with self.assertRaises(QuadraticException):
            find_optimum_2d(xVec, yVec, arrayToFit, mask)
            
    def test_find_optimum_2d_failure_y(self):
        """Test the function fails if the y-axis lacks a quadratic part."""
        nPoints = 10
        xVec = np.arange(nPoints)
        yVec = np.arange(nPoints)
        X, Y = np.meshgrid(xVec, yVec)
        arrayToFit = X**2
        mask = np.ones_like(arrayToFit)
        
        with self.assertRaises(QuadraticException):
            find_optimum_2d(xVec, yVec, arrayToFit, mask)
        

class TestInputFailure(unittest.TestCase):
    """Test suite for valid function inputs."""
    
    def test_find_optimum_1d_inputs(self):
        """Test the inputs of find_optimum_1d."""
        xVec = np.arange(-2., 5., 1)
        arrayToFit = xVec*xVec
        
        # Check standard inputs do not raise anything first
        find_optimum_1d(xVec, arrayToFit)

        for xVecBad in (1, 1.1, 1j, np.ones((5, 10)), 'string'):
            with self.assertRaises(ValueError):
                find_optimum_1d(xVecBad, arrayToFit)
                
        for arrayToFitBad in (1, 1.1, 1j, np.ones((5, 10)), 'string'):
            with self.assertRaises(ValueError):
                find_optimum_1d(xVec, arrayToFitBad)

    def test_find_optimum_2d_inputs(self):
        """Test the inputs of parabolic_fit_to_matrix."""
        xVec = np.arange(-2., 5., 1)
        yVec = np.arange(-6., 3., 1)
        [X, Y] = np.meshgrid(xVec, yVec)
        arrayToFit = X*X + Y*Y
        mask = np.ones_like(arrayToFit)
        
        # Check standard inputs do not raise anything first
        find_optimum_2d(xVec, yVec, arrayToFit, mask)

        # Check inputs
        for xVecBad in (1, 1.1, 1j, np.ones((5, 10)), 'string'):
            with self.assertRaises(ValueError):
                find_optimum_2d(xVecBad, yVec, arrayToFit, mask)
        for yVecBad in (1, 1.1, 1j, np.ones((5, 10)), 'string'):
            with self.assertRaises(ValueError):
                find_optimum_2d(xVec, yVecBad, arrayToFit, mask)
        for arrayToFitBad in (1, 1.1, 1j, np.ones(5), np.ones((3, 3, 3)),
                              'string'):
            with self.assertRaises(ValueError):
                find_optimum_2d(xVec, yVec, arrayToFitBad, mask)
        for maskBad in (1, 1.1, 1j, np.ones(5), np.ones((3, 3, 3)),
                              'string'):
            with self.assertRaises(ValueError):
                find_optimum_2d(xVec, yVec, arrayToFit, maskBad)
                
        # Check shape equality
        with self.assertRaises(ValueError):
            find_optimum_2d(xVec, yVec, arrayToFit, maskBad)
        with self.assertRaises(ValueError):
            find_optimum_2d(np.arange(0, 100), yVec,
                                           arrayToFit, maskBad)
        with self.assertRaises(ValueError):
            find_optimum_2d(xVec, np.arange(0, 100),
                                           arrayToFit, maskBad)
                
            
if __name__ == '__main__':
    unittest.main()
