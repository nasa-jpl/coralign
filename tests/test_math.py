"""Unit tests for the findopt.py utility functions."""
import unittest
from math import isclose

import numpy as np

from coralign.util.math import rms, ceil_odd, ceil_even, cart2pol


class TestFindoptFunctions(unittest.TestCase):
    """Test all normal use cases."""

    def test_rms(self):
        """Test that rms returns the root-mean-square of an array."""
        self.assertTrue(isclose(np.sqrt(2)/2., rms(np.sin(2*np.pi
                        * np.linspace(0., 1., 10001))), rel_tol=0.01))

        array = np.ones((10, 10))
        array[:, ::2] = -1
        self.assertTrue(isclose(1., rms(array)))

    def test_ceil_odd(self):
        """Test that ceil_even returns the next highest even integer."""
        self.assertTrue(ceil_odd(1) == 1)
        self.assertTrue(ceil_odd(1.5) == 3)
        self.assertTrue(ceil_odd(4) == 5)
        self.assertTrue(ceil_odd(3.14159) == 5)
        self.assertTrue(ceil_odd(2001) == 2001)

    def test_ceil_even(self):
        """Test that ceil_even returns the next highest even integer."""
        self.assertTrue(ceil_even(1) == 2)
        self.assertTrue(ceil_even(1.5) == 2)
        self.assertTrue(ceil_even(4) == 4)
        self.assertTrue(ceil_even(3.14159) == 4)
        self.assertTrue(ceil_even(2001) == 2002)


class TestInputFailure(unittest.TestCase):
    """Test suite for valid function inputs."""

    def test_ceil_odd_input(self):
        """Test incorrect inputs of ceil_odd."""
        with self.assertRaises(TypeError):
            ceil_odd('this is a string')
        with self.assertRaises(TypeError):
            ceil_odd(np.array([2.0, 3.1]))

    def test_ceil_even_input(self):
        """Test incorrect inputs of ceil_even."""
        with self.assertRaises(TypeError):
            ceil_even('this is a string')
        with self.assertRaises(TypeError):
            ceil_even(np.array([2.0, 3.1]))

    def test_rms_input(self):
        """Test incorrect inputs of rms."""
        values = ('a')
        for val in values:
            with self.assertRaises(TypeError):
                rms(val)


class TestCart2Pol(unittest.TestCase):
    """Test suite for testing correct Cartesian to polar
    coordinate conversions."""

    def test_x_zero_y_zero(self):
        """Test inputs for cart2pol."""
        x = 0
        y = 0
        rho, theta = cart2pol(x, y)
        abs_tol = 10*np.finfo(float).eps
        self.assertTrue(isclose(rho, 0, abs_tol=abs_tol))
        self.assertTrue(isclose(theta, 0, abs_tol=abs_tol))

    def test_x_zero_y_pos(self):
        """Test inputs for cart2pol."""
        x = 0
        y = 1.89
        rho, theta = cart2pol(x, y)
        abs_tol = 10*np.finfo(float).eps
        self.assertTrue(isclose(rho, y, abs_tol=abs_tol))
        self.assertTrue(isclose(theta, np.pi / 2, abs_tol=abs_tol))

    def test_x_zero_y_neg(self):
        """Test inputs for cart2pol."""
        x = 0
        y = -0.234
        rho, theta = cart2pol(x, y)
        abs_tol = 10*np.finfo(float).eps
        self.assertTrue(isclose(rho, np.abs(y), abs_tol=abs_tol))
        self.assertTrue(isclose(theta, - np.pi / 2, abs_tol=abs_tol))

    def test_x_pos_y_zero(self):
        """Test inputs for cart2pol."""
        x = 1.27
        y = 0
        rho, theta = cart2pol(x, y)
        abs_tol = 10*np.finfo(float).eps
        self.assertTrue(isclose(rho, x, abs_tol=abs_tol))
        self.assertTrue(isclose(theta, 0, abs_tol=abs_tol))

    def test_x_neg_y_zero(self):
        """Test inputs for cart2pol."""
        x = -23.5
        y = 0
        rho, theta = cart2pol(x, y)
        abs_tol = 10*np.finfo(float).eps
        self.assertTrue(isclose(rho, np.abs(x), abs_tol=abs_tol))
        self.assertTrue(isclose(theta, np.pi, abs_tol=abs_tol))

    def test_x_pos_y_pos(self):
        """Test inputs for cart2pol."""
        x = np.sqrt(3) / 2
        y = 1 / 2
        rho, theta = cart2pol(x, y)
        abs_tol = 10*np.finfo(float).eps
        self.assertTrue(isclose(rho, 1, abs_tol=abs_tol))
        self.assertTrue(isclose(theta, np.pi / 6, abs_tol=abs_tol))

    def test_x_neg_y_pos(self):
        """Test inputs for cart2pol."""
        x = - 1 / 2
        y = np.sqrt(3) / 2
        rho, theta = cart2pol(x, y)
        abs_tol = 10*np.finfo(float).eps
        self.assertTrue(isclose(rho, 1, abs_tol=abs_tol))
        self.assertTrue(isclose(theta, 2 * np.pi / 3, abs_tol=abs_tol))

    def test_x_pos_y_neg(self):
        """Test inputs for cart2pol."""
        x = np.sqrt(2) / 2
        y = - np.sqrt(2) / 2
        rho, theta = cart2pol(x, y)
        abs_tol = 10*np.finfo(float).eps
        self.assertTrue(isclose(rho, 1, abs_tol=abs_tol))
        self.assertTrue(isclose(theta, - np.pi / 4, abs_tol=abs_tol))

    def test_x_neg_y_neg(self):
        """Test inputs for cart2pol."""
        x = - 123.53
        y = - 23.4344
        rho, theta = cart2pol(x, y)
        self.assertTrue(rho > 1)
        self.assertTrue(theta > - np.pi)
        self.assertTrue(theta < - np.pi / 2)

    def test_np_array_input(self):
        """Test inputs for cart2pol."""
        x = np.array([32, -234.5])
        y = np.array([23.5, -2334.6])
        rho, theta = cart2pol(x, y)

    def test_bad_input_a(self):
        """Test bad inputs for cart2pol."""
        x = np.array([32])
        y = np.array([23.5, -2334.6])
        with self.assertRaises(ValueError):
            cart2pol(x, y)

    def test_bad_input_b(self):
        """Test bad inputs for cart2pol."""
        x = "foo"
        y = 7
        with self.assertRaises(TypeError):
            cart2pol(x, y)

    def test_bad_input_c(self):
        """Test bad inputs for cart2pol."""
        x = "foo"
        y = np.array([23.5, -2334.6])
        with self.assertRaises(TypeError):
            cart2pol(x, y)

    def test_bad_input_d(self):
        """Test bad inputs for cart2pol."""
        x = 7
        y = "bar"
        with self.assertRaises(TypeError):
            cart2pol(x, y)

    def test_bad_input_e(self):
        """Test bad inputs for cart2pol."""
        x = np.array([23.5, -2334.6])
        y = "bar"
        with self.assertRaises(TypeError):
            cart2pol(x, y)

    def test_bad_input_f(self):
        """Test bad inputs for cart2pol."""
        x = np.array([32j, 5])
        y = np.array([23.5, 10])
        with self.assertRaises(TypeError):
            cart2pol(x, y)


if __name__ == '__main__':
    unittest.main()
