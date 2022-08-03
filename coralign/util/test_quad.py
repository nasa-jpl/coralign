"""Test suite for quad.py."""
import pytest
import numpy as np
from math import isclose

from coralign.util import quad, shapes


def test_quadrant_energy_balance_odd_0_0():
    """Test that a centered circle returns 0 difference in x dimension and 0
    difference in y dimension."""
    image_array = shapes.circle(9, 9, 2, 0, 0)  # perfectly centered
    dim_x_diff, dim_y_diff = quad.quadrant_energy_balance(image_array, 4, 4, 9)
    abs_tol = 10*np.finfo(float).eps
    assert isclose(dim_x_diff, 0, abs_tol=abs_tol)
    assert isclose(dim_y_diff, 0, abs_tol=abs_tol)


def test_quadrant_energy_balance_odd_0_0_neg():
    """Test that a negative image of a centered circle returns 0 difference in
    x dimension and 0 difference in y dimension."""
    image_array = shapes.circle(9, 9, 2, 0, 0)  # perfectly centered
    image_array[image_array == 0] = -1
    image_array[image_array > 0] = 0
    image_array[image_array == -1] = 1
    dim_x_diff, dim_y_diff = quad.quadrant_energy_balance(image_array, 4, 4, 9)
    abs_tol = 10*np.finfo(float).eps
    assert isclose(dim_x_diff, 0, abs_tol=abs_tol)
    assert isclose(dim_y_diff, 0, abs_tol=abs_tol)


def test_quadrant_energy_balance_odd_pos_0():
    """Test that a circle translated right returns + difference in x dimension
    and 0 difference in y dimension."""
    image_array = shapes.circle(9, 9, 2, 1, 0)  # [0] pos
    dim_x_diff, dim_y_diff = quad.quadrant_energy_balance(image_array, 4, 4, 9)
    abs_tol = 10*np.finfo(float).eps
    assert dim_x_diff > 0
    assert isclose(dim_y_diff, 0, abs_tol=abs_tol)


def test_quadrant_energy_balance_odd_neg_0():
    """Test that a circle translated left returns - difference in x dimension
    and 0 difference in y dimension."""
    image_array = shapes.circle(9, 9, 2, -1, 0)  # [0] neg
    dim_x_diff, dim_y_diff = quad.quadrant_energy_balance(image_array, 4, 4, 9)
    abs_tol = 10*np.finfo(float).eps
    assert dim_x_diff < 0
    assert isclose(dim_y_diff, 0, abs_tol=abs_tol)


def test_quadrant_energy_balance_odd_0_pos():
    """Test that a circle translated down returns 0 difference in x dimension
    and + difference in y dimension."""
    image_array = shapes.circle(9, 9, 2, 0, 1)  # [1] pos
    dim_x_diff, dim_y_diff = quad.quadrant_energy_balance(image_array, 4, 4, 9)
    abs_tol = 10*np.finfo(float).eps
    assert isclose(dim_x_diff, 0, abs_tol=abs_tol)
    assert dim_y_diff > 0


def test_quadrant_energy_balance_odd_0_neg():
    """Test that a circle translated up returns 0 difference in x dimension
    and - difference in y dimension."""
    image_array = shapes.circle(9, 9, 2, 0, -1)  # [1] neg
    dim_x_diff, dim_y_diff = quad.quadrant_energy_balance(image_array, 4, 4, 9)
    abs_tol = 10*np.finfo(float).eps
    assert isclose(dim_x_diff, 0, abs_tol=abs_tol)
    assert dim_y_diff < 0


def test_quadrant_energy_balance_even_pos_pos():
    """Test that a non-translated circle returns + difference in x dimension
    and + difference in y dimension."""
    image_array = shapes.circle(10, 10, 2, 0, 0)  # [0] pos and [1] pos
    dim_x_diff, dim_y_diff = quad.quadrant_energy_balance(image_array, 4, 4, 9)
    assert dim_x_diff > 0
    assert dim_y_diff > 0


def test_quadrant_energy_balance_even_neg_neg():
    """Test that a circle translated left and up returns - difference in
    x dimension and - difference in y dimension."""
    image_array = shapes.circle(10, 10, 2, -2, -2)  # [0] neg and [1] neg
    dim_x_diff, dim_y_diff = quad.quadrant_energy_balance(image_array, 4, 4, 9)
    assert dim_x_diff < 0
    assert dim_y_diff < 0


def test_quadrant_energy_balance_even_zero_zero():
    """Test that a circle translated left and up returns 0 difference in
    x dimension and 0 difference in y dimension."""
    image_array = shapes.circle(10, 10, 2, -1, -1)  # perfectly centered
    dim_x_diff, dim_y_diff = quad.quadrant_energy_balance(image_array, 4, 4, 9)
    abs_tol = 10*np.finfo(float).eps
    assert isclose(dim_x_diff, 0, abs_tol=abs_tol)
    assert isclose(dim_y_diff, 0, abs_tol=abs_tol)


def test_quadrant_energy_balance_even_neg_pos():
    """Test that a circle translated left and down returns - difference in
    x dimension and + difference in y dimension."""
    image_array = shapes.circle(10, 10, 2, -1, 1)  # [0] neg and [1] pos
    dim_x_diff, dim_y_diff = quad.quadrant_energy_balance(image_array, 5, 5, 9)
    assert dim_x_diff < 0
    assert dim_y_diff > 0


def test_quadrant_energy_balance_even_pos_neg():
    """Test that a circle translated right and down returns + difference in
    x dimension and - difference in y dimension."""
    image_array = shapes.circle(10, 10, 2, 1, -1)  # [0] pos and [1] neg
    dim_x_diff, dim_y_diff = quad.quadrant_energy_balance(image_array, 5, 5, 9)
    assert dim_x_diff > 0
    assert dim_y_diff < 0


def test_quadrant_energy_balance_bad_input_image_array():
    """Test an incorrect input type for image_array."""
    image_array = 'foobar'
    with pytest.raises(TypeError):
        quad.quadrant_energy_balance(image_array, 5, 5, 9)


def test_quadrant_energy_balance_bad_input_x_origin():
    """Test an incorrect input type for x_origin."""
    image_array = shapes.circle(10, 10, 2, 1, -1)
    with pytest.raises(TypeError):
        quad.quadrant_energy_balance(image_array, -5, 5, 9)


def test_quadrant_energy_balance_bad_input_y_origin():
    """Test an incorrect input type for y_origin."""
    image_array = shapes.circle(10, 10, 2, 1, -1)
    with pytest.raises(TypeError):
        quad.quadrant_energy_balance(image_array, 5, -5, 9)


def test_quadrant_energy_balance_bad_input_x_origin_b():
    """Test an incorrect input type for x_origin."""
    image_array = shapes.circle(10, 10, 2, 1, -1)
    with pytest.raises(ValueError):
        quad.quadrant_energy_balance(image_array, 100, 5, 9)


def test_quadrant_energy_balance_bad_input_y_origin_b():
    """Test an incorrect input type for y_origin."""
    image_array = shapes.circle(10, 10, 2, 1, -1)
    with pytest.raises(ValueError):
        quad.quadrant_energy_balance(image_array, 5, 100, 9)


def test_quadrant_energy_balance_bad_input_radius():
    """Test an incorrect input type for radius."""
    image_array = shapes.circle(10, 10, 2, 1, -1)
    with pytest.raises(TypeError):
        quad.quadrant_energy_balance(image_array, 5, 5, -5)


def test_quadrant_energy_balance_odd_even_0_0():
    """Test that a centered circle returns 0 difference in x dimension and 0
    difference in y dimension."""
    image_array = shapes.circle(101, 100, 20, 0, 0)  # perfectly centered
    dim_x_diff, dim_y_diff = quad.quadrant_energy_balance(image_array, 50, 50,
                                                          70)
    abs_tol = 10*np.finfo(float).eps
    assert isclose(dim_x_diff, 0, abs_tol=abs_tol)
    assert isclose(dim_y_diff, 0, abs_tol=abs_tol)


def test_quadrant_energy_balance_odd_even_pos_neg():
    """Test that a centered circle returns 0 difference in x dimension and 0
    difference in y dimension."""
    image_array = shapes.circle(101, 100, 20, 20, -50)
    dim_x_diff, dim_y_diff = quad.quadrant_energy_balance(image_array, 50, 50,
                                                          70)
    assert dim_x_diff > 0
    assert dim_y_diff < 0
