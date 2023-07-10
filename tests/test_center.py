"""Test suite for center.py."""
import pytest
import numpy as np
from math import isclose

from coralign.util import center, shapes


def test_center_of_energy_odd_0_0():
    """Test that a centered circle returns center x index and
    center y index."""
    image_array = shapes.circle(9, 9, 2, 0, 0)  # perfectly centered
    dim_x_diff, dim_y_diff = center.center_of_energy(image_array)
    abs_tol = 10*np.finfo(float).eps
    assert isclose(dim_x_diff, 4, abs_tol=abs_tol)
    assert isclose(dim_y_diff, 4, abs_tol=abs_tol)


def test_center_of_energy_odd_0_0_neg():
    """Test that a negative image of a centered circle returns center x
    index and center y index."""
    image_array = shapes.circle(9, 9, 2, 0, 0)  # perfectly centered
    image_array[image_array == 0] = -1
    image_array[image_array > 0] = 0
    image_array[image_array == -1] = 1
    dim_x_diff, dim_y_diff = center.center_of_energy(image_array)
    abs_tol = 10*np.finfo(float).eps
    assert isclose(dim_x_diff, 4, abs_tol=abs_tol)
    assert isclose(dim_y_diff, 4, abs_tol=abs_tol)


def test_center_of_energy_odd_pos_0():
    """Test that a circle translated right returns > center x index
    and center y index."""
    image_array = shapes.circle(9, 9, 2, 1, 0)  # [0] pos
    dim_x_diff, dim_y_diff = center.center_of_energy(image_array)
    abs_tol = 10*np.finfo(float).eps
    assert dim_x_diff > 4
    assert isclose(dim_y_diff, 4, abs_tol=abs_tol)


def test_center_of_energy_odd_neg_0():
    """Test that a circle translated left returns < center x index
    and center y index."""
    image_array = shapes.circle(9, 9, 2, -1, 0)  # [0] neg
    dim_x_diff, dim_y_diff = center.center_of_energy(image_array)
    abs_tol = 10*np.finfo(float).eps
    assert dim_x_diff < 4
    assert isclose(dim_y_diff, 4, abs_tol=abs_tol)


def test_center_of_energy_odd_0_pos():
    """Test that a circle translated down returns center x index
    and > center y index."""
    image_array = shapes.circle(9, 9, 2, 0, 1)  # [1] pos
    dim_x_diff, dim_y_diff = center.center_of_energy(image_array)
    abs_tol = 10*np.finfo(float).eps
    assert isclose(dim_x_diff, 4, abs_tol=abs_tol)
    assert dim_y_diff > 4


def test_center_of_energy_odd_0_neg():
    """Test that a circle translated up returns center x index
    and < center y index."""
    image_array = shapes.circle(9, 9, 2, 0, -1)  # [1] neg
    dim_x_diff, dim_y_diff = center.center_of_energy(image_array)
    abs_tol = 10*np.finfo(float).eps
    assert isclose(dim_x_diff, 4, abs_tol=abs_tol)
    assert dim_y_diff < 4


def test_center_of_energy_even_pos_pos():
    """Test that a non-translated circle returns > center x index
    and > center y index."""
    image_array = shapes.circle(10, 10, 2, 0, 0)  # [0] pos and [1] pos
    dim_x_diff, dim_y_diff = center.center_of_energy(image_array)
    assert dim_x_diff > 4
    assert dim_y_diff > 4


def test_center_of_energy_even_neg_neg():
    """Test that a circle translated left and up returns < center x
    index and < center y index."""
    image_array = shapes.circle(10, 10, 2, -2, -2)  # [0] neg and [1] neg
    dim_x_diff, dim_y_diff = center.center_of_energy(image_array)
    assert dim_x_diff < 4
    assert dim_y_diff < 4


def test_center_of_energy_even_zero_zero():
    """Test that a circle translated left and up returns center x
    index and center y index."""
    image_array = shapes.circle(10, 10, 2, -1, -1)  # perfectly centered
    dim_x_diff, dim_y_diff = center.center_of_energy(image_array)
    abs_tol = 10*np.finfo(float).eps
    assert isclose(dim_x_diff, 4, abs_tol=abs_tol)
    assert isclose(dim_y_diff, 4, abs_tol=abs_tol)


def test_center_of_energy_even_neg_pos():
    """Test that a circle translated left and down returns < center x
    index and > center y index."""
    image_array = shapes.circle(10, 10, 2, -1, 1)  # [0] neg and [1] pos
    dim_x_diff, dim_y_diff = center.center_of_energy(image_array)
    assert dim_x_diff < 5
    assert dim_y_diff > 5


def test_center_of_energy_even_pos_neg():
    """Test that a circle translated right and down returns > center x
    index and < center y index."""
    image_array = shapes.circle(10, 10, 2, 1, -1)  # [0] pos and [1] neg
    dim_x_diff, dim_y_diff = center.center_of_energy(image_array)
    assert dim_x_diff > 5
    assert dim_y_diff < 5


def test_center_of_energy_bad_input_image_array():
    """Test an incorrect input type for image_array."""
    image_array = 'foobar'
    with pytest.raises(TypeError):
        center.center_of_energy(image_array)


def test_center_of_energy_odd_even_0_0():
    """Test that a centered circle returns center x index and
    center y index."""
    image_array = shapes.circle(101, 100, 20, 0, 0)  # perfectly centered
    dim_x_diff, dim_y_diff = center.center_of_energy(image_array)
    abs_tol = 10*np.finfo(float).eps
    assert isclose(dim_x_diff, 50, abs_tol=abs_tol)
    assert isclose(dim_y_diff, 50, abs_tol=abs_tol)


def test_center_of_energy_odd_even_pos_neg():
    """Test that a centered circle returns center x index and
    center y index."""
    image_array = shapes.circle(101, 100, 20, 20, -50)
    dim_x_diff, dim_y_diff = center.center_of_energy(image_array)
    assert dim_x_diff > 50
    assert dim_y_diff < 50
