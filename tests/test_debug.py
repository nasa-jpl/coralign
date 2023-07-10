"""Test suite for debug.py."""
import pytest
import numpy as np

from coralign.util import debug, shapes


def test_debug_plot_bad_input_plot_num():
    """Test an incorrect input type for plot_num."""
    dg = True
    plot_num = 'foobar'
    img_arr = shapes.circle(10, 10, 2, 1, -1)
    title = 'TITLE'
    with pytest.raises(TypeError):
        debug.debug_plot(dg, plot_num, img_arr, title)


def test_debug_plot_bad_input_img_arr():
    """Test an incorrect input type for img_arr."""
    dg = True
    plot_num = 0
    img_arr = np.array([1])
    title = 'TITLE'
    with pytest.raises(TypeError):
        debug.debug_plot(dg, plot_num, img_arr, title)


def test_debug_plot_bad_input_title():
    """Test an incorrect input type for title."""
    dg = True
    plot_num = 0
    img_arr = shapes.circle(10, 10, 2, 1, -1)
    title = shapes.circle(10, 10, 2, 1, -1)
    with pytest.raises(TypeError):
        debug.debug_plot(dg, plot_num, img_arr, title)


def test_debug_plot_valid_input():
    """Test a correct input."""
    dg = True
    plot_num = 0
    img_arr = shapes.circle(10, 10, 2, 1, -1)
    title = 'TITLE'
    debug.debug_plot(dg, plot_num, img_arr, title)
