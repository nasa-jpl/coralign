"""Compute energy difference in along x- and y-directions for an image."""
import numpy as np
from coralign.util.check import (twoD_array, real_array, real_nonnegative_scalar,
                        nonnegative_scalar_integer)
from coralign.util.math import cart2pol


def quadrant_energy_balance(image_array, x_origin, y_origin, radius):
    """
    Compute energy difference in along x- and y-directions for an image.

    This function computes the energy difference along the x-
    and y-directions for a 2-D image array repesenting a pupil or focal plane.

    Parameters
    ----------
    image : numpy ndarray
        2-D image taken at the focal or pupil plane.
    x_origin : nonnegative integer
        Pixel index (starting with zero index) of the center coordinate point
        (i.e., the origin) in the x direction (column index)
        to assume in calculation.
    y_origin : nonnegative integer
        Pixel index (starting with zero index) of the center coordinate point
        (i.e., the origin) in the y direction (row index)
        to assume in calculation.
    radius : float
        Radius of circle to consider in computations around the
        origin.

    Returns
    -------
    dim_x_diff : float
        The energy difference along the x-direction.
    dim_y_diff : float
        The energy difference along the y-direction.
    """
    twoD_array(image_array, 'image_array', TypeError)
    real_array(image_array, 'image_array', TypeError)
    real_nonnegative_scalar(radius, 'radius', TypeError)
    nonnegative_scalar_integer(x_origin, 'x_origin', TypeError)
    nonnegative_scalar_integer(y_origin, 'y_origin', TypeError)

    norm_image_array = image_array/np.sum(image_array)

    num_rows, num_columns = norm_image_array.shape

    if not x_origin < num_rows:
        raise ValueError('x_origin must be less than the number of rows in image_array.')
    if not y_origin < num_columns:
        raise ValueError('y_origin must be less than the number of columns in image_array.')

    originRow, originColumn = y_origin, x_origin

    x_in = np.linspace(- (originColumn), num_columns - originColumn - 1,
                       num=num_columns)
    y_in = np.linspace(- (originRow), num_rows - originRow - 1,
                       num=num_rows)

    x, y = np.meshgrid(x_in, y_in)

    [rho, theta] = cart2pol(x, y)

    mask = rho <= radius

    # q4|q3
    # –––––
    # q2|q1

    q1 = np.logical_and(np.logical_and(x >= 0, y >= 0), mask)
    q2 = np.logical_and(np.logical_and(x <= 0, y >= 0), mask)
    q3 = np.logical_and(np.logical_and(x >= 0, y <= 0), mask)
    q4 = np.logical_and(np.logical_and(x <= 0, y <= 0), mask)

    counts_q1 = np.sum(norm_image_array[q1 == True])
    counts_q2 = np.sum(norm_image_array[q2 == True])
    counts_q3 = np.sum(norm_image_array[q3 == True])
    counts_q4 = np.sum(norm_image_array[q4 == True])

    dim_x_diff = (counts_q1+counts_q3)-(counts_q2+counts_q4)
    dim_y_diff = (counts_q1+counts_q2)-(counts_q3+counts_q4)

    return dim_x_diff, dim_y_diff
