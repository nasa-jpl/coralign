"""Module to hold basic math functions."""
import numpy as np

from coralign.util import check


def rms(arrayIn):
    """Compute the root mean square of a real-valued array."""
    check.real_array(arrayIn, 'arrayIn', TypeError)

    return np.sqrt(np.mean(arrayIn**2))


def ceil_odd(x_in):
    """
    Compute the next highest odd integer above the input.

    Parameters
    ----------
    x_in : float
        Scalar value

    Returns
    -------
    x_out : integer
        Odd-valued integer
    """
    check.real_scalar(x_in, 'x_in', TypeError)

    x_out = int(np.ceil(x_in))
    if x_out % 2 == 0:
        x_out += 1
    return x_out


def ceil_even(x_in):
    """
    Compute the next highest even integer above the input.

    Parameters
    ----------
    x_in : float
        Scalar value

    Returns
    -------
    x_out : int
        Even-valued integer
    """
    check.real_scalar(x_in, 'x_in', TypeError)

    return int(2 * np.ceil(0.5 * x_in))


def cart2pol(x, y):
    """
    Convert Cartesian coordinate(s) into polar coordinate(s).

    Parameters
    ----------
    x : float or numpy.ndarray
        x-axis coordinate(s)
    y : float or numpy.ndarray
        y-axis coordinate(s)

    Returns
    -------
    rho : float or numpy.ndarray
        radial coordinate(s)
    theta : float or numpy.ndarray
        azimuthal coordinate(s)
    """
    if(type(x) == np.ndarray and type(y) == np.ndarray):
        check.real_array(x, 'x', TypeError)
        check.real_array(y, 'y', TypeError)
        if not x.shape == y.shape:
            raise ValueError('The two inputs must have the same shape.')
    else:
        check.real_scalar(x, 'x', TypeError)
        check.real_scalar(y, 'y', TypeError)

    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    return(rho, theta)
