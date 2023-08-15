"""Module to hold utility functions for the DMREG module."""
import numpy as np

from coralign.util.check import twoD_array


def remove_piston_tip_tilt(arrayToFit, mask):
    """
    Remove piston and x- and y-ramp terms from an array.

    Parameters
    ----------
    arrayToFit : array_like
        2-D array of values to fit.
    mask : array_like
        2-D boolean mask of which pixels to use in "arrayToFit".
        Must be same size as "arrayToFit".

    Returns
    -------
    arrayOut : array_like
        Input array with piston, tip, and tilt subtracted.
    """
    twoD_array(arrayToFit, 'arrayToFit', ValueError)
    twoD_array(mask, 'mask', ValueError)
    if arrayToFit.shape != mask.shape:
        raise ValueError('arrayToFit and mask must have same shape')

    maskBool = np.asarray(mask).astype(bool)
    nPix = int(np.sum(mask))
    ny = arrayToFit.shape[0]
    nx = arrayToFit.shape[1]
    xVec = np.arange(-nx/2., nx/2.)/nx
    yVec = np.arange(-ny/2., ny/2.)/ny

    #  Set the basis functions
    ONES = np.ones((ny, nx))
    [X, Y] = np.meshgrid(xVec, yVec)
    f0 = X
    f1 = Y
    f2 = ONES

    # Write as matrix equation and solve for coefficients
    A = np.concatenate((f0[maskBool].reshape((nPix, 1)),
                        f1[maskBool].reshape((nPix, 1)),
                        f2[maskBool].reshape((nPix, 1))), axis=1)
    y = arrayToFit[maskBool].flatten()
    temp = np.linalg.lstsq(A, y, rcond=None)
    coeffs = temp[0]
    a, b, c = coeffs[0:3]
    arrayOut = maskBool * (arrayToFit - (a*X + b*Y + c*ONES))

    return arrayOut
