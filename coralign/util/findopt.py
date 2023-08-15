"""Functions for optimization."""

import numpy as np

from coralign.util.check import oneD_array, twoD_array


class QuadraticException(Exception):
    """Exception for a bad quadratic fit."""

    pass


def find_optimum_1d(xVec, arrayToFit):
    """
    Fit a parabola to a 1-D array and return the location of the min or max.

    Parameters
    ----------
    xVec : array_like
        1-D array of coordinate values for the values in arrayToFit.
    arrayToFit : array_like
        1-D array of values to fit.

    Returns
    -------
    xOpt : float
        Best fit value for the optimum (i.e., min or max) of the parabola.
    """
    oneD_array(xVec, 'xVec', ValueError)
    oneD_array(arrayToFit, 'arrayToFit', ValueError)
    if len(xVec) != len(arrayToFit):
        raise ValueError('Both input arrays must have same length.')

    orderOfFit = 2
    p0, p1, _ = np.polyfit(xVec, arrayToFit, orderOfFit)

    # Avoid divide by zero error
    if np.abs(p0) < np.finfo(p0).eps:
        raise QuadraticException('Quadratic fit failed because data being ' +
                                 'fitted has no quadratic component.')

    # Find x value at the optimum
    xOpt = -p1/(2*p0)

    return xOpt


def find_optimum_2d(xVec, yVec, arrayToFit, mask):
    """
    Fit a paraboloid to a 2-D array and return the location of the min or max.

    Parameters
    ----------
    xVec : array_like
        1-D array of coordinate values along axis 1 of arrayToFit.
    yVec : array_like
        1-D array of coordinate values along axis 0 of arrayToFit.
    arrayToFit : array_like
        2-D array of values to fit.
    mask : array_like
        2-D boolean mask of which pixels to use. Same shape as arrayToFit.

    Returns
    -------
    xBest : float
        Best fit value along axis 1.
    yBest : float
        Best fit value along axis 0.

    Notes
    -----
    Modified from code at
    https://au.mathworks.com/matlabcentral/answers/5482-fit-a-3d-curve
    and equations at
    https://math.stackexchange.com/questions/2010758/how-do-i-fit-a-paraboloid-surface-to-nine-points-and-find-the-minimum
    """
    oneD_array(xVec, 'xVec', ValueError)
    oneD_array(yVec, 'yVec', ValueError)
    twoD_array(arrayToFit, 'arrayToFit', ValueError)
    twoD_array(mask, 'mask', ValueError)
    if arrayToFit.shape != mask.shape:
        raise ValueError('arrayToFit and mask must have same shape')
    nx = len(xVec)
    ny = len(yVec)
    if arrayToFit.shape[0] != ny:
        raise ValueError('yVec and axis 0 of arrayToFit must have same length')
    if arrayToFit.shape[1] != nx:
        raise ValueError('xVec and axis 1 of arrayToFit must have same length')

    maskBool = np.asarray(mask).astype(bool)
    nPix = np.sum(mask.astype(int))

    #  Set the basis functions
    [X, Y] = np.meshgrid(xVec, yVec)
    f0 = X**2
    f1 = X * Y
    f2 = Y**2
    f3 = X
    f4 = Y
    f5 = np.ones((ny, nx))

    # Write as matrix equation and solve for coefficients
    A = np.concatenate((f0[maskBool].reshape((nPix, 1)),
                        f1[maskBool].reshape((nPix, 1)),
                        f2[maskBool].reshape((nPix, 1)),
                        f3[maskBool].reshape((nPix, 1)),
                        f4[maskBool].reshape((nPix, 1)),
                        f5[maskBool].reshape((nPix, 1))), axis=1)
    y = arrayToFit[maskBool].flatten()
    temp = np.linalg.lstsq(A, y, rcond=None)
    coeffs = temp[0]

    # Take partial derivatives w.r.t. x and y, set to zero, and solve for
    # x and y.
    # G = 0 = a*X**2 + b*X*Y + c*Y**2 + d*X + e*Y + f
    # @G/@X = 2*a*X + b*Y + d = 0 --> Solve for X. Use eq. for Y below.
    # @G/@Y = b*X + 2*c*Y + e = 0 --> Solve for Y in terms of X.
    a, b, c, d, e = coeffs[0:5]

    # Avoid divide by zero errors
    x_denominator = (b*b - 4.0*a*c)
    y_denominator = (2*c)
    # The factors 4 and 2 come from the values in the denomimators
    if (np.abs(x_denominator) < 4*np.finfo(x_denominator).resolution or
    np.abs(y_denominator) < 2*np.finfo(y_denominator).resolution):
        raise QuadraticException('Quadratic fit failed because data being ' +
                                 'fitted has no quadratic component.')

    xBest = (2.0*c*d - e*b)/(b*b - 4.0*a*c)
    yBest = -(b*xBest + e)/(2*c)

    return xBest, yBest
