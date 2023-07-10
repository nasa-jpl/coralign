"""Utility functions for the FOCALFIT module."""
import numpy as np

from coralign.util import check


def bound_value(inVal, maxVal):
    """
    Restrict a scalar to the range [-maxVal, maxVal].

    Parameters
    ----------
    inVal : float
        Input value to bound.
    maxVal : float
        Maximum allowed value. Must be nonnegative scalar.

    Returns
    -------
    outVal : float
        Bounded value.
    """
    check.real_scalar(inVal, 'inVal', ValueError)
    check.real_nonnegative_scalar(maxVal, 'maxVal', ValueError)

    if inVal > maxVal:
        outVal = maxVal

    elif inVal < -maxVal:
        outVal = -maxVal

    else:
        outVal = inVal

    return outVal


def calc_offset_quadratic(fitCoefPow1, fitCoefPow2, roiSumRatio):
    """
    Solve for the offset from the quadratic equation.

    Parameters
    ----------
    fitCoefPow1 : float
        Based on simulation, there is either a linear or quadratic change in
        the summed ROI ratio versus stellar offset from the mask. fitCoefPow1
        is the coefficient of the linear term in that polynomial fit. Must be a
        positive scalar value to guarantee that the offset estimate has the
        correct sign.
    fitCoefPow2 : float
        Based on simulation, there is either a linear or quadratic change in
        the summed ROI ratio versus stellar offset from the mask. fitCoefPow2
        is the coefficient of the quadratic term in that polynomial fit. Must
        be positive to avoid divide by zero and to avoid choosing the wrong
        answer from the quadratic formula.
    roiSumRatio : float
        Ratio of the summed energy within the regions of interest (ROI).
        A ratio >1 means that there is more energy in the ROI along the
        positive direction of the axis.

    Returns
    -------
    offset : float
        Estimated offset in pixels.

    """
    check.real_scalar(fitCoefPow1, 'fitCoefPow1', ValueError)
    check.real_scalar(fitCoefPow2, 'fitCoefPow2', ValueError)
    check.real_positive_scalar(roiSumRatio, 'roiSumRatio', ValueError)

    if roiSumRatio < 1:  # Mask is to the right of star
        roiSumRatio = 1./roiSumRatio
        sign = -1
    else:  # Mask is to the left of star
        sign = 1

    offset = sign*((-fitCoefPow1 + np.sqrt(fitCoefPow1*fitCoefPow1 -
                                          4*fitCoefPow2*(1.-roiSumRatio))) /
                   (2.*fitCoefPow2))

    return offset


def calc_offset_linear(fitCoefPow1, roiSumRatio):
    """
    Solve for the offset based on a linear relationship.

    Parameters
    ----------
    fitCoefPow1 : float
        Based on simulation, there is either a linear or quadratic change in
        the summed ROI ratio versus stellar offset from the mask. fitCoefPow1
        is the coefficient of the linear term in that polynomial fit. Must be a
        positive scalar value to guarantee that the offset estimate has the
        correct sign.
    roiSumRatio : float
        Ratio of the summed energy within the regions of interest (ROI).
        A ratio >1 means that there is more energy in the ROI along the
        positive direction of the axis.

    Returns
    -------
    offset : float
        Estimated offset in pixels.

    """
    check.real_scalar(fitCoefPow1, 'fitCoefPow1', ValueError)
    check.real_positive_scalar(roiSumRatio, 'roiSumRatio', ValueError)

    if roiSumRatio < 1:
        roiSumRatio = 1./roiSumRatio
        sign = -1
    else:
        sign = 1

    offset = sign*(roiSumRatio - 1)/fitCoefPow1

    return offset
