"""Module for ampthresh, which decides where the pupil is illuminated."""
import numpy as np

import coralign.util.check as check


def ampthresh(pupilMap, nBin=21):
    """
    Threshold a pupil map to give pixels in the open area.

    Given an amplitude, complex amplitude, or intensity map for a pupil plane,
    threshold it to select pixels which are inside the open area. Used
    to create masks to eliminate low SNR regions.

    Parameters
    ----------
    pupilMap : array_like
        2-D pupil map. Can be an intensity image, pupil amplitude, or a
        complex-valued phase retrieval
    nBin : numpy ndarray
        Number of bins used when making a histogram of pupil values.

    Returns
    -------
    boolMask : numpy ndarray
        2-D boolean map of pixels above the threshold value in the pupil
        map. Same size as the input pupilMap.
    """
    check.twoD_array(pupilMap, 'pupilMap', ValueError)
    check.positive_scalar_integer(nBin, 'nBin', ValueError)

    if np.any(np.iscomplex(pupilMap)):
        pupilMaptmp = np.abs(pupilMap)
        pupilMap = pupilMaptmp

    if np.min(pupilMap) == np.max(pupilMap):
        raise ValueError('Cannot threshold an array of uniform values.')

    # use histogram of intensities to choose threshold
    Icount, IbinEdges = np.histogram(pupilMap, bins=nBin)

    # find the minima in the histogram
    bV = np.logical_and(Icount[1:-1] <= Icount[:-2],
                        Icount[1:-1] < Icount[2:])

    # vector of intensity values at the center of each bin
    binCenter = 0.5*(IbinEdges[:-1]+IbinEdges[1:])

    # List of bin values where the histogram has a miminum
    binVal = binCenter[1:-1][bV]

    # Choose the first minimum as the threshold. this will be the lowest
    # intensity value above the background intensity level. All pixels
    # with greater intensity must be "signal". If binVal is empty, the
    # histogram has no o minima except at an endpoint. This is a
    # failure, so return empty.
    if np.size(binVal) > 0:
        thresh = binVal[0]
    else:
        raise ValueError('PupilType failed to find a histogram minimum')

    # Apply theshold to create boolean mask
    boolMask = pupilMap > thresh

    return boolMask
