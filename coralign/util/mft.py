"""
Functions to run forward and inverse matrix Fourier transforms (MFT)
"""

import numpy as np

from coralign.util import check


def do_mft(e, outshape, pixperlod, pixperpupil):
    """
    Use a matrix Fourier transform (MFT) to go from pupil to focus

    MFT is an O(N^3) algorithm, but can get fine sampling without padding, and
    so can under some circumstances be faster than a fast Fourier transform
    (FFT).

    This function assumes the input pupil array has the pupil diameter
    consistent with ``pixperpupil`` and the output array is camera-sampled
    consistent with ``pixperlod``.  This does not assume the input array is
    exactly cropped to pupil dimensions.

    Arguments:
     e: 2D complex-valued array with input field
     outshape: 2-tuple with dimensions of output array as positive integers
     pixperlod: real positive scalar value for pixels per lambda/D in the
      focal plane (output plane)
     pixperpupil: real positive scalar value for pixels across the pupil in
      the pupil plane (input plane) to get the scaling correct.

    Returns:
     2D complex-valued array with input field with size ``outshape``

    """
    # check inputs
    check.twoD_array(e, 'e', TypeError)
    check.real_positive_scalar(pixperlod, 'pixperlod', TypeError)
    check.real_positive_scalar(pixperpupil, 'pixperpupil', TypeError)

    try:
        if len(outshape) != 2:
            raise TypeError('Output dimensions must have 2 elements')
        for index, dim in enumerate(outshape):
            check.positive_scalar_integer(dim, 'outshape['+str(index)+']',
                                          TypeError)
            pass
        pass
    except TypeError:  # not iterable
        raise TypeError('outshape must be an iterable')

    # pupil plane coords
    row, drow = _interval(e.shape[0]/pixperpupil, e.shape[0])
    col, dcol = _interval(e.shape[1]/pixperpupil, e.shape[1])

    # focal plane coords. diam in L/D = diam in pix / pix per L/D
    rowpp = _interval(outshape[0]/pixperlod, outshape[0])[0]
    colpp = _interval(outshape[1]/pixperlod, outshape[1])[0]

    # Take to image plane
    rowmult = np.exp(-2.*np.pi*1j*np.outer(rowpp, row))
    colmult = np.exp(-2.*np.pi*1j*np.outer(col, colpp))

    temp = np.dot(np.dot(rowmult, e), colmult)*drow*dcol
    return temp


def do_offcenter_mft(e, outshape, pixperlod, pixperpupil, pupilShape,
                     yxLowerLeft):
    """
    Use a MFT to go from an offcenter-cropped pupil to (centered) focus.

    MFT is an O(N^3) algorithm, but can get fine sampling without padding, and
    so can under some circumstances be faster than a fast Fourier transform
    (FFT).

    This function assumes the input pupil array has the pupil diameter
    consistent with ``pixperpupil`` and the output array is camera-sampled
    consistent with ``pixperlod``.  This does not assume the input array is
    exactly cropped to pupil dimensions.

    Arguments:
     e: 2-D, offcenter-cropped, complex-valued array with input field
     outshape: 2-tuple with dimensions of output array as positive integers
     pixperlod: real positive scalar value for pixels per lambda/D in the
      focal plane (output plane)
     pixperpupil: real positive scalar value for pixels across the pupil in
      the pupil plane (input plane) to get the scaling correct.
     pupilShape: 2-tuple with the shape of the full-sized E-field array.
     yxLowerLeft: 2-tuple with lower left pixel coordinate at which to insert
      e into the full-sized E-field array of shape pupilShape.

    Returns:
     2D complex-valued array with input field with size ``outshape``

    """
    # check inputs
    check.twoD_array(e, 'e', TypeError)
    check.real_positive_scalar(pixperlod, 'pixperlod', TypeError)
    check.real_positive_scalar(pixperpupil, 'pixperpupil', TypeError)

    try:
        if len(outshape) != 2:
            raise TypeError('Output dimensions must have 2 elements')
        for index, dim in enumerate(outshape):
            check.positive_scalar_integer(dim, 'outshape['+str(index)+']',
                                          TypeError)
    except TypeError:  # not iterable
        raise TypeError('outshape must be an iterable')

    try:
        if len(pupilShape) != 2:
            raise TypeError('pupilShape dimensions must have 2 elements')
        for index, dim in enumerate(pupilShape):
            check.positive_scalar_integer(dim, 'pupilShape['+str(index)+']',
                                          TypeError)
    except TypeError:  # not iterable
        raise TypeError('outshape must be an iterable')

    try:
        if len(yxLowerLeft) != 2:
            raise TypeError('yxLowerLeft dimensions must have 2 elements')
        for index, dim in enumerate(yxLowerLeft):
            check.nonnegative_scalar_integer(
                dim, 'yxLowerLeft['+str(index)+']', TypeError)
    except TypeError:  # not iterable
        raise TypeError('yxLowerLeft must be an iterable')

    # full pupil plane coords
    row, drow = _interval(pupilShape[0]/pixperpupil, pupilShape[0])
    col, dcol = _interval(pupilShape[1]/pixperpupil, pupilShape[1])

    # cropped pupil plane coords
    rowcrop = row[yxLowerLeft[0]:yxLowerLeft[0]+e.shape[0]]
    colcrop = col[yxLowerLeft[1]:yxLowerLeft[1]+e.shape[1]]

    # focal plane coords. diam in L/D = (diam in pix)/(pix per L/D)
    rowpp = _interval(outshape[0]/pixperlod, outshape[0])[0]
    colpp = _interval(outshape[1]/pixperlod, outshape[1])[0]

    # Take to image plane
    rowmult = np.exp(-2.*np.pi*1j*np.outer(rowpp, rowcrop))
    colmult = np.exp(-2.*np.pi*1j*np.outer(colcrop, colpp))

    return np.dot(np.dot(rowmult, e), colmult)*drow*dcol


def do_imft(e, outshape, pixperlod, pixperpupil):
    """
    Use an inverse matrix Fourier transform (MFT) to go from focus to pupil

    MFT is an O(N^3) algorithm, but can get fine sampling without padding, and
    so can under some circumstances be faster than a fast Fourier transform
    (FFT).

    This function assumes the output pupil array has the pupil diameter
    consistent with ``pixperpupil`` and the input array is camera-sampled
    consistent with ``pixperlod``.  This does not assume the input array is
    exactly cropped to pupil dimensions.

    Arguments:
     e: 2D complex-valued array with input field
     outshape: 2-tuple with dimensions of output array as positive integers
     pixperlod: real positive scalar value for pixels per lambda/D in the
      focal plane (input plane)
     pixperpupil: real positive scalar value for pixels across the pupil in
      the pupil plane (output plane) to get the scaling correct.

    Returns:
     2D complex-valued array with input field with size ``outshape``

    """
    #check inputs
    check.twoD_array(e, 'e', TypeError)
    check.real_positive_scalar(pixperlod, 'pixperlod', TypeError)
    check.real_positive_scalar(pixperpupil, 'pixperpupil', TypeError)

    try:
        if len(outshape) != 2:
            raise TypeError('Output dimensions must have 2 elements')
        for index, dim in enumerate(outshape):
            check.positive_scalar_integer(dim, 'outshape['+str(index)+']',
                                          TypeError)
            pass
        pass
    except TypeError: #not iterable
        raise TypeError('outshape must be an iterable')

    pixperlod = float(pixperlod)

    row, drow = _interval(e.shape[0]/pixperlod, e.shape[0])
    col, dcol = _interval(e.shape[1]/pixperlod, e.shape[1])
    rowpp = _interval(outshape[0]/pixperpupil, outshape[0])[0]
    colpp = _interval(outshape[1]/pixperpupil, outshape[1])[0]

    # Take to image plane
    rowmult = np.exp(2.*np.pi*1j*np.outer(rowpp, row))
    colmult = np.exp(2.*np.pi*1j*np.outer(col, colpp))

    temp = np.dot(np.dot(rowmult, e), colmult)*drow*dcol
    return temp



def _interval(width, npts):
    """
    Make a vector spaced to match FFT assumptions (i.e. left side of subpixels)

    Arguments:
     width: full width along an axis (real positive scalar)
     npts: number of subpixels (positive scalar integer)

    Returns
     2-tuple with (vector of values for each subpixel, subpixel width)

    """
    check.real_positive_scalar(width, 'width', TypeError)
    check.positive_scalar_integer(npts, 'npts', TypeError)

    dx = width/npts
    x = np.arange(-(npts//2), npts-(npts//2))*dx
    return x, dx
