"""
Single consistent function to resize centered arrays
"""

import numpy as np

from coralign.util import check


def pad_crop(arr0, outsize, extrapval=0):
    """Insert a 2D array into another array, centered and zero-padded.

    Given an array and a tuple/list denoting the dimensions of a second array,
    this places the smaller array in the center of the larger array and
    returns that larger array.  The smaller array will be zero-padded to the
    size of the larger.  If ``arr0.shape`` is larger than ``outsize`` in
    either dimension, this dimension will be truncated.

    If both sizes of a dimension are even or odd, the array will be centered
    in that dimension.  If the input size is even and the output size is odd,
    the smaller array will be shifted one element toward the start of the
    array.  If the input is odd and the output is even, the smaller array will
    be shifted one element toward the end of the array.  This sequence is
    intended to ensure transitivity, so several ``pad_crop()`` calls can be
    chained in no particular order without changing the final result.

    The output array will be of the same type as the input array. It will be
    a copy even if the arrays are the same size.

    Arguments:
     arr0: an ndarray
     outsize: a 2-element tuple/list/ndarray of positive integers giving
      dimensions of output array.

    Returns:
     an ndarray of the same size as ``outsize`` and type as ``arr0``

    """

    check.twoD_array(arr0, 'arr0', TypeError)
    check.real_scalar(extrapval, 'extrapval', TypeError)

    sh0 = arr0.shape
    sh1 = outsize

    # check inputs
    try:
        if len(sh1) != 2:
            raise TypeError('Output dimensions must have 2 elements')
    except TypeError: # not iterable
        raise TypeError('outsize must be an iterable')
    check.positive_scalar_integer(sh1[0], 'sh1[0]', TypeError)
    check.positive_scalar_integer(sh1[1], 'sh1[1]', TypeError)

    out = extrapval * np.ones(outsize)
    out = np.array(out, dtype=arr0.dtype)

    xneg = min(sh0[1]//2, sh1[1]//2)
    xpos = min(sh0[1] - sh0[1]//2, sh1[1] - sh1[1]//2)
    yneg = min(sh0[0]//2, sh1[0]//2)
    ypos = min(sh0[0] - sh0[0]//2, sh1[0] - sh1[0]//2)

    slice0 = (slice(sh0[0]//2-yneg, sh0[0]//2+ypos), \
              slice(sh0[1]//2-xneg, sh0[1]//2+xpos))
    slice1 = (slice(sh1[0]//2-yneg, sh1[0]//2+ypos), \
              slice(sh1[1]//2-xneg, sh1[1]//2+xpos))

    out[slice1] = arr0[slice0]

    return out


def offcenter_crop(
        arrayIn, pixel_count_across, output_center_y, output_center_x):
    """
    Crop a 2-D array to be centered at the specified pixel.

    This function crops a 2-D array about the center pixel specified by
    output_center_x and output_center_y. The input array can be
    rectangular with even or odd side lengths. The output will be a
    square of side length pixel_count_across. If the output array has
    regions outside the original 2-D array, those pixels are included
    and set to zero. If the specified cropping region is fully outside the
    input array, then the output array is all zeros.

    The center pixel of an odd-sized array is the array center, and the center
    pixel of an even-sized array follows the FFT center pixel convention.

    Parameters
    ----------
    arrayIn : array_like
        2-D input array
    pixel_count_across : int
        Width of the 2-D, square output array. Units of pixels.
    output_center_y, output_center_x : float or int
        Indices of the pixel to be used as the output array's center.
        Floating point values are rounded to the nearest integer.
        Convention in this function is that y is the first axis.
        Values can be negative and/or lie outside the input array.

    Returns
    -------
    recentered_image : numpy ndarray
        2-D square array

    Notes
    -----
    All alignment units are in detector pixels.
    """
    check.twoD_array(arrayIn, 'arrayIn', TypeError)
    check.positive_scalar_integer(pixel_count_across, 'pixel_count_across',
                            TypeError)
    check.real_scalar(output_center_y, 'output_center_y', TypeError)
    check.real_scalar(output_center_x, 'output_center_x', TypeError)

    y_center = int(np.round(output_center_y))
    x_center = int(np.round(output_center_x))
    [pixel_count_y, pixel_count_x] = arrayIn.shape

    # Compute how much to pad the array in x (if any)
    x_pad_pre = 0
    x_pad_post = 0
    if np.ceil(-pixel_count_across/2.) + x_center < 0:
        x_pad_pre = np.abs(np.ceil(-pixel_count_across/2.) + x_center)
    if np.ceil(pixel_count_across/2.) + x_center > (pixel_count_x - 1):
        x_pad_post = np.ceil(pixel_count_across/2.) + x_center - \
            (pixel_count_x)
    x_pad = int(np.max((x_pad_pre, x_pad_post)))

    # Compute how much to pad the array in y (if any)
    y_pad_pre = 0
    y_pad_post = 0
    if np.ceil(-pixel_count_across/2.) + y_center < 0:
        y_pad_pre = np.abs(np.ceil(-pixel_count_across/2.) + y_center)
    if np.ceil(pixel_count_across/2.) + y_center > (pixel_count_y - 1):
        y_pad_post = np.ceil(pixel_count_across/2.) + y_center - \
            (pixel_count_y)
    y_pad = int(np.max((y_pad_pre, y_pad_post)))

    padded_image = pad_crop(arrayIn, (pixel_count_y+2*y_pad,
                                      pixel_count_x+2*x_pad))

    x_center += x_pad
    y_center += y_pad

    # Buffer needed to keep output array correct size
    if pixel_count_across % 2 == 1:
        buffer_ = 1
    else:
        buffer_ = 0

    recentered_image = padded_image[
        y_center-pixel_count_across//2:(y_center+pixel_count_across//2 +
                                        buffer_),
        x_center-pixel_count_across//2:(x_center+pixel_count_across//2 +
                                        buffer_)
    ]

    return recentered_image
