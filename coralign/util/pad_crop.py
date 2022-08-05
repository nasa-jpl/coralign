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
