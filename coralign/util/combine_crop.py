"""
Utility functions to combine a frame and bad pixel map, and to crop a frame
to a size

Used internal to FSW to create HOWFSC GITL frames to be sent to the ground
"""

import numpy as np

from coralign.util import check


def combine(frame, bp):
    """
    Combine a frame and a bad pixel map

    Bad pixels (bp == True) are set to NaNs in the combined frame.

    FSW will use this for cleaned frames (1024x1024), but the function is not
    limited to this size.

    Arguments:
     frame: 2D array
     bp: 2D boolean array.  Same size as frame.  Bad pixels are True.

    Returns:
     2D float array of the same size as frame

    """
    check.twoD_array(frame, 'frame', TypeError)
    check.twoD_array(bp, 'bp', TypeError)
    if bp.shape != frame.shape:
        raise TypeError('frame and bp must be the same shape')
    if bp.dtype != bool:
        raise TypeError('bp must be a boolean array')

    out = frame.copy().astype('float')
    out[bp] = np.nan

    return out


def crop(frame, lower_row, lower_col, row_width, col_width):
    """
    Crop a subregion of a frame

    If the specified crop region extends outside the frame, pad the crop
    region with NaNs

    Arguments:
     frame: 2D array
     lower_row: integer >= 0; starting row of cropped region *inclusive*. Must
      be < number of rows in frame.
     lower_col: integer >= 0; starting column of cropped region *inclusive*.
      Must be < number of columns in frame.
     row_width: integer > 0: number of rows in crop region
     col_width: integer > 0: number of columns in crop region

    Returns:
     2D array of size (row_width, col_width)

    """
    check.twoD_array(frame, 'frame', TypeError)
    check.nonnegative_scalar_integer(lower_row, 'lower_row', TypeError)
    check.nonnegative_scalar_integer(lower_col, 'lower_col', TypeError)
    check.positive_scalar_integer(row_width, 'row_width', TypeError)
    check.positive_scalar_integer(col_width, 'col_width', TypeError)
    if lower_row >= frame.shape[0]:
        raise ValueError('Starting row must be within frame')
    if lower_col >= frame.shape[1]:
        raise ValueError('Starting col must be within frame')

    excerpt = frame[lower_row:lower_row+row_width,
                    lower_col:lower_col+col_width]

    # may be smaller than (row_width, col_width); Python quietly truncates
    erow, ecol = excerpt.shape

    out = np.full((row_width, col_width), np.nan)
    out[:erow, :ecol] = excerpt

    return out
