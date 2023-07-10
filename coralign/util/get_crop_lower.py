"""
Utility function to get parameter values for combine_crop.py
"""

from coralign.util import check


def get_crop_lower(row_center, col_center, row_width, col_width, bound=False):
    """
    Utility function to get the lower corner given desired center and size

    Not directly part of any pipeline, but will be useful if we need to pick a
    new center location to get the inputs to e.g. crop.

    As this function isn't part of any pipeline, we're right now choosing to
    allow the option of having lower_row and lower_col be negative if you ask
    to center a crop region too close to the corner.  Use bound=True to force
    the output to be >= 0.

    For odd-sized widths, the *_center argument will be in the exact center,
    e.g. 0 0 X 0 0.

    For even-sized width, the *_center argument will be the larger-indexed of
    the two central elements, e.g. 0 0 X 0.

    Arguments:
     row_center: center of cropped region along the row axis.  Integer >= 0
     col_center: center of cropped region along the col axis.  Integer >= 0
     row_width: integer > 0: number of rows in crop region
     col_width: integer > 0: number of columns in crop region

    Returns:
     2-tuple with lower_row, lower_col

    """

    check.nonnegative_scalar_integer(row_center, 'row_center', TypeError)
    check.nonnegative_scalar_integer(col_center, 'col_center', TypeError)
    check.positive_scalar_integer(row_width, 'row_width', TypeError)
    check.positive_scalar_integer(col_width, 'col_width', TypeError)
    check.boolean(bound, 'bound', TypeError)

    lower_row = row_center - row_width//2
    lower_col = col_center - col_width//2

    if bound:
        lower_row = max(lower_row, 0)
        lower_col = max(lower_col, 0)
        pass

    return lower_row, lower_col
