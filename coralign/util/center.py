"""Compute average index along x- and y-directions for an image."""
import numpy as np
from coralign.util.check import twoD_array, real_array


def center_of_energy(image_array):
    """
    Compute average index along x- and y-directions for an image.

    This function computes the average index along the x-
    and y-directions for a 2-D image array repesenting a pupil or focal plane.

    Parameters
    ----------
    image : numpy ndarray
        2-D image taken at the focal or pupil plane.

    Returns
    -------
    avg_row : float
        The average index along the y-direction.
    avg_col : float
        The average index along the x-direction.
    """
    twoD_array(image_array, 'image_array', TypeError)
    real_array(image_array, 'image_array', TypeError)

    # save idxs of nonzero pixels
    rows, cols = np.nonzero(image_array)

    # extract energies of nonzero pixels
    energies = image_array[np.nonzero(image_array)]

    # sum energy
    total_energy = np.sum(energies)

    # compute weight of idxs based on relative energy
    weights = energies/total_energy

    # computed weighted averge of idxs
    row_values = rows * weights
    col_values = cols * weights
    avg_row = np.sum(row_values)
    avg_col = np.sum(col_values)

    return avg_col, avg_row
