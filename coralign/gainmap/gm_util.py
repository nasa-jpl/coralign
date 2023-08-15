"""Utility functions for computing gain maps or height maps of DM actuators."""
import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits

from coralign.util import check


def compute_delta_height_map_from_command_maps(commandMapBefore,
                                               commandMapAfter,
                                               fnRefCube,
                                               fnRefCommandVec):
    """
    Compute difference of actuator heights for two command maps.

    Computes the delta actuator heights (to add to a flight flat setting)
    from the difference of the ground dark hole setting and the ground flat
    setting.

    Parameters
    ----------
    commandMapBefore : array_like
        2-D array of DM actuator commands at the earlier setting.
    commandMapAfter : array_like
        2-D array of DM actuator commands at the later setting.
    fnRefCube : str
        Name of the FITS file containing the datacube with measured DM gains
        at the DM commands specified by fnRefCommandVec.
    fnRefCommandVec : str
        Name of the FITS file containing the DM command values at which the
        gains in fnRefCube were found.

    Returns
    -------
    deltaHeightMap : array_like
        2-D array of the difference in actuator heights for the two command
        maps provided.
    """
    check.twoD_array(commandMapBefore, 'commandMapBefore', ValueError)
    check.twoD_array(commandMapAfter, 'commandMapAfter', ValueError)
    if not np.allclose(np.array(commandMapBefore).shape,
                       np.array(commandMapAfter).shape):
        raise ValueError('Both command maps must have the same shape.')

    heightMapBefore = compute_heights_for_command_map(commandMapBefore,
                                                      fnRefCube,
                                                      fnRefCommandVec)
    heightMapAfter = compute_heights_for_command_map(commandMapAfter,
                                                     fnRefCube,
                                                     fnRefCommandVec)
    deltaHeightMap = heightMapAfter - heightMapBefore

    return deltaHeightMap


def compute_commands_for_height_map(heightMap, fnRefCube, fnRefCommandVec):
    """
    Compute actuator commands for a 2-D array of actuator heights.

    This performs a cubic spline interpolation of data from the reference
    datacube of measured heights vs voltage. The interpolator function is
    simply scipy.interpolate.interp1d(), called separately for each actuator.

    Parameters
    ----------
    heightMap : array_like
        2-D array of actuator heights (not surface heights) to convert to
        actuator commands.
    fnRefCube : str
        Name of the FITS file containing the datacube with measured DM gains
        at the DM commands specified by fnRefCommandVec.
    fnRefCommandVec : str
        Name of the FITS file containing the DM command values at which the
        gains in fnRefCube were found.

    Returns
    -------
    commandMap : array_like
        2-D array of actuator commands corresponding to the given heights.

    """
    check.twoD_array(heightMap, 'heightMap', ValueError)
    nRows = heightMap.shape[0]
    nCols = heightMap.shape[1]

    refHeightCube = fits.getdata(fnRefCube)
    check.threeD_array(refHeightCube, 'refHeightCube', ValueError)
    if nRows != refHeightCube.shape[2]:
        raise ValueError('Number of actuator columns must be consistent.')
    if nCols != refHeightCube.shape[1]:
        raise ValueError('Number of actuator rows must be consistent.')

    refCommandVec = fits.getdata(fnRefCommandVec)
    check.oneD_array(refCommandVec, 'refCommandVec', ValueError)
    verify_vector_has_unique_values(refCommandVec)
    verify_vector_is_presorted(refCommandVec)

    commandMap = np.zeros((nRows, nCols))
    for iCol in range(nCols):
        for iRow in range(nRows):
            interpolator = interp1d(refHeightCube[:, iRow, iCol],
                                    refCommandVec,
                                    kind='cubic',
                                    bounds_error='true',
                                    )
            commandMap[iRow, iCol] = interpolator(heightMap[iRow, iCol])

    return commandMap


def compute_heights_for_command_map(commandMap, fnRefCube, fnRefCommandVec):
    """
    Compute DM actuator heights for an array of DM commands.

    This performs a cubic spline interpolation of data from the reference
    datacube of measured heights vs voltage. The interpolator function is
    simply scipy.interpolate.interp1d(), called separately for each actuator.

    This function does not:
    - allow extrapolation; a ValueError is raised instead.
    - multiply commands by a gain to get heights.
    - use the influence function to compute the heights.

    Parameters
    ----------
    commandMap : array_like
        The array of absolute DM commands about which to compute the gainmap.
        2-D array of floats.
    fnRefCube : str
        Name of the FITS file containing the datacube with measured DM gains
        at the DM commands specified by fnRefCommandVec.
    fnRefCommandVec : str
        Name of the FITS file containing the DM command values at which the
        gains in fnRefCube were found.


    Returns
    -------
    heightMap : array_like
        2-D array of DM actuator heights corresponding to the provided
        command map. This is not the same as a surface height map, which
        would require use of the influence function.

    """
    check.twoD_array(commandMap, 'commandMap', ValueError)
    nRows = commandMap.shape[0]
    nCols = commandMap.shape[1]

    refHeightCube = fits.getdata(fnRefCube)
    check.threeD_array(refHeightCube, 'refHeightCube', ValueError)
    if nRows != refHeightCube.shape[2]:
        raise ValueError('Number of actuator columns must be consistent.')
    if nCols != refHeightCube.shape[1]:
        raise ValueError('Number of actuator rows must be consistent.')

    refCommandVec = fits.getdata(fnRefCommandVec)
    check.oneD_array(refCommandVec, 'refCommandVec', ValueError)
    verify_vector_has_unique_values(refCommandVec)
    verify_vector_is_presorted(refCommandVec)

    heightMap = np.zeros((nRows, nCols))
    for iCol in range(nCols):
        for iRow in range(nRows):
            interpolator = interp1d(refCommandVec,
                                    refHeightCube[:, iRow, iCol],
                                    kind='cubic',
                                    bounds_error='true',
                                    )
            heightMap[iRow, iCol] = interpolator(commandMap[iRow, iCol])

    return heightMap


def compute_gain_cube_from_height_cube(fnRefCube, fnRefCommandVec):
    """
    Compute the cube of gains from stored cube of heights.

    Parameters
    ----------
    fnRefCube : str
        Name of the FITS file containing the datacube with measured DM gains
        at the DM commands specified by fnRefCommandVec.
    fnRefCommandVec : str
        Name of the FITS file containing the DM command values at which the
        gains in fnRefCube were found.

    Returns
    -------
    gainCube : array_like
        Datacube of DM actuator gains

    """
    diffVec, meanVec = compute_diff_and_mean_of_vector(fnRefCommandVec)

    refHeightCube = fits.getdata(fnRefCube)
    check.threeD_array(refHeightCube, 'refHeightCube', ValueError)

    nSlicesHeight = refHeightCube.shape[0]
    nRows = refHeightCube.shape[1]
    nCols = refHeightCube.shape[2]
    if (diffVec.size+1) != nSlicesHeight:
        raise ValueError(('Number of reference slices must equal the number '
                          'of reference commands.'))

    nSlicesGain = nSlicesHeight - 1
    gainCube = np.zeros((nSlicesGain, nRows, nCols))
    for ii in range(nSlicesGain):
        gainCube[ii, :, :] = (
            (refHeightCube[ii+1, :, :] - refHeightCube[ii, :, :]) /
            diffVec[ii]
        )

    return gainCube


def compute_diff_and_mean_of_vector(fnVec):
    """
    Compute difference and mean of neighboring values in a vector.

    Because this is used on the stored DM commands used for gain
    calculations, a ValueError is raised if the values are not
    unique or if they do not monotonically increase.

    Parameters
    ----------
    fnVec : str
        Name of the FITS file containing the vector of floats.


    Returns
    -------
    diffVec : array_like
        1-D array of the differences between neighboring values of vecIn.
    meanVec : array_like
        1-D array of the mean of neighboring values of vecIn.

    """
    vecIn = fits.getdata(fnVec)

    # Checks on vecIn
    check.oneD_array(vecIn, 'vecIn', ValueError)
    verify_vector_has_unique_values(vecIn)
    verify_vector_is_presorted(vecIn)

    nOut = len(vecIn) - 1
    diffVec = np.zeros(nOut)
    meanVec = np.zeros(nOut)
    for ii in range(nOut):
        diffVec[ii] = vecIn[ii+1] - vecIn[ii]
        meanVec[ii] = (vecIn[ii+1] + vecIn[ii])/2

    return diffVec, meanVec


def verify_vector_has_unique_values(vecIn):
    """Check that a vector has all unique values."""
    # Verify is a vector
    check.oneD_array(vecIn, 'vecIn', ValueError)

    # Verify unique values
    vecIn = np.array(vecIn)
    valuesAreUnique = (vecIn.size == np.unique(vecIn).size)
    if not valuesAreUnique:
        raise ValueError('Vector must have unique values.')

    return None


def verify_vector_is_presorted(vecIn):
    """Check that a vector has monotonically increasing values."""
    # Verify is a vector
    check.oneD_array(vecIn, 'vecIn', ValueError)

    # Verify sorted values
    vecIn = np.array(vecIn)
    valuesAreSorted = np.allclose(vecIn,
                                  np.sort(vecIn),
                                  atol=np.finfo(float).eps)
    if not valuesAreSorted:
        raise ValueError('Vector must have monotonically increasing values.')

    return None
