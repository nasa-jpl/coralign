"""Functions for computing gain maps or height maps of DM actuators."""
import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits

from coralign.util import check
import coralign.gainmap.gm_util as gmu


def compute_starting_commands_for_flight(flatCommandMap,
                                         deltaHeightMap,
                                         fnRefCube,
                                         fnRefCommandVec):
    """
    Compute starting commands for EFC from a flat map and a delta height map.

    Parameters
    ----------
    flatCommandMap : array_like
        2-D array of DM actuator commands for the flattened wavefront setting.
    deltaHeightMap : array_like
        2-D array of differenced actuator heights, not to be confused with
        surface heights.
    fnRefCube : str
        Name of the FITS file containing the datacube with measured DM gains
        at the DM commands specified by fnRefCommandVec.
    fnRefCommandVec : str
        Name of the FITS file containing the DM command values at which the
        gains in fnRefCube were found.

    Returns
    -------
    startingCommandMap : array_like
        2-D array of DM actuator commands about which to start EFC.

    """
    check.twoD_array(flatCommandMap, 'flatCommandMap', ValueError)
    check.twoD_array(deltaHeightMap, 'deltaHeightMap', ValueError)
    if not np.allclose(np.array(flatCommandMap).shape,
                       np.array(deltaHeightMap).shape):
        raise ValueError('flatMap and deltaHeightMap must have same shape.')

    flatHeightMap = gmu.compute_heights_for_command_map(flatCommandMap,
                                                        fnRefCube,
                                                        fnRefCommandVec)
    startingHeightMap = flatHeightMap + deltaHeightMap

    # Restrict heights to stay within known height range so that no
    # out-of-bounds exceptions occur.
    # There are other checks on the reference vector to make sure that the
    # datacube slices are sorted from smallest deflection to largest.
    refHeightCube = fits.getdata(fnRefCube)
    check.threeD_array(refHeightCube, 'refHeightCube', ValueError)
    maxHeightMap = refHeightCube[-1, :, :]
    minHeightMap = refHeightCube[0, :, :]
    startingHeightMap[startingHeightMap < minHeightMap] = \
        minHeightMap[startingHeightMap < minHeightMap]
    startingHeightMap[startingHeightMap > maxHeightMap] = \
        maxHeightMap[startingHeightMap > maxHeightMap]

    startingCommandMap = gmu.compute_commands_for_height_map(startingHeightMap,
                                                             fnRefCube,
                                                             fnRefCommandVec)

    return startingCommandMap


def compute_gainmap(commandMap, fnRefCube, fnRefCommandVec):
    """
    Compute a gainmap for absolute DM commands based on reference data.

    Finds the gain map by linearly interpolating between gain values
    computed via finite differencing of a data cube of heights. Each
    actuator is independent and thus has a separate call to
    scipy.interpolate.interp1d(). Gains are found by extrapolation if
    the absolute DM command is above or below the bounding values for
    which gains were computed via finite differencing.

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
    gainMap : array_like
        2-D array of DM actuator gains about the provided command map.

    """
    _, meanCommandVec = gmu.compute_diff_and_mean_of_vector(fnRefCommandVec)
    gainCube = gmu.compute_gain_cube_from_height_cube(fnRefCube,
                                                      fnRefCommandVec)

    check.twoD_array(commandMap, 'commandMap', ValueError)
    nRows = commandMap.shape[0]
    nCols = commandMap.shape[1]
    if nRows != gainCube.shape[2]:
        raise ValueError('Number of actuator columns must be consistent.')
    if nCols != gainCube.shape[1]:
        raise ValueError('Number of actuator rows must be consistent.')

    gainMap = np.zeros((nRows, nCols))
    for iCol in range(nCols):
        for iRow in range(nRows):
            interpolator = interp1d(meanCommandVec,
                                    gainCube[:, iRow, iCol],
                                    kind='linear',
                                    fill_value='extrapolate',
                                    )
            gainMap[iRow, iCol] = interpolator(commandMap[iRow, iCol])

    return gainMap
