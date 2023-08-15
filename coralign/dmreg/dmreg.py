"""Module to hold methods for the DMREG package."""
import os
import numpy as np
from astropy.io import fits
from scipy.ndimage import binary_erosion

from coralign.util.ampthresh import ampthresh
import coralign.util.check as check
from coralign.util.dmhtoph import dmhtoph, volts_to_dmh
from coralign.util.findopt import find_optimum_1d, find_optimum_2d
from coralign.util.pad_crop import pad_crop as inin
from coralign.util.loadyaml import loadyaml
from coralign.dmreg.dr_util import remove_piston_tip_tilt

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))


def calc_software_mask(amp0, ph0, amp1, ph1, whichDM, nErode,
                       deltaPhaseThresh):
    """
    Compute which subset of pixels to use during DM registration.

    For DM registration, first compute which pixels to use later when computing
    translation, rotation, and scale. Using AMPTHRESH alone is not sufficient
    because that doesn't take into account dead actuators. This function
    expects an actuated piston between the two different phase retrievals, and
    throws out pixels where the phase exceeds some specified variation from
    that piston value (either because there is a non-moving actuator there, or
    the phase estimate was unreliable there). There is also an option to erode
    (pad) the pupil obscurations to eliminate any bad phase estimates at the
    edges of the pupil obscurations; this also enlarges the masked areas for
    dead actuators.

    Parameters
    ----------
    amp0 : array_like
        2-D amplitude map of the initial DM state.
    ph0 : array_like
        2-D unwrapped phase map of the initial DM state.
    amp1 : array_like
        2-D amplitude map of the perturbed DM state.
    ph1 : array_like
        2-D unwrapped phase map of the perturbed DM state.
    whichDM : {1, 2}
        Which DM to register. Can be 1 or 2 only.
    nErode : int
        Number of pixels to "erode" the AMPTHRESH'ed aperture by.
        Eroding means increasing the size of the obscurations.
        Units of ExCAM pixels.
    deltaPhaseThresh : float
        Minimum deviation from the median phase at which to threshold when
        making the usable pixel map. Must be positive. Units of radians.

    Returns
    -------
    usablePixelMap : array_like
        2-D boolean map of which pixels to use when fitting DM shapes for DM
        registration. Pixels near pupil obscurations and pixels at dead
        actuators are not included.
    """
    check.twoD_array(amp0, 'amp0', ValueError)
    check.twoD_array(ph0, 'ph0', ValueError)
    check.twoD_array(amp1, 'amp1', ValueError)
    check.twoD_array(ph1, 'ph1', ValueError)
    if amp0.shape != ph0.shape:
        raise ValueError('amp0 and ph0 must have same shape')
    if amp0.shape != amp1.shape:
        raise ValueError('amp0 and amp1 must have same shape')
    if ph0.shape != ph1.shape:
        raise ValueError('ph0 and ph1 must have same shape')
    check.positive_scalar_integer(whichDM, 'whichDM', ValueError)
    if whichDM not in (1, 2):
        raise ValueError('whichDM must have value of 1 or 2')
    check.nonnegative_scalar_integer(nErode, 'nErode', ValueError)
    check.real_positive_scalar(deltaPhaseThresh, 'deltaPhaseThresh',
                               ValueError)

    phDiff = ph1 - ph0

    # Remove piston, tip, and tilt
    usablePixelMap = ampthresh(amp0)
    phDiff = remove_piston_tip_tilt(phDiff, usablePixelMap)

    # Threshold to get rid of large phase deviations near the pupil obscuration
    # edges or at dead DM actuators.
    usablePixelMap[np.abs(phDiff) > deltaPhaseThresh] = False

    # Enlarge (erode) the pupil obscurations to block out bad or large phase
    # values near the edges
    usablePixelMap = binary_erosion(usablePixelMap,
                                structure=np.ones((1+2*nErode, 1+2*nErode)))

    return usablePixelMap


def calc_translation_clocking(
        amp0, ph0, amp1, ph1, usablePixelMap, whichDM, deltaV, 
        xOffsetPupil, yOffsetPupil, dm_fn, fn_fit_params, lam,
        data_path=LOCAL_PATH):
    """
    Compute the rotation and lateral offsets of the DM grid.

    Compute the rotation of the DM grid in degrees counterclockwise relative to
    the EXCAM x- and y-axes, and compute the translation of the DM grid from
    the center pixel of the array in EXCAM pixels, in both EXCAM x- and y-axes.

    The recommended delta DM command for this function is a pair of uniformly
    actuated rows and a pair of uniformly actuated columns symmetric about the
    center of the DM. A simulation-tested example of this is in
    `coralign/dmreg/testdata/delta_DM_V_for_trans_rot_calib.fits`, but note
    that this command map should be adjusted to counteract non-uniform actuator
    gains. The symmetry is desired to make the DM shape insensitive to parity
    flips or 90-, 180-, and 270-degree rotations. The two actuated rows/columns
    are not neighboring because the Roman pupil struts would mostly block them
    along one dimension, and because the fitting seems to do better with
    actuated lines that are one (instead of two) actuators wide. The smoothness
    of the actuated lines seems to help as well--diagonal actuated lines are
    bumpier and give lower accuracy estimates.

    Other patterns may be used--no exception will be thrown--but the
    performance of this function has not been tested with them and cannot be
    guaranteed.

    For this function: all filenames must be relative paths.  This is
    different from most other functions in Calibration, and is true for
    both input arguments and filenames in YAML files.  They will be relative
    to the value supplied in data_path, which is the location of this module
    by default unless specified otherwise.

    Parameters
    ----------
    amp0 : array_like
        2-D amplitude map of the initial DM state.
    ph0 : array_like
        2-D unwrapped phase map of the initial DM state.
    amp1 : array_like
        2-D amplitude map of the perturbed DM state.
    ph1 : array_like
        2-D unwrapped phase map of the perturbed DM state.
    usablePixelMap : array_like
        2-D boolean array indicating which image pixels to use for DM
        registration.
    whichDM : {1, 2}
        Which DM to register. Can be 1 or 2 only.
    deltaV : array_like
        2-D array of delta DM commands for the translation and clocking
        registration. Units of volts.
    xOffsetPupil, yOffsetPupil : float
        Lateral offsets of the pupil relative to the ExCAM image. Units of
        ExCAM pixels.
    dm_fn : str
        Name of the YAML file containing pre-registration calibration data for
        DMs 1 and 2. The gainmap and influence function should be known ahead
        of DM registration. To make a model-based DM surface for comparison,
        approximate, a priori scaling values need to be in this file.
    fn_fit_params : str
        Name of the YAML file with the tuning parameters for translation and
        rotation.
    lam : float
        Center wavelength for the phase retrieval maps. Units of meters.
    data_path : str
        Directory to serve as a base for relative paths with YAML files.
        If not supplied, defaults to directory containing this function. If
        using versions of the YAMLs delivered with the repository, the default
        will point to data files also delivered with the repository.

    Returns
    -------
    xOffsetEst, yOffsetEst : float
        Estimated x- and y-offsets of the DM grid center from the center pixel
        in the ExCAM image.
    clockEst : float
        Clocking estimate of the DM grid relative to the ExCAM image.
        Units of degrees.
    """
    check.twoD_array(amp0, 'amp0', ValueError)
    check.twoD_array(ph0, 'ph0', ValueError)
    check.twoD_array(amp1, 'amp1', ValueError)
    check.twoD_array(ph1, 'ph1', ValueError)
    check.twoD_array(usablePixelMap, 'usablePixelMap', ValueError)
    if amp0.shape != ph0.shape:
        raise ValueError('amp0 and ph0 must have same shape')
    if amp0.shape != amp1.shape:
        raise ValueError('amp0 and amp1 must have same shape')
    if amp0.shape != ph1.shape:
        raise ValueError('amp0 and ph1 must have same shape')
    if amp0.shape != usablePixelMap.shape:
        raise ValueError('amp0 and usablePixelMap must have same shape')
    check.positive_scalar_integer(whichDM, 'whichDM', ValueError)
    if whichDM not in (1, 2):
        raise ValueError('whichDM must have value of 1 or 2')
    check.real_scalar(xOffsetPupil, 'xOffsetPupil', ValueError)
    check.real_scalar(yOffsetPupil, 'yOffsetPupil', ValueError)
    check.twoD_square_array(deltaV, 'deltaV', ValueError)
    check.real_positive_scalar(lam, 'lam', ValueError)

    # Load Fitting Parameter Values
    fn_fit_params_fullpath = os.path.join(data_path, fn_fit_params)
    fitParamDict = loadyaml(fn_fit_params_fullpath)
    nIterBoth = fitParamDict['nIterBoth']
    offsetMax = fitParamDict['offsetMax']
    nOffset = fitParamDict['nOffset']
    nIterOffset = fitParamDict['nIterOffset']
    shrinkFac = fitParamDict['shrinkFac']
    clockMaxDeg = fitParamDict['clockMaxDeg']
    nClock = fitParamDict['nClock']
    nIterClock = fitParamDict['nIterClock']
    # Check Fitting Parameters
    check.positive_scalar_integer(nIterBoth, 'nIterBoth', ValueError)
    check.real_scalar(offsetMax, 'offsetMax', ValueError)
    check.positive_scalar_integer(nOffset, 'nOffset', ValueError)
    check.positive_scalar_integer(nIterOffset, 'nIterOffset', ValueError)
    check.real_positive_scalar(shrinkFac, 'shrinkFac', ValueError)
    if shrinkFac > 1:
        raise ValueError('shrinkFac must be less than or equal to 1.')
    check.real_scalar(clockMaxDeg, 'clockMaxDeg', ValueError)
    check.positive_scalar_integer(nClock, 'nClock', ValueError)
    check.positive_scalar_integer(nIterClock, 'nIterClock', ValueError)

    # Load Initial DM Calibration Data
    dm_fn_fullpath = os.path.join(data_path, dm_fn)
    dmDict = loadyaml(dm_fn_fullpath)
    if whichDM == 1:
        dm = dmDict["dms"]["DM1"]
    elif whichDM == 2:
        dm = dmDict["dms"]["DM2"]
    dmr = dm['registration']
    flipx = dmr['flipx']  # DM pattern for trans and rot is flipx agnostic
    gainfn = os.path.join(data_path, dm['voltages']['gainfn'])
    gainmap = fits.getdata(gainfn)
    inffn = os.path.join(data_path, dmr['inffn'])
    inf_func = fits.getdata(inffn)
    dmrad = volts_to_dmh(gainmap, deltaV, lam)

    phDiffMeas = ph1 - ph0
    phDiffMeas = np.angle(np.exp(1j*phDiffMeas))  # eliminate phase wrapping
    phDiffMeas = remove_piston_tip_tilt(phDiffMeas, usablePixelMap)

    # Initializations for the nested loops
    xOffsetVec = np.linspace(-offsetMax, offsetMax, nOffset)
    yOffsetVec = np.linspace(-offsetMax, offsetMax, nOffset)
    costMat = np.zeros((nOffset, nOffset))
    xOffsetEst = xOffsetPupil
    yOffsetEst = yOffsetPupil
    clockEst = dmr['thact']
    [nrow, ncol] = amp0.shape

    # Loop over translation and clocking
    for iterBoth in range(nIterBoth):

        # Fit Translation
        for iterOffset in range(nIterOffset):
            for iy, yOffsetTemp in enumerate(yOffsetVec):
                for ix, xOffsetTemp in enumerate(xOffsetVec):
                    phDiffSim = dmhtoph(nrow, ncol, dmrad, dmr['nact'],
                                        inf_func, dmr['ppact_d'],
                                        dmr['ppact_cx'], dmr['ppact_cy'],
                                        xOffsetEst+xOffsetTemp,
                                        yOffsetEst+yOffsetTemp, clockEst,
                                        flipx)
                    phDiffSim *= 2  # surface to wavefront
                    phDiffSim = remove_piston_tip_tilt(phDiffSim,
                                                       usablePixelMap)

                    costMat[iy, ix] = np.sum(usablePixelMap *
                                             np.abs(phDiffMeas - phDiffSim))

            # Use argmin the first time only,
            # and then switch to a quadratic fit.
            if iterOffset == 0 and nIterOffset > 1:
                bestInds = np.unravel_index(np.argmin(costMat), costMat.shape)
                xOffsetTemp = xOffsetVec[bestInds[1]]
                yOffsetTemp = yOffsetVec[bestInds[0]]
            else:
                [xOffsetTemp, yOffsetTemp] = \
                    find_optimum_2d(xOffsetVec, yOffsetVec, costMat,
                                    np.ones_like(costMat))
            xOffsetEst += xOffsetTemp
            yOffsetEst += yOffsetTemp
            xOffsetVec = (shrinkFac**iterOffset) *\
                np.linspace(-offsetMax, offsetMax, nOffset)
            yOffsetVec = (shrinkFac**iterOffset) *\
                np.linspace(-offsetMax, offsetMax, nOffset)

        # Fit Rotation
        clockVec = np.linspace(-clockMaxDeg, clockMaxDeg, nClock)
        costVec = np.zeros((nClock,))
        for iterClock in range(nIterClock):
            for ic, deltaClock in enumerate((shrinkFac**iterClock)*clockVec):
                phDiffSim = dmhtoph(nrow, ncol, dmrad, dmr['nact'], inf_func,
                                    dmr['ppact_d'],
                                    dmr['ppact_cx'],
                                    dmr['ppact_cy'],
                                    xOffsetEst, yOffsetEst,
                                    deltaClock+clockEst, flipx)
                costVec[ic] = np.sum(usablePixelMap *
                                     np.abs(phDiffMeas - phDiffSim))

            # Use argmin the first time only,
            # and then switch to a quadratic fit.
            if iterClock == 0 and nIterClock > 1:
                bestInd = np.argmin(costVec)
                clockEst += clockVec[bestInd]
            else:
                clockEst += find_optimum_1d(clockVec, costVec)

    return xOffsetEst, yOffsetEst, clockEst


def calc_scale(amp0, ph0, amp1, ph1, usablePixelMap, whichDM, deltaV,
               dm_fn, fn_fit_params, lam, xOffset, yOffset, clocking,
               data_path=LOCAL_PATH):
    """
    Compute the scaling of the DM grid in x and y.

    Compute the scaling of the DM grid, in EXCAM pixels per actuator, along
    both x- and y-axes by using parabolic fits.

    The recommended delta DM command for the scaling function is a fully
    inscribed within the pupil, two-actuator-wide outline of a square
    centered on the DM. A simulation-tested example of this is in
    `coralign/dmreg/testdata/delta_DM_V_for_scale_calib.fits`, but note that
    this command map should be adjusted to counteract non-uniform actuator
    gains. The symmetry is desired to make the DM shape insensitive to parity
    flips or 90-, 180-, and 270-degree rotations. The outline is two actuators
    wide (instead of just one) so that the algorithm will work over a greater
    range of scale factors.

    Other patterns may be used--no exception will be thrown--but the
    performance of this function has not been tested with them and cannot be
    guaranteed.

    For this function: all filenames must be relative paths.  This is
    different from most other functions in Calibration, and is true for
    both input arguments and filenames in YAML files.  They will be relative
    to the value supplied in data_path, which is the location of this module
    by default unless specified otherwise.

    Parameters
    ----------
    amp0 : array_like
        2-D amplitude map of the initial DM state.
    ph0 : array_like
        2-D unwrapped phase map of the initial DM state.
    amp1 : array_like
        2-D amplitude map of the perturbed DM state.
    ph1 : array_like
        2-D unwrapped phase map of the perturbed DM state.
    usablePixelMap : array_like
        2-D boolean array indicating which image pixels to use for DM
        registration.
    whichDM : {1, 2}
        Which DM to register. Can be 1 or 2 only.
    deltaV : array_like
        2-D array of delta DM commands for the scale registration.
        Units of volts.
    dm_fn : str
        Name of the YAML file containing pre-registration calibration data for
        DMs 1 and 2. The gainmap and influence function should be known ahead
        of DM registration. To make a model-based DM surface for comparison,
        approximate, a priori scaling values need to be in this file.
    fn_fit_params : str
        Name of the YAML file with the scale tuning parameters.
    shrinkFac : float
        Factor by which to change maxDeltaScaleFac after each iteration of the
        fit. Must be >0.0 and <= 1.0.
    lam : float
        Center wavelength for the phase retrieval maps. Units of meters.
    xOffset, yOffset : float
        Lateral offsets of the DM grid relative to the ExCAM image center.
        Units of ExCAM pixels.
    clocking : float
        Counter-clockwise rotation of the DM grid with respect to the ExCAM
        pixel array. Units of degrees.
    data_path : str
        Directory to serve as a base for relative paths with YAML files.
        If not supplied, defaults to directory containing this function. If
        using versions of the YAMLs delivered with the repository, the default
        will point to data files also delivered with the repository.

    Returns
    -------
    ppact_cx_est, ppact_cy_est : float
        Estimates of the DM scaling along the x- and y-axes. Units of ExCAM
        pixels per actuator.
    """
    check.twoD_array(amp0, 'amp0', ValueError)
    check.twoD_array(ph0, 'ph0', ValueError)
    check.twoD_array(amp1, 'amp1', ValueError)
    check.twoD_array(ph1, 'ph1', ValueError)
    check.twoD_array(usablePixelMap, 'usablePixelMap', ValueError)
    if amp0.shape != ph0.shape:
        raise ValueError('amp0 and ph0 must have same shape')
    if amp0.shape != amp1.shape:
        raise ValueError('amp0 and amp1 must have same shape')
    if amp0.shape != ph1.shape:
        raise ValueError('amp0 and ph1 must have same shape')
    if amp0.shape != usablePixelMap.shape:
        raise ValueError('amp0 and usablePixelMap must have same shape')
    check.positive_scalar_integer(whichDM, 'whichDM', ValueError)
    if whichDM not in (1, 2):
        raise ValueError('whichDM must have value of 1 or 2')
    check.twoD_square_array(deltaV, 'deltaV', ValueError)
    check.real_positive_scalar(lam, 'lam', ValueError)
    check.real_scalar(xOffset, 'xOffset', ValueError)
    check.real_scalar(yOffset, 'yOffset', ValueError)
    check.real_scalar(clocking, 'clocking', ValueError)

    # Load fit parameters
    fn_fit_fullpath = os.path.join(data_path, fn_fit_params)
    fitDict = loadyaml(fn_fit_fullpath)
    maxDeltaScaleFac = fitDict['maxDeltaScaleFac']
    nScale = fitDict['nScale']
    nIter = fitDict['nIter']
    shrinkFac = fitDict['shrinkFac']
    check.real_scalar(maxDeltaScaleFac, 'maxDeltaScaleFac', ValueError)
    check.positive_scalar_integer(nScale, 'nScale', ValueError)
    check.positive_scalar_integer(nIter, 'nIter', ValueError)
    check.real_positive_scalar(shrinkFac, 'shrinkFac', ValueError)
    if shrinkFac > 1:
        raise ValueError('shrinkFac must be less than or equal to 1.')

    # Load DM Calibration Data
    dm_fn_fullpath = os.path.join(data_path, dm_fn)
    dmConfig = loadyaml(dm_fn_fullpath)
    if whichDM == 1:
        dm = dmConfig["dms"]["DM1"]
    elif whichDM == 2:
        dm = dmConfig["dms"]["DM2"]
    dmr = dm['registration']
    gainfn = os.path.join(data_path, dm['voltages']['gainfn'])
    gainmap = fits.getdata(gainfn)
    inffn = os.path.join(data_path, dmr['inffn'])
    inf_func = fits.getdata(inffn)
    dmrad = volts_to_dmh(gainmap, deltaV, lam)
    flipx = False  # DM registration pattern is flipx agnostic

    phDiffMeas = ph1 - ph0
    phDiffMeas = np.angle(np.exp(1j*phDiffMeas))  # eliminate phase wrapping
    phDiffMeas = remove_piston_tip_tilt(phDiffMeas, usablePixelMap)

    xScaleVec = np.linspace(1-maxDeltaScaleFac, 1+maxDeltaScaleFac, nScale)
    yScaleVec = np.linspace(1-maxDeltaScaleFac, 1+maxDeltaScaleFac, nScale)
    costMat = np.zeros((nScale, nScale))
    xScale = 1  # initialize
    yScale = 1  # initialize
    for iter_ in range(nIter):
        for iy, yScaleTemp in enumerate(yScaleVec):
            for ix, xScaleTemp in enumerate(xScaleVec):
                phDiffSim = dmhtoph(amp0.shape[1], amp0.shape[0], dmrad,
                                    dmr['nact'], inf_func, dmr['ppact_d'],
                                    xScaleTemp*xScale*dmr['ppact_cx'],
                                    yScaleTemp*yScale*dmr['ppact_cy'],
                                    xOffset, yOffset,
                                    clocking, flipx)
                phDiffSim *= 2  # surface to wavefront
                phDiffSim = inin(phDiffSim, amp0.shape)
                phDiffSim = remove_piston_tip_tilt(phDiffSim, usablePixelMap)

                costMat[iy, ix] = np.sum(usablePixelMap *
                                         np.abs(phDiffMeas - phDiffSim))

        if iter_ == 0 and nIter > 1:
            bestInds = np.unravel_index(np.argmin(costMat), costMat.shape)
            xScaleTemp = xScaleVec[bestInds[1]]
            yScaleTemp = yScaleVec[bestInds[0]]
        else:
            [xScaleTemp, yScaleTemp] = find_optimum_2d(
                xScaleVec, yScaleVec, costMat, np.ones_like(costMat))
        xScale = xScaleTemp*xScale
        yScale = yScaleTemp*yScale
        maxDeltaScaleFac = shrinkFac*maxDeltaScaleFac
        xScaleVec = np.linspace(1-maxDeltaScaleFac, 1+maxDeltaScaleFac, nScale)
        yScaleVec = np.linspace(1-maxDeltaScaleFac, 1+maxDeltaScaleFac, nScale)

    ppact_cx_est = xScale*dmr['ppact_cx']
    ppact_cy_est = yScale*dmr['ppact_cy']

    return ppact_cx_est, ppact_cy_est
