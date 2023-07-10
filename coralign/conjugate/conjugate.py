"""Functions for the CONJUGATE module."""
import os
import copy

import numpy as np
from scipy.ndimage import rotate
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import fminbound
from astropy.convolution import Box2DKernel, convolve
from astropy.io import fits

from coralign.util.ampthresh import ampthresh
from coralign.util import check
from coralign.util.dmhtoph import dmh_to_volts
from coralign.util.fit_surf_to_dm import fit_surf_to_dm, build_prefilter
from coralign.util.pad_crop import pad_crop as inin
from coralign.util.loadyaml import loadyaml
from coralign.util.math import rms, ceil_odd
from coralign.util.nollzernikes import gen_zernikes, fit_zernikes

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))


def flatten(cmdDM1, cmdDM2, cmdFCM, flagUseDMs, flagUseFCM, flagHighZern,
            lam, amp, ph, diamPupil, xOffset, yOffset,
            dm_fn, fcm_fn, flatten_params_fn, data_path=LOCAL_PATH):
    """
    Compute commands to flatten the given WFE using the FCM and/or both DMs.

    Then apply those delta commands to the existing DM and FCM commands.

    Dead actuators (i.e., those stuck at 0 volts) are accounted for. Any
    wavefront error assigned to a dead DM1 actuator gets reassigned to DM2,
    and vice versa.

    Cases:
    - FCM only
      - FCM corrects Z4 only.
    - DM1 and DM2 only
      - Only DM1 corrects >Zmax if flagHighZern == True.
      - Z4-Zmax correction split between DM1 and DM2 such that RMS for each DM
      phase applied is same.
    - DM1 and DM2 and FCM
      - FCM corrects Z4 only
      - Only DM1 corrects >Zmax if flagHighZern == True.
      - Z5-Zmax correction split between DM1 and DM2 such that RMS for each DM
      phase applied is same.

    Parameters
    ----------
    cmdDM1, cmdDM2 : numpy ndarray
        2-D square arrays of total commands currently applied to DM1 and DM2
    cmdFCM : int
        Absolute coarse step position of the FCM.
    flagUseDMs : bool
        Boolean flag specifying whether to use both DM1 and DM2 here.
    flagUseFCM : bool
        Boolean flag specifying whether to use the FCM here.
    flagHighZern : bool
        Boolean flag for correcting aberrations above Z-maxNollZern or
        not from the provided WFE map.
    lam : float
        Wavelength, used only for phase-to-height conversion. Units of meters.
    amp : str
        2-D array of phase retrieval amplitude data.
    ph : str
        2-D array of phase retrieval phase data in units of radians.
        Must have same shape as amp.
    diamPupil : str
        Pupil diameter. Units of EXCAM pixels. Used for defining Zernike
        polynomials.
    xOffset, yOffset : float
        x- and y-offsets of the pupil from the image center pixel. Units of
        EXCAM pixels. Used for defining Zernike polynominals.
    dm_fn : str
        Name of the YAML file containing calibration data for DMs 1 and 2.
    fcm_fn : str
        Name of the YAML file containing conversion factor values for the FCM.
    flatten_params_fn : str
        Name of file containing static input values needed for the flatten()
        algorithm.
    data_path : str
        Directory to serve as a base for relative paths with YAML files.
        If not supplied, defaults to directory containing this function. If
        using versions of the YAMLs delivered with the repository, the default
        will point to data files also delivered with the repository.

    Returns
    -------
    cmdDM1, cmdDM2 : numpy ndarray
        2-D array of total DM commands with the new deltas included.
    cmdFCM : int
        Absolute coarse step position of the FCM.
    zernCoefVec : array_like
        Zernike mode coefficients fitted to the input WFE map. Used only for
        verifying output in tests and not used in general operations.
    """
    check.twoD_square_array(cmdDM1, 'cmdDM1', ValueError)
    check.real_array(cmdDM1, 'cmdDM1', ValueError)
    check.twoD_square_array(cmdDM2, 'cmdDM2', ValueError)
    check.real_array(cmdDM2, 'cmdDM2', ValueError)
    check.scalar_integer(cmdFCM, 'cmdFCM', ValueError)
    check.boolean(flagUseDMs, 'flagUseDMs', TypeError)
    check.boolean(flagUseFCM, 'flagUseFCM', TypeError)
    check.boolean(flagHighZern, 'flagHighZern', TypeError)
    check.real_positive_scalar(lam, 'lam', ValueError)
    check.twoD_array(amp, 'amp', ValueError)
    check.real_array(amp, 'amp', ValueError)
    check.twoD_array(ph, 'ph', ValueError)
    check.real_array(ph, 'ph', ValueError)
    check.real_positive_scalar(diamPupil, 'diamPupil', ValueError)
    check.real_scalar(xOffset, 'xOffset', ValueError)
    check.real_scalar(yOffset, 'yOffset', ValueError)
    check.string(dm_fn, 'dm_fn', TypeError)
    check.string(fcm_fn, 'fcm_fn', TypeError)
    check.string(flatten_params_fn, 'flatten_params_fn', TypeError)
    check.string(data_path, 'data_path', TypeError)

    # Load DM Calibration Data
    dmConfig = loadyaml(dm_fn)
    dm = dmConfig["dms"]
    dm1r = dm['DM1']['registration']
    dm2r = dm['DM2']['registration']
    # Determine the region of the array corresponding to the DM surface
    # for use in the fitting.
    dpad = 2
    nxres1 = round(dpad*dm1r['nact']*dm1r['ppact_cx'])
    nyres1 = round(dpad*dm1r['nact']*dm1r['ppact_cy'])
    wfeShape1 = (nyres1, nxres1)
    nxres2 = round(dpad*dm2r['nact']*dm2r['ppact_cx'])
    nyres2 = round(dpad*dm2r['nact']*dm2r['ppact_cy'])
    wfeShape2 = (nyres2, nxres2)
    nxresMax = int(np.max((nxres1, nxres2)))
    nyresMax = int(np.max((nyres1, nyres2)))
    wfeShapeMax = (nyresMax, nxresMax)

    # Load FCM Calibration Data
    fcmConfig = loadyaml(fcm_fn)
    focusPerStroke = fcmConfig["focusPerStroke"]  # [m RMS per um of stroke]
    strokePerStep = fcmConfig["strokePerStep"]  # [um stroke per coarse step]
    check.real_scalar(focusPerStroke, 'focusPerStroke', ValueError)
    check.real_scalar(strokePerStep, 'strokePerStep', ValueError)
    # -          0.459 nm rms = 1 um stroke
    # -          5.3um per coarse FCM step.

    # Load fixed settings for CONJUGATE
    flattenDict = loadyaml(flatten_params_fn)
    maxNollZern = flattenDict["maxNollZern"]
    nIterSmooth = flattenDict["nIterSmooth"]
    nBin = flattenDict["nBin"]
    arrayExtFac = flattenDict["arrayExtFac"]
    check.positive_scalar_integer(maxNollZern, 'maxNollZern', ValueError)
    check.positive_scalar_integer(nIterSmooth, 'nIterSmooth', ValueError)
    check.positive_scalar_integer(nBin, 'nBin', ValueError)
    check.real_positive_scalar(arrayExtFac, 'arrayExtFac', ValueError)

    # Crop or pad the PR maps
    wfeIn = inin(ph, wfeShapeMax)
    ampIn = inin(amp, wfeShapeMax)
    # Use the amplitude map to define which pixels to use from the PR map.
    usablePixelMap = ampthresh(ampIn, nBin=nBin)

    zernCoefVec = fit_zernikes(wfeIn, usablePixelMap, maxNollZern,
                               diamPupil, xOffset, yOffset)

    wfeZ1to3, wfeZ4, wfeZ4or5toZmax = gen_split_wfe(
        flagUseFCM, maxNollZern, zernCoefVec, diamPupil, wfeShapeMax,
        xOffset, yOffset)
    wfeAboveZmax = (wfeIn - wfeZ4or5toZmax - wfeZ4 - wfeZ1to3)*usablePixelMap

    #  Equally Distribute RMS Phase Between DM1 and DM2
    if flagUseDMs:
        rmsAboveZmax = rms(wfeAboveZmax[usablePixelMap])
        rmsZ4or5toZmax = rms(wfeZ4or5toZmax[usablePixelMap])

        if flagHighZern:
            if rmsZ4or5toZmax > rmsAboveZmax:
                alphaBest = fminbound(phase_cost_function, 0., 1.,
                                      args=(wfeZ4or5toZmax, wfeAboveZmax,
                                            usablePixelMap), disp=0)
            else:
                alphaBest = 0
                pass

            DM1Phase = alphaBest*wfeZ4or5toZmax + wfeAboveZmax*usablePixelMap
            DM2Phase = (1-alphaBest) * wfeZ4or5toZmax

        else:  # Split WFE evenly between DMs if just middle Zernikes
            DM1Phase = 0.5 * wfeZ4or5toZmax
            DM2Phase = 0.5 * wfeZ4or5toZmax

    else:  # Don't use either DM
        DM1Phase = np.zeros(wfeShape1)
        DM2Phase = np.zeros(wfeShape2)

    # Create the Zernike Maps beyond the edges of the DM and limit
    #  phase to the range of the actual illuminated area
    nArraySmall = ceil_odd(arrayExtFac*diamPupil)
    usablePixelMapCrop = inin(usablePixelMap, (nArraySmall, nArraySmall))

    DM1Phase0 = inin(DM1Phase, (nArraySmall, nArraySmall))
    minPhase1 = np.min(DM1Phase0[usablePixelMapCrop == 1])
    maxPhase1 = np.max(DM1Phase0[usablePixelMapCrop == 1])
    DM1Phase0[DM1Phase0 < minPhase1] = minPhase1
    DM1Phase0[DM1Phase0 > maxPhase1] = maxPhase1

    DM2Phase0 = inin(DM2Phase, (nArraySmall, nArraySmall))
    minPhase2 = np.min(DM2Phase0[usablePixelMapCrop == 1])
    maxPhase2 = np.max(DM2Phase0[usablePixelMapCrop == 1])
    DM2Phase0[DM2Phase0 < minPhase2] = minPhase2
    DM2Phase0[DM2Phase0 > maxPhase2] = maxPhase2

    # Create phase values behind the struts to get a continuous phase. This
    # reduces WFE along the edges of the pupil obscurations when flattening
    # the phase.
    # Used only for aberrations >Zmax since Z1-Zmax are defined everywhere.
    surfDesDM1 = DM1Phase0.copy()/2.
    surfDesDM2 = DM2Phase0.copy()/2.

    if flagHighZern:
        windowWidth1 = round(np.max((dm1r['ppact_cx'], dm1r['ppact_cy'])))
        for _ in range(nIterSmooth):
            # smear out phase to fill in un-illuminated areas
            surfDesDM1 = smooth_surface(surfDesDM1, windowWidth1)
            # reset values within the pupil
            surfDesDM1[usablePixelMapCrop] = \
                DM1Phase0[usablePixelMapCrop].copy()/2.

        windowWidth2 = round(np.max((dm2r['ppact_cx'], dm2r['ppact_cy'])))
        for _ in range(nIterSmooth):
            # smear out phase to fill in un-illuminated areas
            surfDesDM2 = smooth_surface(surfDesDM2, windowWidth2)
            # reset values within the pupil
            surfDesDM2[usablePixelMapCrop] = \
                DM2Phase0[usablePixelMapCrop].copy()/2.

    # Compute new DM commands.
    hdm1 = conv_surf_to_dm_cmd(surfDesDM1, dm['DM1'],
                               data_path=data_path)  # radians
    hdm2 = conv_surf_to_dm_cmd(surfDesDM2, dm['DM2'],
                               data_path=data_path)  # radians

    # Ressign WFE from dead DM1 actuators to DM2
    dm1tiefn = os.path.join(data_path, dm['DM1']['voltages']['tiefn'])
    tiemap1 = fits.getdata(dm1tiefn)
    nact1 = dm1r['nact']
    isdeadmap1 = np.zeros((nact1, nact1))
    isdeadmap1[tiemap1 == -1] = 1
    h1to2 = isdeadmap1 * hdm1
    hdm2 += h1to2

    # Ressign WFE from dead DM2 actuators to DM1
    dm2tiefn = os.path.join(data_path, dm['DM2']['voltages']['tiefn'])
    tiemap2 = fits.getdata(dm2tiefn)
    nact2 = dm2r['nact']
    isdeadmap2 = np.zeros((nact2, nact2))
    isdeadmap2[tiemap2 == -1] = 1
    h2to1 = isdeadmap2 * hdm2
    hdm1 += h2to1

    # Zero out commands for dead actuators
    hdm1[isdeadmap1 == 1] = 0
    hdm2[isdeadmap2 == 1] = 0

    # Compute delta commands and updated total commands
    dm1gainfn = os.path.join(data_path, dm['DM1']['voltages']['gainfn'])
    dm1gainmap = fits.getdata(dm1gainfn)
    deltaCmdDM1 = dmh_to_volts(dm1gainmap, hdm1, lam)

    dm2gainfn = os.path.join(data_path, dm['DM2']['voltages']['gainfn'])
    dm2gainmap = fits.getdata(dm2gainfn)
    deltaCmdDM2 = dmh_to_volts(dm2gainmap, hdm2, lam)

    cmdDM1 = cmdDM1 - deltaCmdDM1
    cmdDM2 = cmdDM2 - deltaCmdDM2

    # Compute new FCM command
    if flagUseFCM:
        deltaCmdFCM = zernCoefVec[3]*lam/(2*np.pi)/focusPerStroke/strokePerStep
        cmdFCM = int(round(cmdFCM + deltaCmdFCM))

    return cmdDM1, cmdDM2, cmdFCM, zernCoefVec


def gen_split_wfe(flagUseFCM, maxNollZern, zernCoef, diamPupil, wfeShape,
                  xOffset, yOffset):
    """
    Generate WFE maps split into different sets of Zernike modes.

    Parameters
    ----------
    flagUseFCM: bool
        Boolean flag denoting whether Z4 will be corrected with the FCM
        or not (in which case the DM(s) will.)
    maxNollZern : int
        The maximum Noll Zernike mode to include as part of the low-
        order modes to split between DMs 1 and 2.
    zernCoefVec : numpy ndarray
        1-D array of Zernike coefficients. The coefficient for Noll mode 1 goes
        in position 0 of the array. The max number of Noll modes used is chosen
        as the size of this array.
    diamPupil : float
        (Unmasked) diameter of the pupil in the image. The Zernike maps will be
        generated to have this diameter. Units of pixels.
    wfeShape : array_like
        1-D array giving the shape of the WFE map arrays to produce.
    xOffset, yOffset : float
        x- and y-offsets of the pupil from the center pixel in the pupil
        image. Units of pixels.

    Returns
    -------
    wfeZ1to3 : numpy ndarray
        2-D WFE map for Zernike modes Z1-Z3. Used for verification only.
    wfeZ4 : numpy ndarray
        2-D WFE map for Zernike mode Z4. Used for verification only. Is zero
        if flagUseFCM == False.
    wfeZ4or5toZmax
        2-D WFE map for Zernike modes Z4-Zmax if flagUseFCM == False, or
        Z5-Zmax if flagUseFCM == True. Zmax is the largest Zernike mode
        specified by the length of the coefficients vector zernCoef.
    """
    check.boolean(flagUseFCM, 'flagUseFCM', TypeError)
    check.positive_scalar_integer(maxNollZern, 'maxNollZern', ValueError)
    check.oneD_array(zernCoef, 'zernCoef', ValueError)
    check.real_array(zernCoef, 'zernCoef', ValueError)
    # if not zernCoef.size >= 5:
    #     raise ValueError('zernCoef must have at least 5 values for correct' +
    #                      ' WFE map indexing.')
    check.real_positive_scalar(diamPupil, 'diamPupil', ValueError)
    check.oneD_array(wfeShape, 'wfeShape', ValueError)
    check.real_array(wfeShape, 'wfeShape', ValueError)
    if not np.array(wfeShape).size == 2:
        raise ValueError('wfeShape must contain two values')
    check.real_scalar(xOffset, 'xOffset', ValueError)
    check.real_scalar(yOffset, 'yOffset', ValueError)

    # Change zernCoef to length maxNollZern
    if zernCoef.size < maxNollZern:
        zernCoef0 = copy.copy(zernCoef)
        zernCoef = np.zeros((maxNollZern,))
        zernCoef[0:zernCoef0.size] = zernCoef0
    elif zernCoef.size > maxNollZern:
        zernCoef = zernCoef[0:maxNollZern]

    nArray = round(np.max(wfeShape))

    if flagUseFCM:
        zCoef = zernCoef[4::]
        Z4coef = zernCoef[3]  # FCM's component
        zSetDM1 = np.arange(5, maxNollZern+1)

        wfeZ4 = gen_zernikes(np.array([4]), np.array([Z4coef]), xOffset,
                             yOffset, diamPupil, nArray=nArray)
        wfeZ4 = inin(wfeZ4, wfeShape)
    else:
        zCoef = zernCoef[3::]
        Z4coef = 0.  # FCM's component
        zSetDM1 = np.arange(4, maxNollZern+1)

        wfeZ4 = np.zeros(wfeShape)

    wfeZ1to3 = gen_zernikes(np.array([1, 2, 3]), zernCoef[0:3], xOffset,
                            yOffset, diamPupil, nArray=nArray)
    wfeZ1to3 = inin(wfeZ1to3, wfeShape)

    wfeZ4or5toZmax = gen_zernikes(zSetDM1, zCoef, xOffset, yOffset,
                                  diamPupil, nArray=nArray)
    wfeZ4or5toZmax = inin(wfeZ4or5toZmax, wfeShape)

    return wfeZ1to3, wfeZ4, wfeZ4or5toZmax


def smooth_surface(arrayIn, windowWidth):
    """
    Smooth values in a 2-D array via convolution.

    The square convolution kernel has width "windowWidth".

    Parameters
    ----------
    arrayIn : array_like
        2-D, real-valued array to be smoothed
    windowWidth : float
        width of the square convolution kernel for smoothing

    Returns
    -------
    smoothedArray : numpy ndarray
        Smoothed version of the input 2-D map. Same shape as input.
    """
    check.twoD_array(arrayIn, 'arrayIn', ValueError)
    check.real_array(arrayIn, 'arrayIn', ValueError)
    check.real_positive_scalar(windowWidth, 'windowWidth', ValueError)

    smoothedArray = convolve(arrayIn, Box2DKernel(windowWidth))
    smoothedArray = inin(smoothedArray, arrayIn.shape)

    return smoothedArray


def conv_surf_to_dm_cmd(surfaceToFit, dm, data_path=LOCAL_PATH):
    """
    Compute the DM commands that recreate a given surface.

    All filenames may be absolute or relative paths.  If relative in input
    arguments, they will be relative to the current working directory, not to
    any particular location in Calibration.  If relative within YAML files
    (e.g. the inffn key in dm), this will be relative to
    the path in the data_path argument.

    Parameters
    ----------
    surfaceToFit : array_like
        2-D array of the surface heights for the DM to fit
    dm : dict
        Dictionary containing DM parameters and registration.
    data_path : str
        Directory to serve as a base for relative paths with YAML files.
        If not supplied, defaults to directory containing this function. If
        using versions of the YAMLs delivered with the repository, the default
        will point to data files also delivered with the repository.

    Returns
    -------
    Vout : numpy ndarray
        2-D array of DM voltage commands
    """
    # Check direct inputs
    check.twoD_array(surfaceToFit, 'surfaceToFit', ValueError)
    check.real_array(surfaceToFit, 'surfaceToFit', ValueError)
    if not isinstance(dm, dict):
        raise TypeError('dm must be a dict')

    dmReg = dm['registration']
    actPitch = dm['pitch']
    dx = dmReg['dx']
    dy = dmReg['dy']
    ppact_cx = dmReg['ppact_cx']
    ppact_cy = dmReg['ppact_cy']
    thact = dmReg['thact']
    nact = dmReg['nact']
    ppact_d = dmReg['ppact_d']
    inffn = os.path.join(data_path, dmReg['inffn'])
    flipx = dmReg['flipx']

    # Check values of the keys
    check.real_positive_scalar(actPitch, 'actPitch', ValueError)
    check.real_scalar(dx, 'dx', ValueError)
    check.real_scalar(dy, 'dy', ValueError)
    check.real_positive_scalar(ppact_cx, 'ppact_cx', ValueError)
    check.real_positive_scalar(ppact_cy, 'ppact_cy', ValueError)
    check.real_scalar(thact, 'thact', ValueError)
    check.positive_scalar_integer(nact, 'nact', ValueError)
    check.real_positive_scalar(ppact_d, 'ppact_d', ValueError)
    check.boolean(flipx, 'flipx', TypeError)
    if not isinstance(inffn, str):
        raise TypeError('inffn must be a string')
    try:
        infFunc = fits.getdata(inffn)
        check.twoD_square_array(infFunc, 'infFunc', ValueError)
    except OSError:
        raise OSError('Could not read influence function from %s' % inffn)

    dx_inf0 = actPitch/float(ppact_d)
    actres1 = actPitch/dx_inf0

    # Undo flip that dmhtoph does
    if flipx:
        surfaceToFit = np.fliplr(surfaceToFit)

    # Do rotation the way that dmhtoph does.
    surfaceToFit = rotate(surfaceToFit, thact, reshape=False)

    # Define coordinates for the surface
    [nY, nX] = surfaceToFit.shape
    if nX % 2 == 0:
        xsA = np.linspace(-nX/2, (nX/2)-1, nX) / ppact_cx
    else:
        xsA = np.linspace(-(nX-1)/2, (nX-1)/2, nX)/ppact_cx

    if nY % 2 == 0:
        ysA = np.linspace(-nY/2, (nY/2)-1, nY) / ppact_cy
    else:
        ysA = np.linspace(-(nY-1)/2, (nY-1)/2, nY) / ppact_cy

    ppact_new = 1
    nB = nact*ppact_new
    xsB = np.linspace(-(nB-1)/2, (nB-1)/2, nB)/ppact_new  # centered on DM
    ysB = xsB
    # RectBivariateSpline is MUCH faster than interp2d
    interp_spline = RectBivariateSpline(ysA, xsA, surfaceToFit)
    gridDerotAtActRes = interp_spline(ysB+dy/ppact_cy, xsB+dx/ppact_cx)

    sparseMat = build_prefilter(gridDerotAtActRes.shape[0],
                                gridDerotAtActRes.shape[1], infFunc, actres1)
    Vout = fit_surf_to_dm(gridDerotAtActRes, sparseMat)

    return Vout


def phase_cost_function(alpha, wfeZ4or5toZmax, wfeAboveZmax, usablePixelMap):
    """
    Compute the quadratic cost of the RMS phase difference for DMs 1 and 2.

    This function is used in evenly distributing the RMS phase between
    DMs 1 and 2. The weight alpha can vary between 0 and 1. The solver
    fminbound uses the cost function from this function to compute the best
    value of alpha.

    Parameters
    ----------
    alpha : float
        Weighting value between 0.0 and 1.0, inclusive. Determines how much of
        wfeZ4or5toZmax to allocate to DM1 on top of wfeAboveZmax.
        (1-alpha)*wfeZ4or5toZmax goes to DM2.
    wfeZ4or5toZmax : array_like
        2-D WFE map comprised of Zernike modes Z4 or Z5 to Zmax. Starts at Z4
        if FCM is not used or at Z5 if FCM is used.
    wfeAboveZmax : array_like
        2-D WFE map comprised of everything excluding Zernike terms Z1-Zmax
        from the original WFE map.
    usablePixelMap : array_like
        2-D boolean map of pixels to use in the WFE maps.

    Returns
    -------
    cost : float
        Quadratic cost of RMS phase difference for DMs 1 and 2.

    """
    check.real_scalar(alpha, 'alpha', ValueError)
    if alpha < 0 or alpha > 1:
        raise ValueError('alpha must be in range [0, 1].')
    check.twoD_array(wfeZ4or5toZmax, 'wfeZ4or5toZmax', ValueError)
    check.real_array(wfeZ4or5toZmax, 'wfeZ4or5toZmax', ValueError)
    check.twoD_array(wfeAboveZmax, 'wfeAboveZmax', ValueError)
    check.real_array(wfeAboveZmax, 'wfeAboveZmax', ValueError)
    check.twoD_array(usablePixelMap, 'usablePixelMap', ValueError)
    if not (usablePixelMap == usablePixelMap.astype(bool)).all():
        raise TypeError('usablePixelMap must be an array of booleans')
    if not wfeZ4or5toZmax.shape == usablePixelMap.shape:
        raise ValueError(('wfeZ4or5toZmax and usablePixelMap ' +
                          'must have same shape'))
    if not wfeAboveZmax.shape == usablePixelMap.shape:
        raise ValueError(('wfeAboveZmax and usablePixelMap ' +
                          'must have same shape'))

    cost = (rms(wfeAboveZmax[usablePixelMap] +
                alpha*wfeZ4or5toZmax[usablePixelMap]) -
            rms((1.-alpha)*wfeZ4or5toZmax[usablePixelMap]))**2

    return cost
