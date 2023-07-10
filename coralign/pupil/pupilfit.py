# pylint: disable=maybe-no-member
"""Fit lateral offsets, magnification, clocking, and Zernikes for a pupil."""
from astropy.io import fits
import numpy as np
import os
from scipy.ndimage import binary_erosion

from coralign.util.ampthresh import ampthresh
from coralign.util.loadyaml import loadyaml
from coralign.util.fit_shapes import fit_ellipse
from coralign.util.pad_crop import pad_crop
from coralign.util.math import ceil_even
from coralign.util.nollzernikes import fit_zernikes, gen_zernikes
import coralign.util.check as check
from coralign.util.debug import debug_plot
from coralign.maskgen.maskgen import rotate_shift_downsample_amplitude_mask
from coralign.pupil.util import (
    compute_norm_factor, compute_lateral_offsets, compute_clocking,
)

# some default values
LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
SELEM_DEFAULT = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0],
])


def fit_unmasked_pupil(pupil, fn_tuning, fn_pupil_ellipse_fitting,
                       data_path=LOCAL_PATH, debug=False):
    """
    Find diameter, clocking, and centering offsets of the unmasked pupil.

    All filenames may be absolute or relative paths.  If relative in input
    arguments, they will be relative to the current working directory, not to
    any particular location in Calibration.  If relative within YAML files
    (e.g. the fnMaskRefHighRes key in fn_tuning), this will be relative to the
    path in the data_path argument.

    Parameters
    ----------
    pupil : array_like
        2-D array containing the pupil amplitude or pupil intensity
    fn_tuning : str
        Name of the YAML file containing the tuning parameters and other
        data used for fitting the pupil.
    fn_pupil_ellipse_fitting : str
        Name of the YAML file containing the tuning parameters for fitting an
        ellipse to the pupil.
    data_path : str
        Directory to serve as a base for relative paths with YAML files.
        If not supplied, defaults to directory containing this function. If
        using versions of the YAMLs delivered with the repository, the default
        will point to data files also delivered with the repository.

    Returns
    -------
    xOffset, yOffset: float
        Estimate of pupil's x- and y-axis offsets from the image center pixel.
        Units of ExCAM pixels.
    clockEst : float
        Estimate of pupil's clocking with respect to the array containing it.
        Units of degrees.
    diamEst : float
        Estimate of the pupil's diameter. Units of ExCAM pixels.

    Notes
    -----
    Works well for the unmasked pupils. Does not work well for a shaped
    pupil probably because the number of edges is much higher, which can throw
    off how many pixels get rounded up or down by ampthresh().
    """
    check.twoD_array(pupil, 'pupil', TypeError)
    check.real_array(pupil, 'pupil', TypeError)
    check.string(fn_tuning, 'fn_tuning', TypeError)
    check.string(fn_pupil_ellipse_fitting, 'fn_pupil_ellipse_fitting',
                 TypeError)
    check.string(data_path, 'data_path', TypeError)

    inp = loadyaml(fn_tuning)
    nIterOffsets = inp['nIterOffsets']
    nPadFFT = inp['nPadFFT']
    nFocusCrop = inp['nFocusCrop']
    nPhaseSteps = inp['nPhaseSteps']
    dPixel = inp['dPixel']
    clockDegMax = inp['clockDegMax']
    nClock = inp['nClock']
    fnMaskRefHighRes = inp['fnMaskRefHighRes']
    diamHighRes = inp['diamHighRes']
    rotRef = inp['rotRef']
    flipxRef = inp['flipxRef']
    padFacRef = inp['padFacRef']
    deltaAmpMax = inp['deltaAmpMax']
    percentileForImageNorm = inp['percentileForImageNorm']
    zeroPaddingForHighResMask = inp['zeroPaddingForHighResMask']

    check.positive_scalar_integer(nIterOffsets, 'nIterOffsets', TypeError)
    check.positive_scalar_integer(nPadFFT, 'nPadFFT', TypeError)
    check.positive_scalar_integer(nFocusCrop, 'nFocusCrop', TypeError)
    check.positive_scalar_integer(nPhaseSteps, 'nPhaseSteps', TypeError)
    check.real_positive_scalar(dPixel, 'dPixel', TypeError)
    check.real_positive_scalar(clockDegMax, 'clockDegMax', TypeError)
    check.positive_scalar_integer(nClock, 'nClock', TypeError)
    if not isinstance(fnMaskRefHighRes, str):
        raise TypeError('Wrong type for fn_prefix_high_res_sim_mask')
    check.real_positive_scalar(diamHighRes, 'diamHighRes', TypeError)
    check.real_scalar(rotRef, 'rotRef', TypeError)
    check.boolean(flipxRef, 'flipxRef', TypeError)
    check.real_positive_scalar(padFacRef, 'padFacRef', TypeError)
    check.real_nonnegative_scalar(deltaAmpMax, 'deltaAmpMax', TypeError)
    check.real_positive_scalar(percentileForImageNorm,
                         'percentileForImageNorm', TypeError)
    check.real_nonnegative_scalar(zeroPaddingForHighResMask,
                            'zeroPaddingForHighResMask', TypeError)

    # Compute pupil diameter and coarse estimates of x- and y-offsets
    debug_plot(debug, 1, pupil, 'INPUT IMAGE')
    pupil = ampthresh(pupil).astype(float)
    diamEst, xOffsetCoarse, yOffsetCoarse = fit_ellipse(
        pupil, fn_pupil_ellipse_fitting)

    # Compute fine estimates of x- and y-offsets
    xOffsetRef = 0
    yOffsetRef = 0
    maskRefHighRes = fits.getdata(os.path.join(data_path, fnMaskRefHighRes))
    pupilRef = rotate_shift_downsample_amplitude_mask(
        maskRefHighRes, rotRef, diamEst/diamHighRes, xOffsetRef, yOffsetRef,
        padFac=padFacRef, flipx=flipxRef)
    pupilRef = pad_crop(pupilRef, (nPadFFT, nPadFFT))
    debug_plot(debug, 2, pupilRef, 'AFTER MASK ADDITION TO IMAGE')

    yOffset, xOffset = compute_lateral_offsets(
        pupil, pupilRef, diamEst, nPhaseSteps, dPixel, nPadFFT,
        nFocusCrop, nIter=nIterOffsets, useCoarse=False,
        xOffsetCoarse=xOffsetCoarse, yOffsetCoarse=yOffsetCoarse)

    # Compute clocking
    nPad = int(ceil_even(np.max(maskRefHighRes.shape) +
                         zeroPaddingForHighResMask))
    maskRefHighResPad = pad_crop(maskRefHighRes, (nPad, nPad))
    debug_plot(debug, 3, maskRefHighResPad, 'AFTER MASK PADDING ADDED')
    clockEst = compute_clocking(
        pupil, diamEst, maskRefHighResPad, diamHighRes, xOffset, yOffset,
        clockDegMax, nClock, percentileForImageNorm, deltaAmpMax)

    return xOffset, yOffset, clockEst, diamEst


def fit_unmasked_pupil_old(pupilAmp, fn_tuning):
    """
    Find diameter, clocking, and centering offsets of the unmasked pupil.

    Parameters
    ----------
    pupilAmp : array_like
        2-D array containing either the real- or complex-valued pupil amplitude
    fn_tuning : str
        Name of the YAML file containing the tuning parameters and other
        data used for fitting the pupil.

    Returns
    -------
    xOffset1, yOffset1: float
        Estimate of pupil's x- and y-axis offsets from the image center pixel.
        Units of ExCAM pixels.
    clockEst0 : float
        Estimate of pupil's clocking with respect to the array containing it.
        Units of degrees.
    diamEst0 : float
        Estimate of the pupil's diameter. Units of ExCAM pixels.

    Notes
    -----
    Works well for the unmasked WFIRST pupil. Does not work well for a shaped
    pupil probably because the number of edges is much higher, which can throw
    off how many pixels get rounded up or down by AMPTHRESH.
    """
    check.twoD_array(pupilAmp, 'pupilAmp', TypeError)

    inp = loadyaml(fn_tuning)
    nBeamNom = inp['nBeamNom']
    nBin = inp['nBin']
    nPadFFT = inp['nPadFFT']
    nFocusCrop = inp['nFocusCrop']
    nPhaseSteps = inp['nPhaseSteps']
    dPixel = inp['dPixel']
    clockDegMax = inp['clockDegMax']
    nClock = inp['nClock']
    fnMaskRefNomRes = inp['fnMaskRefNomRes']
    fnMaskRefHighRes = inp['fnMaskRefHighRes']
    diamHighRes = inp['diamHighRes']
    deltaAmpMax = inp['deltaAmpMax']
    percentileForImageNorm = inp['percentileForImageNorm']
    zeroPaddingForHighResMask = inp['zeroPaddingForHighResMask']

    if not isinstance(nBeamNom, (float, int)):
        raise TypeError('nBeamNom')
    if not isinstance(nBin, int):
        raise TypeError('nBin')
    if not isinstance(nPadFFT, int):
        raise TypeError('nPadFFT')
    if not isinstance(nFocusCrop, int):
        raise TypeError('nFocusCrop')
    if not isinstance(nPhaseSteps, int):
        raise TypeError('nPhaseSteps')
    if not isinstance(dPixel, (float, int)):
        raise TypeError('dPixel')
    if not isinstance(clockDegMax, (float, int)):
        raise TypeError('clockDegMax')
    if not isinstance(nClock, int):
        raise TypeError('nClock')
    if not isinstance(fnMaskRefNomRes, str):
        raise TypeError('Wrong type for fn_prefix_low_res_sim_mask')
    if not isinstance(fnMaskRefHighRes, str):
        raise TypeError('Wrong type for fn_prefix_high_res_sim_mask')
    if not isinstance(diamHighRes, (float, int)):
        raise TypeError('diamHighRes')
    check.real_nonnegative_scalar(deltaAmpMax, 'deltaAmpMax', TypeError)
    check.real_positive_scalar(percentileForImageNorm,
                          'percentileForImageNorm', TypeError)
    check.real_nonnegative_scalar(zeroPaddingForHighResMask,
                            'zeroPaddingForHighResMask',
                            TypeError)

    localpath = os.path.dirname(os.path.abspath(__file__))
    maskRefNomRes = fits.getdata(os.path.join(localpath, fnMaskRefNomRes))
    maskRefHighRes = fits.getdata(os.path.join(localpath, fnMaskRefHighRes))

    ###########################################################################
    # Run AMPTHRESH
    ampBool = ampthresh(pupilAmp, nBin=nBin)
    amp = ampBool.astype(float)

    ###########################################################################
    # Get pupil diameter estimate

    pupilMeas = np.abs(pupilAmp)
    pupilRef = np.abs(maskRefNomRes)

    normFacMeas = compute_norm_factor(pupilMeas, ampthresh(pupilMeas),
                                      percentileForImageNorm)
    normFacRef = compute_norm_factor(pupilRef, ampthresh(pupilRef),
                                      percentileForImageNorm)
    pupilMeasNorm = pupilMeas/normFacMeas
    pupilRefNorm = pupilRef/normFacRef

    diamEst0 = (nBeamNom
                * np.sqrt(np.sum(pupilMeasNorm[ampthresh(pupilMeasNorm)])
                          / np.sum(pupilRefNorm[ampthresh(pupilRefNorm)])))

    ###########################################################################
    # Get estimates of x- and y-offsets
    xOffset0 = 0
    yOffset0 = 0
    clockEst0 = 0
    pupilRef = rotate_shift_downsample_amplitude_mask(
        maskRefHighRes, clockEst0, diamEst0/diamHighRes, xOffset0, yOffset0,
    )
    pupilRef = pad_crop(pupilRef, (nPadFFT, nPadFFT))

    yOffset1, xOffset1 = compute_lateral_offsets(amp, pupilRef, diamEst0,
                                                  nPhaseSteps, dPixel,
                                                  nPadFFT, nFocusCrop)

    ###########################################################################
    # CLOCKING

    nPad = int(ceil_even(np.max(maskRefHighRes.shape) +
                          zeroPaddingForHighResMask))

    maskRefHighResPad = pad_crop(maskRefHighRes, (nPad, nPad))
    clockEst1 = compute_clocking(pupilMeas, diamEst0, maskRefHighResPad,
                                  diamHighRes, xOffset1, yOffset1, clockDegMax,
                                  nClock, percentileForImageNorm, deltaAmpMax)

    return xOffset1, yOffset1, clockEst1, diamEst0


def fit_pupil_zernikes(wfe, amp, fn_pupil_fit_params, fn_ellipse_fit_params,
                  bMask=None, Z_noll_max=11, mask_selem=SELEM_DEFAULT,
                  data_path=LOCAL_PATH):
    """
    Perform Zernike analysis of pupil wavefront.

    Given a wavefront phase and either its amplitude array or a binary pupil
    mask:
    1. compute pupil diameter and center from wavefront amplitude
    2. create a pupil binary mask if necessary
    3. calculate Zernike coefficients
    4. calculate residual wavefront phase (=wavefront - zernike fit)

    All filenames may be absolute or relative paths.  If relative in input
    arguments, they will be relative to the current working directory, not to
    any particular location in Calibration.  If relative within YAML files
    (e.g. the fnMaskRefHighRes key in fn_tuning), this will be relative to the
    path in the data_path argument.

    Parameters
    ----------
    wfe : 2-d numpy array, wavefront phase, units are arbitrary, but typically
     radians or nm

    amp : wavefront amplitude, numpy array same size as wfe,
        used to determine pupil center and normalization radius for zernike
        fitting. Used to create pupil mask if bMask is None

    fn_pupil_fit_params : str
        Passed directly to pupilfit_open.fit_unmasked_pupil().
        Name of the YAML file containing the tuning parameters and other
        data used for fitting the pupil.

    fn_ellipse_fit_params : str
        Passed directly to pupilfit_open.fit_unmasked_pupil().
        Name of the YAML file containing the tuning parameters for fitting an
        ellipse to the pupil.

    bMask : (optional) binary numpy array, same size as wfe
        pupil mask, only wfe pixels where bMask == 1 are used in the Zernike
        fit.  If bMask is not input, a pupil is created from amp

    Z_noll_max : (default = 11) max Noll Zernike coefficient to calculate,
        min = 3

    mask_selem : (default = SELEM_DEFAULT)
        if the binary pupil mask is calculated from amp, mask_selem is the
        kernel used to erode the edges of the mask. The default kernel erodes
        the mask edges by one pixel.

        If optional bMask is given, then edge erosion with mask_selem is NOT
        done.  To skip the erosion step, set mask_selem = None
    data_path : str
        Directory to serve as a base for relative paths with YAML files.
        If not supplied, defaults to directory containing this function. If
        using versions of the YAMLs delivered with the repository, the default
        will point to data files also delivered with the repository.

    Returns
    -------
    Zvec : numpy ndarray
        1-D array of Zernike coefficients. Same units as the wfe map

    dictResults : dict
        'bMask': binary mask used for Zernike fit, so that it can be reused
         for next
        'wfe_res_ptt': 2-d array size(wfe) = wfe - zernike fit piston, tip,
         tilt
        'wfe_res' : 2-d array size(wfe) = wfe - zernike fit Zvec
        'diamEst' : diameter (pixels) used for Zernike normalization radius
        'xOffset' : float, x-offset from image center pixel to pupil center
            Units of EXCAM pixels.
        'yOffset' : float, y-offset from image center pixel to pupil center.
            Units of EXCAM pixels.
    """
    # Check inputs
    check.twoD_array(wfe, 'wfe', TypeError)
    check.real_array(wfe, 'wfe', TypeError)
    check.twoD_array(amp, 'amp', TypeError)
    check.real_array(amp, 'amp', TypeError)
    if bMask is not None:
        check.twoD_array(bMask, 'bMask', TypeError)
        check.real_array(bMask, 'bMask', TypeError)
        if np.logical_and(bMask != 0, bMask != 1).any():
            raise TypeError('bMask can only contain booleans')
    check.positive_scalar_integer(Z_noll_max, 'Z_noll_max', TypeError)
    if Z_noll_max < 3:
        print('warning: Z_noll_max < 3, setting Z_noll_max = 3')
        Z_noll_max = 3
    if mask_selem is not None:
        check.twoD_array(mask_selem, 'mask_selem', TypeError)
        check.real_array(mask_selem, 'mask_selem', TypeError)
    check.string(fn_pupil_fit_params, 'fn_pupil_fit_params', TypeError)
    check.string(fn_ellipse_fit_params, 'fn_ellipse_fit_params', TypeError)
    check.string(data_path, 'data_path', TypeError)

    # 1. compute pupil diameter
    xOffset, yOffset, _, diamEst = fit_unmasked_pupil(
        amp, fn_pupil_fit_params, fn_ellipse_fit_params, data_path=data_path,
    )

    # 2. make a pupil mask (check against mask used for unwrapping = bMask
    # above?)
    if bMask is None:
        # create bMask from amp
        bMask = ampthresh(amp)
        if mask_selem is not None:
            bMask = binary_erosion(bMask, structure=mask_selem)

    # 3. zernike decomposition, return Zvec same units as wfe (radians)
    Zvec = fit_zernikes(wfe, bMask, Z_noll_max, diamEst, xOffset, yOffset)

    # 4. generate residual maps for analysis and convenience
    wfe_res_ptt = wfe - bMask*gen_zernikes(
        np.array([1, 2, 3]), Zvec[:3], xOffset, yOffset, diamEst,
        nArray=wfe.shape[0],
    )
    wfe_res = wfe - bMask*gen_zernikes(
        np.arange(1, 1+Z_noll_max), Zvec, xOffset, yOffset, diamEst,
        nArray=wfe.shape[0],
    )

    # return values
    return Zvec, {
        # binary mask used for Zernike fit, so that it can be reused for next
        'bMask': bMask,
        # residual = wfe - zernike fit piston,tip,tilt
        'wfe_res_ptt': wfe_res_ptt,
        # residual = wfe - zernike fit Zvec
        'wfe_res': wfe_res,
        # diameter (pixels) used for Zernike normalization radius
        'diamEst': diamEst,
        # x-offset from image center pixel to pupil center (EXCAM pixels)
        'xOffset': xOffset,
        # y-offset from image center pixel to pupil center (EXCAM pixels)
        'yOffset': yOffset,
    }
