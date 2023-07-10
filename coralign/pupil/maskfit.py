# pylint: disable=maybe-no-member
"""Fit lateral offsets, magnification, and/or clocking of pupil masks."""
from astropy.io import fits
import numpy as np
import os
import scipy.ndimage
import matplotlib.pyplot as plt

from coralign.util import check
from coralign.util.ampthresh import ampthresh
from coralign.util.loadyaml import loadyaml
# from coralign.util.fit_shapes import fit_ellipse
from coralign.util.pad_crop import pad_crop, offcenter_crop
# from coralign.util.math import ceil_even

from coralign.util.debug import debug_plot
from coralign.dmreg.dr_util import remove_piston_tip_tilt
from coralign.maskgen.maskgen import rotate_shift_downsample_amplitude_mask
from coralign.pupil.util import (
    coarsely_locate_pupil_offset, compute_norm_factor,
    compute_lateral_offsets, compute_clocking,
)
from coralign.psffit.psffit import psffit
import coralign.util.unwrap as pru


LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))


def fit_shaped_pupil_mask_offsets(
        imageUnmasked, imageMasked, offsetGain, fnMaskParams, fnTuningParams,
        xOffsetPupil, yOffsetPupil, diamPupil, magAtMask, clockDegMask,
        xOffsetSPM, yOffsetSPM, data_path=LOCAL_PATH, debug=False):
    """
    Compute the lateral shaped pupil offsets compared to an unmasked image.

    This function is a wrapper for fit_pupil_mask_offsets() with one step
    performed beforehand, which is shifting the unmasked image. The shaped
    pupil mask (SPM) lateral offset is more difficult to estimate because the
    reflective SPM can be in a different plane than the fold mirror used
    during initial calibrations. This can cause a lateral shear of the
    underlying pupil image when the SPM is moved into the beam. This function
    shifts the underlying, unmasked pupil image by the estimated z-induced
    beam shear (separately found with fit_z_induced_spm_shear()) before
    calling fit_pupil_mask_offsets().

    All filenames may be absolute or relative paths.  If relative in input
    arguments, they will be relative to the current working directory, not to
    any particular location in Calibration.  If relative within YAML files
    (e.g. the fnMaskRefHighRes key in fnMaskParams), this will be relative to
    the path in the data_path argument.

    Parameters
    ----------
    imageUnmasked : array_like
        Unmasked intensity image of the pupil or mask to be masked.
    imageMasked : array_like
        Intensity image of the mask over the pupil or other mask. Must have
        same array shape and centering as imageUnmasked.
    offsetGain : float
        The gain used when iterating to compute the offset value. Leave as 1
        for most cases. Should be >1 for the SPC-WFOV Lyot stop to SPM case
        because the underlying shaped pupil hides the outer edge of the Lyot
         stop and makes each offset estimate a few times smaller than what
         it actually is.
    fnMaskParams : str
        Name of YAML file containing info about the high-resolution reference
        mask to use in the fitting.
    fnTuningParams : str
        Name of YAML file containing fitting parameter values.
    xOffsetPupil, yOffsetPupil : float
        Estimated lateral offsets of the underlying, nominal telescope pupil.
        Units of pixels.
    diamPupil : float
        Estimated diameter of the nominal, unmasked telescope pupil as
        seen at the detector. Units of pixels.
    magAtMask : float
        Estimated magnification (compared to the nominal telescope pupil) at
        the plane of the mask being aligned. Usually is close to 1.
    clockDegMask : float
        Estimated clocking of the Mask being aligned. Units of degrees.
    xOffsetSPM, yOffsetSPM : float
        Estimated lateral pupil beam shear that occurs when switching from
        the unmasked pupil to the shaped pupil mask. Units of pixels.
    data_path : str
        Directory to serve as a base for relative paths with YAML files.
        If not supplied, defaults to directory containing this function. If
        using versions of the YAMLs delivered with the repository, the default
        will point to data files also delivered with the repository.

    Returns
    -------
    xOffsetEst, yOffsetEst : float
        Lateral offset estimate of the mask compared to the underlying pupil
        or mask. Units of ExCAM pixels.
    """
    check.twoD_array(imageUnmasked, 'imageUnmasked', TypeError)
    check.twoD_array(imageMasked, 'imageMasked', TypeError)
    sh0 = imageUnmasked.shape
    sh1 = imageMasked.shape
    if (sh0[0] != sh1[0]) or (sh0[1] != sh1[1]):
        raise ValueError('imageUnmasked and imageMasked must have same shape')
    check.real_positive_scalar(offsetGain, 'offsetGain', TypeError)
    check.string(fnMaskParams, 'fnMaskParams', TypeError)
    check.string(fnTuningParams, 'fnTuningParams', TypeError)
    check.real_scalar(yOffsetPupil, 'yOffsetPupil', TypeError)
    check.real_scalar(xOffsetPupil, 'xOffsetPupil', TypeError)
    check.real_positive_scalar(diamPupil, 'diamPupil', TypeError)
    check.real_positive_scalar(magAtMask, 'magAtMask', TypeError)
    check.real_scalar(clockDegMask, 'clockDegMask', TypeError)
    check.real_scalar(xOffsetSPM, 'xOffsetSPM', TypeError)
    check.real_scalar(yOffsetSPM, 'yOffsetSPM', TypeError)
    check.string(data_path, 'data_path', TypeError)

    debug_plot(debug, 1, imageUnmasked, 'INPUT IMAGE (UNMASKED)')
    debug_plot(debug, 2, imageMasked, 'INPUT IMAGE (MASKED)')

    imageUnmasked = scipy.ndimage.shift(imageUnmasked,
                                        [yOffsetSPM, xOffsetSPM])

    debug_plot(debug, 3, imageUnmasked, 'IMAGE AFTER SHIFT (UNMASKED)')

    # New beam center location with SPM in place
    xOffsetTotal = xOffsetPupil + xOffsetSPM
    yOffsetTotal = yOffsetPupil + yOffsetSPM

    xOffsetEst, yOffsetEst = fit_pupil_mask_offsets(
        imageUnmasked, imageMasked, offsetGain, fnMaskParams, fnTuningParams,
        xOffsetTotal, yOffsetTotal, diamPupil, magAtMask, clockDegMask,
        data_path=data_path,
    )

    return xOffsetEst, yOffsetEst


def fit_pupil_mask_offsets(
        imageUnmasked, imageMasked, offsetGain, fnMaskParams, fnTuningParams,
        xOffsetPupil, yOffsetPupil, diamPupil, magAtMask, clockDegMask,
        data_path=LOCAL_PATH, debug=False):
    """
    Compute the lateral mask offsets compared to an unmasked image.

    Parameters
    ----------
    imageUnmasked : array_like
        Unmasked intensity image of the pupil or mask to be masked.
    imageMasked : array_like
        Intensity image of the mask over the pupil or other mask. Must have
        same array shape and centering as imageUnmasked.
    offsetGain : float
        The gain used when iterating to compute the offset value. Leave as 1
        for most cases. Should be >1 for the SPC-WFOV Lyot stop to SPM case
        because the underlying shaped pupil hides the outer edge of the Lyot
         stop and makes each offset estimate a few times smaller than what
         it actually is.
    fnMaskParams : str
        Name of YAML file containing info about the high-resolution reference
        mask to use in the fitting.
    fnTuningParams : str
        Name of YAML file containing fitting parameter values.
    xOffsetPupil, yOffsetPupil : float
        Estimated lateral offsets of the underlying, nominal telescope pupil.
        Units of pixels.
    diamPupil : float
        Estimated diameter of the nominal, unmasked telescope pupil as
        seen at the detector. Units of pixels.
    magAtMask : float
        Estimated magnification (compared to the nominal telescope pupil) at
        the plane of the mask being aligned. Usually is close to 1.
    clockDegMask : float
        Estimated clocking of the Mask being aligned. Units of degrees.
    data_path : str
        Directory to serve as a base for relative paths with YAML files.
        If not supplied, defaults to directory containing this function. If
        using versions of the YAMLs delivered with the repository, the default
        will point to data files also delivered with the repository.

    Returns
    -------
    xOffsetEst, yOffsetEst : float
        Lateral offset estimate of the mask compared to the underlying pupil
        or mask. Units of ExCAM pixels.
    """
    check.twoD_array(imageUnmasked, 'imageUnmasked', TypeError)
    check.twoD_array(imageMasked, 'imageMasked', TypeError)
    sh0 = imageUnmasked.shape
    sh1 = imageMasked.shape
    if (sh0[0] != sh1[0]) or (sh0[1] != sh1[1]):
        raise ValueError('imageUnmasked and imageMasked must have same shape')
    check.real_positive_scalar(offsetGain, 'offsetGain', TypeError)
    check.string(fnMaskParams, 'fnMaskParams', TypeError)
    check.string(fnTuningParams, 'fnTuningParams', TypeError)
    check.real_scalar(yOffsetPupil, 'yOffsetPupil', TypeError)
    check.real_scalar(xOffsetPupil, 'xOffsetPupil', TypeError)
    check.real_positive_scalar(diamPupil, 'diamPupil', TypeError)
    check.real_positive_scalar(magAtMask, 'magAtMask', TypeError)
    check.real_scalar(clockDegMask, 'clockDegMask', TypeError)
    check.string(data_path, 'data_path', TypeError)

    imageShape = np.shape(imageUnmasked)

    debug_plot(debug, 1, imageUnmasked, 'INPUT IMAGE (UNMASKED)')
    debug_plot(debug, 2, imageMasked, 'INPUT IMAGE (MASKED)')

    # Load tuning parameters from YAML file
    inp = loadyaml(fnTuningParams)
    nPadFFT = inp['nPadFFT']
    nFocusCrop = inp['nFocusCrop']
    nPhaseSteps = inp['nPhaseSteps']
    dPixel = inp['dPixel']
    nIterOuterLoop = inp['nIterOuterLoop']
    nIterInnerLoop = inp['nIterInnerLoop']
    percentileForImageNorm = inp['percentileForImageNorm']

    # Load size and name of the high-res mask template from a YAML file
    inp = loadyaml(fnMaskParams)
    diamHighResMaskRef = inp['diamHighResMaskRef']
    fnMaskRefHighRes = inp['fnMaskRefHighRes']
    maskRefHighRes = fits.getdata(os.path.join(data_path, fnMaskRefHighRes))

    # Normalize images and convert to amplitude
    normFactorUnmasked = compute_norm_factor(imageUnmasked,
                                             ampthresh(imageUnmasked),
                                             percentileForImageNorm)
    imageUnmasked = imageUnmasked/normFactorUnmasked
    imageUnmasked[imageUnmasked < 0] = 0
    ampUnmasked = np.sqrt(imageUnmasked)

    normFactorMasked = compute_norm_factor(imageMasked,
                                           ampthresh(imageMasked),
                                           percentileForImageNorm)
    imageMasked = imageMasked/normFactorMasked
    imageMasked[imageMasked < 0] = 0
    ampMasked = np.sqrt(imageMasked)

    # Not used because useCoarse=True, but still need a value assigned to the
    # input argument for compute_lateral_offsets()
    diamMask = diamPupil

    xOffsetEst = 0
    yOffsetEst = 0
    for iter_ in range(nIterOuterLoop):

        maskRef = rotate_shift_downsample_amplitude_mask(
            maskRefHighRes, clockDegMask,
            diamPupil*magAtMask/diamHighResMaskRef, xOffsetEst+xOffsetPupil,
            yOffsetEst+yOffsetPupil, padFac=1.2, flipx=False)
        maskRef = pad_crop(maskRef, imageShape)
        arraySim = ampUnmasked*maskRef

        # Crop to be centered on the pupil
        xOffsetPupilRound = int(np.round(xOffsetPupil))
        yOffsetPupilRound = int(np.round(yOffsetPupil))
        xCenterPupil = imageShape[1]//2 + xOffsetPupilRound
        yCenterPupil = imageShape[0]//2 + yOffsetPupilRound
        ampMaskedRecenter = offcenter_crop(ampMasked, nPadFFT,
                                           yCenterPupil, xCenterPupil)
        arraySimRecenter = offcenter_crop(arraySim, nPadFFT,
                                          yCenterPupil, xCenterPupil)

        debug_plot(debug, 3, ampMaskedRecenter, 'ampMaskedRecenter')
        debug_plot(debug, 4, arraySimRecenter, 'arraySimRecenter')

        yOffsetDelta, xOffsetDelta = compute_lateral_offsets(
            ampMaskedRecenter, arraySimRecenter, diamMask, nPhaseSteps, dPixel,
            nPadFFT, nFocusCrop, nIter=nIterInnerLoop, useCoarse=False,
        )
        xOffsetEst += offsetGain*xOffsetDelta
        yOffsetEst += offsetGain*yOffsetDelta

    return xOffsetEst, yOffsetEst


def fit_z_induced_spm_shear(ampUnpoked0, phUnpoked0, phPoked0,
                            ampUnpoked1, phUnpoked1, phPoked1,
                            fnConfig, debug=False):
    """
    Compute the z-induced lateral shear of the SPM relative to the pupil.

    Because the shaped pupil masks (SPMs) are reflective, they can lie in a
    different plane than the regular fold mirror used for HLC and DM
    registration. This can cause beam shear, which complicates the task of
    aligning an SPM to the moved underlying pupil. This function uses a poked
    phase pattern before and after moving the SPM (roughly) into place to
    determine how much the underlying pupil sheared. This estimated pupil shear
    needs to be included in both the assumed beam position and the DM
    registration for both DMs. Because the SPAM is after the DMs, the relative
    alignment between DM1 and DM2 is unchanged by the SPMs.

    The poked phase pattern on the DM should be performed in an area that is
    (minimally) unonbscured for both the pupil and SPM. A small "+" shape can
    be used.

    Note: If the SPMs are not coplanar with the xy alignment stage holding
    them, then the beam shear can be different for each SPM. In that case,
    this calibration should be performed separately for each SPM.

    Parameters
    ----------
    ampUnpoked0 : array_like
        amplitude part of the pupil phase retrieval without the poke pattern
        on the deformable mirror.
    phUnpoked0 : array_like
        phase part of the pupil phase retrieval without the poke pattern
        on the deformable mirror.
    phPoked0 : array_like
        phase part of the pupil phase retrieval with the poke pattern
        on the deformable mirror.
    ampUnpoked1 : array_like
        amplitude part of the SPM phase retrieval without the poke pattern
        on the deformable mirror.
    phUnpoked1 : array_like
        phase part of the SPM phase retrieval without the poke pattern
        on the deformable mirror.
    phPoked1 : array_like
        phase part of the SPM phase retrieval with the poke pattern
        on the deformable mirror.
    fnConfig : str
        Name of file with tuning parameters for PSFFIT.

    Returns
    -------
    xOffset : float
        x shear of the beam in the second set of E-fields compared to the
        first set. Units of pixels.
    yOffset : float
        y shear of the beam in the second set of E-fields compared to the
        first set. Units of pixels.
    """
    # Type checks
    check.twoD_array(ampUnpoked0, 'ampUnpoked0', TypeError)
    check.twoD_array(phUnpoked0, 'phUnpoked0', TypeError)
    check.twoD_array(phPoked0, 'phPoked0', TypeError)
    check.twoD_array(ampUnpoked1, 'ampUnpoked0', TypeError)
    check.twoD_array(phUnpoked1, 'phUnpoked1', TypeError)
    check.twoD_array(phPoked1, 'phPoked1', TypeError)
    check.string(fnConfig, 'fnConfig', TypeError)

    # Array size checks
    shape0 = ampUnpoked0.shape
    if phUnpoked0.shape != shape0:
        raise ValueError('ampUnpoked0 and phUnpoked0 must have same shape')
    if phPoked0.shape != shape0:
        raise ValueError('ampUnpoked0 and phPoked0 must have same shape')
    shape1 = ampUnpoked1.shape
    if phUnpoked1.shape != shape1:
        raise ValueError('ampUnpoked1 and phUnpoked1 must have same shape')
    if phPoked1.shape != shape0:
        raise ValueError('ampUnpoked1 and phPoked1 must have same shape')

    debug_plot(debug, 1, ampUnpoked0, 'ampUnpoked0')
    debug_plot(debug, 2, phUnpoked0, 'phUnpoked0')
    debug_plot(debug, 3, phPoked0, 'phPoked0')
    debug_plot(debug, 4, ampUnpoked0, 'ampUnpoked0')
    debug_plot(debug, 5, phUnpoked1, 'phUnpoked1')
    debug_plot(debug, 6, phPoked1, 'phPoked1')

    phDiff0 = phPoked0 - phUnpoked0
    phDiff0, swMask0 = pru.unwrap(phDiff0, ampUnpoked0)
    phDiff0 = remove_piston_tip_tilt(phDiff0, swMask0)

    phDiff1 = phPoked1 - phUnpoked1
    phDiff1, swMask1 = pru.unwrap(phDiff1, ampUnpoked1)
    phDiff1 = remove_piston_tip_tilt(phDiff1, swMask1)
    phDiff1 = pad_crop(phDiff1, shape0)
    swMask1 = pad_crop(swMask1, shape0)

    shift_computed, _ = psffit(swMask0*phDiff0, swMask1*phDiff1, fnConfig)
    xOffset = shift_computed[1]
    yOffset = shift_computed[0]

    return xOffset, yOffset


def fit_lyot_stop_mag_clocking(
        imageUnmasked, imageMasked, xOffsetPupil, yOffsetPupil, fnMaskParams,
        fnOffsetParams, fnLyotCalib, data_path=LOCAL_PATH, debug=False):
    """
    Compute the magnification and clocking of the SPC-Spec Lyot stop mask.

    The clocking of the other Lyot stops will be determined by this function's
    output combined with relative clocking measurements of all the Lyot stops
    after they are bonded onto the LSAM plate. Only the SPC-Spec Lyot stop is
    used by this function because the other required Lyot stops have struts
    that cause confusion when comparing clocking against the unmasked pupil,
    which also has struts.

    All filenames may be absolute or relative paths.  If relative in input
    arguments, they will be relative to the current working directory, not to
    any particular location in Calibration.  If relative within YAML files
    (e.g. the fnMaskRefHighRes key in fnMaskParams), this will be relative to
    the path in the data_path argument.

    Parameters
    ----------
    imageUnmasked : array_like
        Unmasked intensity image of the unmasked pupil.
    imageMasked : array_like
        Intensity image of the Lyot stop applied to the pupil (no SPM in
        place). Must have same array shape and centering as imageUnmasked.
    xOffsetPupil, yOffsetPupil : float
        Estimated lateral offsets of the underlying, nominal telescope pupil.
        Units of ExCAM pixels.
    fnMaskParams : str
        Name of YAML file containing the Lyot stop parameters needed for
        this fitting routine.
    fnOffsetParams : str
        Name of YAML file containing offset fitting parameters.
    fnLyotCalib : str
        Name of YAML file containing Lyot stop calibration parameters.
    data_path : str
        Directory to serve as a base for relative paths with YAML files.
        If not supplied, defaults to directory containing this function. If
        using versions of the YAMLs delivered with the repository, the default
        will point to data files also delivered with the repository.

    Returns
    -------
    magEst : float
        Estimated magnification (compared to the nominal telescope pupil) at
        the LSAM.
    clockEst : float
        Estimated clocking compared to the reference Lyot stop file.
        Units of degrees.

    Notes
    -----
    Variables loaded from the YAML file of tuning values are:
    clockMaxDeg : float
        Maximum +/- clocking values to try. Units of degrees. Somewhat larger
        values than the expected range can be useful for getting a good
        quadratic fit.
    nClock : int
        Number of uniformly-spaced clocking values to try.
    deltaAmpMax : float
        Maximum allowed amplitude difference allowed between a measured mask
        and a reference mask for that pixel to be counted as matching in value.
    diamPupil : float
        Estimated diameter of the nominal telescope pupil.
        Units of ExCAM pixels.
    magShrinkFac : float
        Factor by which to shrink deltaMag each loop iteration.
        Must be > 0 and < 1.
    deltaMag : float
        +/- amount of delta magnification to use in a line search of
        magnification values. Must be > 0.
    nMag : int
        Number of magnification values to use in the line search.
    nIterMag : int
        Number of iterations over which to refine the magnification estimate.
    """
    check.twoD_array(imageUnmasked, 'imageUnmasked', TypeError)
    check.twoD_array(imageMasked, 'imageMasked', TypeError)
    sh0 = imageUnmasked.shape
    sh1 = imageMasked.shape
    if (sh0[0] != sh1[0]) or (sh0[1] != sh1[1]):
        raise ValueError('imageUnmasked and imageMasked must have same shape')
    check.real_scalar(xOffsetPupil, 'xOffsetPupil', TypeError)
    check.real_scalar(yOffsetPupil, 'yOffsetPupil', TypeError)

    inp = loadyaml(fnLyotCalib)
    clockMaxDeg = inp['clockMaxDeg']
    nClock = inp['nClock']
    deltaAmpMax = inp['deltaAmpMax']
    diamPupil = inp['diamPupil']
    magShrinkFac = inp['magShrinkFac']
    deltaMag = inp['deltaMag']
    nMag = inp['nMag']
    nIterMag = inp['nIterMag']
    check.real_positive_scalar(clockMaxDeg, 'clockMaxDeg', TypeError)
    check.positive_scalar_integer(nClock, 'nClock', TypeError)
    check.real_positive_scalar(deltaAmpMax, 'deltaAmpMax', TypeError)
    check.real_positive_scalar(diamPupil, 'diamPupil', TypeError)
    if deltaAmpMax >= 1:
        raise ValueError('deltaAmpMax must be less than 1')
    check.real_positive_scalar(magShrinkFac, 'magShrinkFac', TypeError)
    if magShrinkFac >= 1:
        raise ValueError('magShrinkFac must be less than 1')
    check.real_positive_scalar(deltaMag, 'deltaMag', TypeError)
    check.positive_scalar_integer(nMag, 'nMag', TypeError)
    check.positive_scalar_integer(nIterMag, 'nIterMag', TypeError)
    check.string(data_path, 'data_path', TypeError)

    inp = loadyaml(fnMaskParams)
    diamHighResMaskRef = inp['diamHighResMaskRef']
    fnMaskRefHighRes = inp['fnMaskRefHighRes']
    OD_LS = inp['OD_LS']

    inp = loadyaml(fnOffsetParams)
    nPadFFT = inp['nPadFFT']
    nFocusCrop = inp['nFocusCrop']
    nPhaseSteps = inp['nPhaseSteps']
    dPixel = inp['dPixel']
    percentileForImageNorm = inp['percentileForImageNorm']

    maskRefHighRes = fits.getdata(os.path.join(data_path, fnMaskRefHighRes))

    debug_plot(debug, 1, imageUnmasked, 'INPUT IMAGE (UNMASKED)')
    debug_plot(debug, 2, imageMasked, 'INPUT IMAGE (MASKED)')

    # Normalize and square root images
    # normFacMasked = compute_norm_factor(imageMasked, ampthresh(imageMasked),
    #                                     percentileForImageNorm)
    # imageMasked = imageMasked/normFacMasked
    # imageMasked[imageMasked < 0] = 0
    # ampMasked = np.sqrt(imageMasked)

    normFacUnmasked = compute_norm_factor(imageUnmasked,
                                          ampthresh(imageUnmasked),
                                          percentileForImageNorm)
    imageUnmasked = imageUnmasked/normFacUnmasked
    imageUnmasked[imageUnmasked < 0] = 0
    ampUnmasked = np.sqrt(imageUnmasked)

    normFacMasked = normFacUnmasked
    imageMasked = imageMasked/normFacMasked
    imageMasked[imageMasked < 0] = 0
    ampMasked = np.sqrt(imageMasked)

    # INITIAL MAGNIFICATION ESTIMATE
    xOffset0 = 0
    yOffset0 = 0
    clockDegEst = 0
    imageShape = np.shape(imageUnmasked)
    maskLyotRef0 = rotate_shift_downsample_amplitude_mask(
        maskRefHighRes, clockDegEst, diamPupil/diamHighResMaskRef,
        xOffset0+xOffsetPupil, yOffset0+yOffsetPupil,
    )
    maskLyotRef0 = pad_crop(maskLyotRef0, imageShape)

    # Use ampthresh on the image instead of the amplitude because ampthresh
    # doesn't work well when on square-rooted noisy images with values less
    # than zero zeroed out. Otherwise, lots of pixels outside the pupil
    # get counted as well.
    magEst0 = np.sqrt(np.sum(ampMasked[ampthresh(imageMasked)])
                      / np.sum((ampUnmasked*maskLyotRef0)[ampthresh
                                                          (imageUnmasked)]))

    # LATERAL TRANSLATION
    maskLyotRef1 = rotate_shift_downsample_amplitude_mask(
        maskRefHighRes, clockDegEst, magEst0*diamPupil/diamHighResMaskRef,
        xOffset0+xOffsetPupil, yOffset0+yOffsetPupil,
    )
    maskLyotRef1 = pad_crop(maskLyotRef1, imageShape)
    ampRef1 = maskLyotRef1*ampUnmasked
    if (debug is True):
        plt.figure(8)
        plt.imshow(maskLyotRef0)
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.title('AFTER MASK ADDITION TO IMAGE')
        plt.pause(0.1)

    diamBeamEst = magEst0*diamPupil
    diamMaskEst = magEst0*diamPupil*OD_LS
    xOffset1, yOffset1 = coarsely_locate_pupil_offset(ampMasked, diamMaskEst)

    ampMaskedCrop = offcenter_crop(ampMasked, int(np.max(imageShape)),
                                   yOffset1+imageShape[0]//2,
                                   xOffset1+imageShape[1]//2)
    ampRef1Crop = offcenter_crop(ampRef1, int(np.max(imageShape)),
                                 yOffset1+imageShape[0]//2,
                                 xOffset1+imageShape[1]//2)
    yOffset2, xOffset2 = compute_lateral_offsets(ampMaskedCrop, ampRef1Crop,
                                                 diamBeamEst, nPhaseSteps,
                                                 dPixel, nPadFFT,
                                                 nFocusCrop, useCoarse=False,
                                                 nIter=2)

    debug_plot(debug, 3, ampMaskedCrop, 'ampMaskedCrop')
    debug_plot(debug, 4, ampRef1Crop, 'ampRef1Crop')

    # CLOCKING
    clockEst = compute_clocking(ampMasked, diamBeamEst, maskRefHighRes,
                                diamHighResMaskRef, xOffset2+xOffsetPupil,
                                yOffset2+yOffsetPupil, clockMaxDeg, nClock,
                                percentileForImageNorm, deltaAmpMax)

    # MAGNIFICATION UPDATE: LINE SEARCH
    magEst = magEst0

    for iter_ in range(nIterMag):
        magVec = magEst + magShrinkFac*np.linspace(-deltaMag, deltaMag, nMag)
        overlapVec = np.zeros((nMag,))

        for im, mag in enumerate(magVec):

            pupilRef = rotate_shift_downsample_amplitude_mask(
                maskRefHighRes, clockEst, mag*diamPupil/diamHighResMaskRef,
                xOffset2+xOffsetPupil, yOffset2+yOffsetPupil,
            )
            pupilRef = pad_crop(pupilRef, ampMasked.shape)
            pupilRef = pupilRef**2

            normFacRef = compute_norm_factor(pupilRef, ampthresh(pupilRef),
                                             percentileForImageNorm)
            pupilRefNorm = pupilRef/normFacRef
            compMatrix = np.abs(ampMasked - pupilRefNorm) < deltaAmpMax
            overlapVec[im] = np.sum(compMatrix)

        # Find the maximum of the overlap
        if iter_ == 0 and nIterMag > 1:
            bestInd = np.argmax(overlapVec)
            magEst = magVec[bestInd]
        else:
            # Use quadratic fit to find best clocking estimate
            # 1) Subtract off the peak
            # 2) Square the data (because it looks like an abs value function)
            # 3) Perform a fit to a parabola
            # polyCoefs[0]*clockVec**2 + polyCoefs[1]*clockVec + polyCoefs[2]
            overlapVec = overlapVec - np.max(overlapVec)
            polyCoefs = np.polyfit(magVec, overlapVec**2, 2)
            if polyCoefs[0] == 0:  # Avoid divide by zero
                magEst = 1.
            else:
                magEst = -polyCoefs[1]/(2*polyCoefs[0])

    return magEst, clockEst
