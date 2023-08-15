"""Functions for finding the stellar offsets from the specified focal mask."""
import numpy as np

from coralign.util import check, shapes
from coralign.util.loadyaml import loadyaml
from coralign.util.debug import debug_plot
import coralign.focalfit.ff_util as ffu


def calc_bowtie_offsets_from_spots(imageSpotted, xOffsetStarFromCenterPixel,
                                   yOffsetStarFromCenterPixel, fnTuning,
                                   debug=False):
    """
    Calculate stellar offset from bowtie FPM by comparing satellite spots.

    The offset is calculated assuming a quadratic relationship with the ratio
    of summed spot intensities. The y-offset calculation needs two pairs of
    spots placed in the corners of the bowtie mask, and the x-offset
    calculation needs one pair of spots placed at the outer, center part of the
    bowtie openings. By default this function always generates two pairs of
    spots, so for the x-offset calculation the two spot pairs need to overlap.

    Parameters
    ----------
    imageSpotted : numpy ndarray
        Processed intensity image of the probes (spots). Calculated outside
        this function from probed images as (Iplus + Iminus)/2 - Iunprobed.
    xOffsetStarFromCenterPixel, yOffsetStarFromCenterPixel : float
        Number of pixels in x and y that the star is offset from the center
        pixel of the array imageSpotted.
    fnTuning : str
        Name of the YAML file containing the tuning parameters and other data
        used for fitting the bowtie focal plane mask.

    Returns
    -------
    xOffsetStarFromMask, yOffsetStarFromMask : float
        Estimated x- and y-offsets of the star from the bowtie focal plane
        mask. Although only the x or y estimate should be used from a given
        call to this function, both are returned at once (because this function
        originally aimed to compute both from the same set of images, but that
        did not work). Therefore, ignore the value returned for the axis that
        you are not computing the offset for.

    Notes
    -----
    Variables in the YAML file are explained below:
    spotSepPix : float
        Expected separation of the satellite spots from the star. Used as the
        separation for the center of the region of interest. Units of pixels.
    spotRotDeg : float
        Expected rotations of the two satellite spot pairs around the star will
        be +/-spotRotDeg from the x-axis. Units of degrees.
    spotExtraRotDeg : float
        Extra clocking applied to make the regions of interest land on the
        spot pairs in the image. Might be needed to account for clocking
        among the DMs, coronagraph masks, and/or detector pixel grid. Can
        also be set to 90 to have the ROIs be clustered about the y-axis.
        Units of degrees. Must be a real positive scalar.
    roiRadiusPix : float
        Radius of each region of interest used when summing the intensity of a
        satellite spot. Units of pixels.
    nSubpixels : int
        Number of subpixels across used to make edge values of the region-of-
        interest mask. The value of the edge pixels in the ROI is the mean of
        all the subpixel values.
    maxStep : float
        Maximum allowed estimate of the star's offset from the mask. This max
        exists because after a certain offset one spot gets completely blocked
        and the true offset is unknown. This max value is chosen to be equal to
        or less than the offset value at which the spot gets completely
        blocked. Units of pixels.
    fitCoefPow1x : float
        Based on simulation, there is a quadratic change in the summed
        ROI ratio versus stellar offset from the bowie mask. fitCoefPow1x
        is the coefficient of the linear term in that polynomial fit. Must be a
        positive scalar value to guarantee that the offset estimate has the
        correct sign. This value is for the x-axis.
    fitCoefPow2x : float
        Based on simulation, there is a quadratic change in the summed
        ROI ratio versus stellar offset from the bowtie mask. fitCoefPow2x
        is the coefficient of the quadratic term in that polynomial fit. Must
        be positive to avoid divide by zero and to avoid choosing the wrong
        answer from the quadratic formula. This value is for the x-axis.
    fitCoefPow1y : float
        Based on simulation, there is a quadratic change in the summed
        ROI ratio versus stellar offset from the bowie mask. fitCoefPow1y
        is the coefficient of the linear term in that polynomial fit. Must be a
        positive scalar value to guarantee that the offset estimate has the
        correct sign. This value is for the y-axis.
    fitCoefPow2y : float
        Based on simulation, there is a quadratic change in the summed
        ROI ratio versus stellar offset from the bowtie mask. fitCoefPow2y
        is the coefficient of the quadratic term in that polynomial fit. Must
        be positive to avoid divide by zero and to avoid choosing the wrong
        answer from the quadratic formula. This value is for the y-axis.
    xRatioTarget : float
       Desired final ratio of summed intensities along the x-axis. Unitless.
    yRatioTarget : float
       Desired final ratio of summed intensities along the y-axis. Unitless.
    """
    check.twoD_array(imageSpotted, 'imageSpotted', TypeError)
    check.real_scalar(xOffsetStarFromCenterPixel, 'xOffsetStarFromCenterPixel',
                      TypeError)
    check.real_scalar(yOffsetStarFromCenterPixel, 'yOffsetStarFromCenterPixel',
                      TypeError)

    inp = loadyaml(fnTuning)
    spotSepPix = inp['spotSepPix']
    spotRotDeg = inp['spotRotDeg']
    spotExtraRotDeg = inp['spotExtraRotDeg']
    roiRadiusPix = inp['roiRadiusPix']
    nSubpixels = inp['nSubpixels']
    maxStep = inp['maxStep']
    fitCoefPow1x = inp['fitCoefPow1x']
    fitCoefPow2x = inp['fitCoefPow2x']
    fitCoefPow1y = inp['fitCoefPow1y']
    fitCoefPow2y = inp['fitCoefPow2y']
    xRatioTarget = inp['xRatioTarget']
    yRatioTarget = inp['yRatioTarget']

    check.real_positive_scalar(spotSepPix, 'spotSepPix', TypeError)
    check.real_positive_scalar(roiRadiusPix, 'roiRadiusPix', TypeError)
    check.positive_scalar_integer(nSubpixels, 'nSubpixels', TypeError)
    check.real_positive_scalar(maxStep, 'maxStep', TypeError)
    check.real_positive_scalar(fitCoefPow1x, 'fitCoefPow1x', TypeError)
    check.real_positive_scalar(fitCoefPow2x, 'fitCoefPow2x', TypeError)
    check.real_positive_scalar(fitCoefPow1y, 'fitCoefPow1y', TypeError)
    check.real_positive_scalar(fitCoefPow2y, 'fitCoefPow2y', TypeError)

    debug_plot(debug, 1, imageSpotted, 'INPUT IMAGE')

    # Generate the regions of interest (ROIs) for each of the 4 spots
    ny, nx = imageSpotted.shape
    spotRotRad = np.pi/180.0*spotRotDeg
    clocking = np.pi/180.0*spotExtraRotDeg
    roiQuad1 = shapes.circle(nx, ny, roiRadiusPix,
                      spotSepPix*np.cos(spotRotRad+clocking) +
                      xOffsetStarFromCenterPixel,
                      spotSepPix*np.sin(spotRotRad+clocking)
                      + yOffsetStarFromCenterPixel, nSubpixels=nSubpixels)
    roiQuad2 = shapes.circle(nx, ny, roiRadiusPix,
                      -spotSepPix*np.cos(-spotRotRad+clocking) +
                      xOffsetStarFromCenterPixel,
                      -spotSepPix*np.sin(-spotRotRad+clocking)
                      + yOffsetStarFromCenterPixel, nSubpixels=nSubpixels)
    roiQuad3 = shapes.circle(nx, ny, roiRadiusPix,
                      -spotSepPix*np.cos(spotRotRad+clocking) +
                      xOffsetStarFromCenterPixel,
                      -spotSepPix*np.sin(spotRotRad+clocking)
                      + yOffsetStarFromCenterPixel, nSubpixels=nSubpixels)
    roiQuad4 = shapes.circle(nx, ny, roiRadiusPix,
                      spotSepPix*np.cos(-spotRotRad+clocking) +
                      xOffsetStarFromCenterPixel,
                      spotSepPix*np.sin(-spotRotRad+clocking)
                      + yOffsetStarFromCenterPixel, nSubpixels=nSubpixels)

    # Sum intensities in the relevant regions
    sumRoiQuad1 = np.sum(roiQuad1*imageSpotted)
    sumRoiQuad2 = np.sum(roiQuad2*imageSpotted)
    sumRoiQuad3 = np.sum(roiQuad3*imageSpotted)
    sumRoiQuad4 = np.sum(roiQuad4*imageSpotted)

    minVal = np.finfo(float).eps  # Avoid divide by 0
    if sumRoiQuad1 <= 0:
        sumRoiQuad1 = minVal
    if sumRoiQuad2 <= 0:
        sumRoiQuad2 = minVal
    if sumRoiQuad3 <= 0:
        sumRoiQuad3 = minVal
    if sumRoiQuad4 <= 0:
        sumRoiQuad4 = minVal

    upSum = sumRoiQuad1 + sumRoiQuad2
    downSum = sumRoiQuad3 + sumRoiQuad4
    leftSum = sumRoiQuad2 + sumRoiQuad3
    rightSum = sumRoiQuad1 + sumRoiQuad4

    xRatio = rightSum / leftSum
    xOffsetStarFromMask = -ffu.calc_offset_quadratic(fitCoefPow1x,
                                                     fitCoefPow2x,
                                                     xRatio)
    xOffsetTarget = -ffu.calc_offset_quadratic(fitCoefPow1x,
                                              fitCoefPow2x,
                                              xRatioTarget)
    xOffsetStarFromMask = xOffsetStarFromMask - xOffsetTarget

    yRatio = upSum / downSum
    yOffsetStarFromMask = -ffu.calc_offset_quadratic(fitCoefPow1y,
                                                     fitCoefPow2y,
                                                     yRatio)
    yOffsetTarget = -ffu.calc_offset_quadratic(fitCoefPow1y,
                                               fitCoefPow2y,
                                               yRatioTarget)
    yOffsetStarFromMask = yOffsetStarFromMask - yOffsetTarget

    # Set to max allowed magnitude if it exceeds it.
    xOffsetStarFromMask = ffu.bound_value(xOffsetStarFromMask, maxStep)
    yOffsetStarFromMask = ffu.bound_value(yOffsetStarFromMask, maxStep)

    return xOffsetStarFromMask, yOffsetStarFromMask


def calc_offset_from_spots(imageSpotted, xProbeRotDeg,
                           xOffsetStarFromCenterPixel,
                           yOffsetStarFromCenterPixel, fnTuning, fnTarget,
                           debug=False):
    """
    Calculate the stellar offset from a focal mask.

    The offset is calculated assuming a linear or quadratic change in the ratio
    of summed spot intensities from a pair of DM-generated satellite spots.
    This calculation is only along one axis. To use the other axis, just add
    90 degrees to xProbeRotDeg. The variables are named for the x-axis because
    a rotation of zero places the spots along the horizontal axis of the array.

    Parameters
    ----------
    imageSpotted : numpy ndarray
        Processed intensity image of the probes (spots). Calculated outside
        this function from probed images as (Iplus + Iminus)/2 - Iunprobed.
    xProbeRotDeg : float
        How many degrees from the x-axis to rotate the regions of interest used
        when summing the satellite spots.
    xOffsetStarFromCenterPixel, yOffsetStarFromCenterPixel : float
        Number of pixels in x and y that the star is offset from the center
        pixel of the array imageSpotted.
    fnTuning : str
        Name of the YAML file containing the tuning parameters and other
        data used for fitting the focal plane mask or field stop.
    fnTarget : str
        Name of the YAML file containing the target ratio value.

    Returns
    -------
    xOffsetStarFromMask : float
        Estimated offset of the star from the mask along the axis defined by
        the selected pair of satellite spots.

    Notes
    -----
    Variables in the YAML files are explained below:
    spotSepPix : float
        Expected separation of the satellite spots from the star. Used as the
        separation for the center of the region of interest. Units of pixels.
    roiRadiusPix : float
        Radius of each region of interest used when summing the intensity of a
        satellite spot. Units of pixels.
    nSubpixels : int
        Number of subpixels across used to make edge values of the region-of-
        interest mask. The value of the edge pixels in the ROI is the mean of
        all the subpixel values.
    maxStep : float
        Maximum allowed estimate of the star's offset from the mask. This max
        exists because after a certain offset one spot gets completely blocked
        and the true offset is unknown. This max value is chosen to be equal to
        or less than the offset value at which the spot gets completely
        blocked. Units of pixels.
    fitCoefPow1 : float
        Based on simulation, there is either a linear or quadratic change in
        the summed ROI ratio versus stellar offset from the mask. fitCoefPow1
        is the coefficient of the linear term in that polynomial fit. Must be a
        positive scalar value to guarantee that the offset estimate has the
        correct sign.
    fitCoefPow2 : float
        Based on simulation, there is either a linear or quadratic change in
        the summed ROI ratio versus stellar offset from the mask. fitCoefPow2
        is the coefficient of the quadratic term in that polynomial fit. Must
        be positive to avoid divide by zero and to avoid choosing the wrong
        answer from the quadratic formula.
    powerOfFit : int, {1, 2}
        Whether to perform a linear or quadratic fit. 1 for linear or 2 for
        quadratic. If 1, then the value of fitCoefPow2 is ignored.
    maskIsInsideSpots : int {0, 1}
        Whether the mask doing the cutting of the spots is inside or outside
        the spots.  An NFOV FPM has the mask on the inside (=1); an NFOV field
        stop has the mask on the outside (=0).
    targetRatio : float
        Desired final ratio of summed intensities along the specified axis.
        Unitless.
    """
    check.twoD_array(imageSpotted, 'imageSpotted', TypeError)
    check.real_scalar(xProbeRotDeg, 'xProbeRotDeg', TypeError)
    check.real_scalar(xOffsetStarFromCenterPixel, 'xOffsetStarFromCenterPixel',
                      TypeError)
    check.real_scalar(yOffsetStarFromCenterPixel, 'yOffsetStarFromCenterPixel',
                      TypeError)

    inp = loadyaml(fnTarget)
    xRatioTarget = inp['targetRatio']
    check.real_positive_scalar(xRatioTarget, 'xRatioTarget', TypeError)

    inp = loadyaml(fnTuning)
    spotSepPix = inp['spotSepPix']
    roiRadiusPix = inp['roiRadiusPix']
    nSubpixels = inp['nSubpixels']
    maxStep = inp['maxStep']
    fitCoefPow1 = inp['fitCoefPow1']
    fitCoefPow2 = inp['fitCoefPow2']
    powerOfFit = inp['powerOfFit']
    maskIsInsideSpots = inp['maskIsInsideSpots']

    check.real_positive_scalar(spotSepPix, 'spotSepPix', TypeError)
    check.real_positive_scalar(roiRadiusPix, 'roiRadiusPix', TypeError)
    check.positive_scalar_integer(nSubpixels, 'nSubpixels', TypeError)
    check.real_positive_scalar(maxStep, 'maxStep', TypeError)
    check.real_positive_scalar(fitCoefPow1, 'fitCoefPow1', TypeError)
    check.real_positive_scalar(fitCoefPow2, 'fitCoefPow2', TypeError)
    check.positive_scalar_integer(powerOfFit, 'powerOfFit', TypeError)
    if not ((powerOfFit == 1) or (powerOfFit == 2)):
        raise ValueError('powerOfFit must be 1 or 2')
    check.scalar_integer(maskIsInsideSpots, 'maskIsInsideSpots', TypeError)
    if not ((maskIsInsideSpots == 0) or (maskIsInsideSpots == 1)):
        raise ValueError('maskIsInsideSpots must be 0 or 1')

    debug_plot(debug, 1, imageSpotted, 'INPUT IMAGE')

    ny, nx = imageSpotted.shape

    xProbeRotRad = np.pi/180.*xProbeRotDeg
    xProbePlusCoord = np.array([np.sin(xProbeRotRad)*spotSepPix +
                                yOffsetStarFromCenterPixel,
                               np.cos(xProbeRotRad)*spotSepPix +
                               xOffsetStarFromCenterPixel])
    xProbeMinusCoord = np.array([np.sin(xProbeRotRad+np.pi)*spotSepPix +
                                 yOffsetStarFromCenterPixel,
                                np.cos(xProbeRotRad+np.pi)*spotSepPix +
                                xOffsetStarFromCenterPixel])

    xProbePlusMask = shapes.circle(nx, ny, roiRadiusPix, xProbePlusCoord[1],
                                   xProbePlusCoord[0], nSubpixels=nSubpixels)
    xProbeMinusMask = shapes.circle(nx, ny, roiRadiusPix, xProbeMinusCoord[1],
                                    xProbeMinusCoord[0], nSubpixels=nSubpixels)

    xProbePlusSum = np.sum(xProbePlusMask * imageSpotted)
    xProbeMinusSum = np.sum(xProbeMinusMask * imageSpotted)

    # Avoid divide by 0
    minVal = np.finfo(float).eps
    if xProbeMinusSum <= 0:
        xProbeMinusSum = minVal
    if xProbePlusSum <= 0:
        xProbePlusSum = minVal

    # Compute offset based on ratio of summed spot intensities.
    xRatio = xProbePlusSum / xProbeMinusSum
    if powerOfFit == 1:
        xOffsetStarFromMask = ffu.calc_offset_linear(fitCoefPow1, xRatio)
        xOffsetTarget = ffu.calc_offset_linear(fitCoefPow1, xRatioTarget)

    elif powerOfFit == 2:
        xOffsetStarFromMask = ffu.calc_offset_quadratic(fitCoefPow1,
                                                        fitCoefPow2,
                                                        xRatio)
        xOffsetTarget = ffu.calc_offset_quadratic(fitCoefPow1,
                                                 fitCoefPow2,
                                                 xRatioTarget)

    xOffsetStarFromMask = xOffsetStarFromMask - xOffsetTarget

    xOffsetStarFromMask = ffu.bound_value(xOffsetStarFromMask, maxStep)

    # Sign of offset is inverted when spots are cut from the outside
    if not maskIsInsideSpots:
        xOffsetStarFromMask *= -1

    return xOffsetStarFromMask
