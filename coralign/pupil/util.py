# pylint: disable=maybe-no-member
"""Utility functions for pupil and pupil mask alignment and calibration."""
import numpy as np
# import matplotlib.pyplot as plt

from coralign.maskgen.maskgen import rotate_shift_downsample_amplitude_mask
from coralign.util import check, findopt
from coralign.util.ampthresh import ampthresh
from coralign.util.pad_crop import pad_crop, offcenter_crop


def compute_clocking(ampMeas, diamBeamAtMask, maskRefHighRes, diamHighRes,
                     xOffset, yOffset, clockDegMax, nClock, percentileForNorm,
                     deltaAmpMax, useAmpthresh=False):
    """
    Compute the clocking of a pupil or pupil mask compared to a reference file.

    Clocking is computed by rotating a reference mask by several values,
    summing the matching pixels for each value, and then performing a quadratic
    fit of the squared cost function to find the best clocking value. There are
    two options for what "matching" pixels means. The default is to normalize
    the mask representations and call pixels matching if they are within
    deltaAmpMax of each other. That option works well in simulation but might
    not work as well with real images (TBD). The non-default option is to use
    AMPTHRESH'ed (i.e., boolean) versions of the measured and reference pupils
    and call pixels matching if they have the same exact value (0 or 1).

    Parameters
    ----------
    ampMeas : array_like
        2-D measurement of pupil amplitude. Can be real- or complex-valued.
    diamBeamAtMask : float
        Diameter of beam arriving at mask to fit. Units of pixels.
    maskRefHighRes : array_like
        2-D reference mask amplitude at high resolution.
    diamHighRes : float
        Diameter of beam relative to the high-resolution mask. Units of pixels.
    xOffset, yOffset : float
        Known or estimated lateral offsets of the mask relative to the array
        center pixel. Units of pixels.
    clockDegMax : float
        Maximum +/- clocking values to try. Units of degrees. Somewhat larger
        values than the expected range can be useful for getting a good
        quadratic fit.
    nClock : int
        Number of uniformly-spaced clocking values to try between
        +/-clockDegMax and centered around zero.
    percentileForNorm : float
        Percentile at which to normalize the measured and reference masks if
        useAmpthresh == False. The pixels used in determining the percentile
        are found by running AMPTHRESH on the masks; the AMPTHRESH'ed masks are
        not used for later calculations, though.
    deltaAmpMax : float
        Maximum allowed amplitude difference allowed between a measured mask
        and a reference mask for that pixel to be counted as matching in value.
        Used only if useAmpthresh == False.
    useAmpthresh : bool, optional
        Whether to use AMPTHRESH'ed versions of the input arrays instead of
        percentile-normalized versions. Works better if False in simulation,
        but for real images it may be more accurate if set to True.
        The default is False.

    Returns
    -------
    clockEst : float
        Estimate of mask clocking angle compared to the reference mask.
        Units of degrees.

    """
    check.twoD_array(ampMeas, 'ampMeas', TypeError)
    check.real_positive_scalar(diamBeamAtMask, 'diamBeamAtMask', TypeError)
    check.twoD_array(maskRefHighRes, 'maskRefHighRes', TypeError)
    check.real_positive_scalar(diamHighRes, 'diamHighRes', TypeError)
    check.real_scalar(xOffset, 'xOffset', TypeError)
    check.real_scalar(yOffset, 'yOffset', TypeError)
    check.real_scalar(clockDegMax, 'clockDegMax', TypeError)
    check.positive_scalar_integer(nClock, 'nClock', TypeError)
    check.real_nonnegative_scalar(percentileForNorm, 'percentileForNorm', TypeError)
    check.real_nonnegative_scalar(deltaAmpMax, 'deltaAmpMax', TypeError)
    if not isinstance(useAmpthresh, bool):
        raise TypeError('useAmpthresh')

    # Line search for clocking
    clockVec = np.linspace(-clockDegMax, clockDegMax, nClock)
    padFac = 1  # zero padding should be done outside this function

    ampMeas = np.abs(ampMeas)
    if not useAmpthresh:  # Use edge values between 0 and 1 for fitting
        normFacMeas = compute_norm_factor(ampMeas, ampthresh(ampMeas**2),
                                          percentileForNorm)
        ampNorm = ampMeas/normFacMeas
        overlapVec = np.zeros((nClock,))
        for ic, clockDeg in enumerate(clockVec):

            pupilRef = rotate_shift_downsample_amplitude_mask(
                maskRefHighRes, clockDeg, diamBeamAtMask/diamHighRes, xOffset, yOffset, padFac)
            pupilRef = pad_crop(pupilRef, ampMeas.shape)

            normFacRef = compute_norm_factor(pupilRef, ampthresh(pupilRef),
                                             percentileForNorm)
            pupilRefNorm = pupilRef/normFacRef
            compMatrix = np.abs(ampNorm - pupilRefNorm) < deltaAmpMax
            overlapVec[ic] = np.sum(compMatrix)

    else:  # Use only binary values (i.e., 0 and 1) for fitting
        ampBool = ampthresh(ampMeas**2)
        overlapVec = np.zeros((nClock,))
        for ic, clockDeg in enumerate(clockVec):

            pupilRef = rotate_shift_downsample_amplitude_mask(
                maskRefHighRes, clockDeg, diamBeamAtMask/diamHighRes, xOffset, yOffset, padFac)
            pupilRef = pad_crop(pupilRef, ampMeas.shape)

            pupilRefBool = ampthresh(pupilRef)
            overlapVec[ic] = np.sum(ampBool == pupilRefBool)

    # Use quadratic fit to find best clocking estimate
    # 1) Subtract off the peak
    # 2) Square the data (because it looks like an absolute value function)
    # 3) Perform a fit to a parabola
    # parabola: polyCoefs[0]*clockVec**2 + polyCoefs[1]*clockVec + polyCoefs[2]
    overlapVec = overlapVec - np.max(overlapVec)
    polyCoefs = np.polyfit(clockVec, overlapVec**2, 2)
    if polyCoefs[0] == 0:  # Avoid divide by zero
        clockEst = 0.
    else:
        clockEst = -polyCoefs[1]/(2*polyCoefs[0])

    return clockEst


def compute_lateral_offsets(arrayMeas, arrayRef, diamEst, nPhaseSteps, dPixel,
                            nPadFFT, nFocusCrop, nIter=1, useCoarse=True,
                            xOffsetCoarse=0, yOffsetCoarse=0):
    """
    Compute the lateral offsets of a pupil/mask compared to a reference mask.

    The offsets of the measured mask compared to the reference mask are
    computed in the Fourier plane by FFTing the measured and reference
    pupils/masks and then finding the phase ramp that gives the best match of
    the FFTed masks' phases. This function is for fine estimation of the
    offsets whereas coarsely_locate_pupil_offset() is for coarse estimation.

    Parameters
    ----------
    arrayMeas : array_like
        2-D measurement of pupil amplitude. Can be real- or complex-valued.
    arrayRef : array_like
        2-D reference pupil or mask amplitude.
    diamEst : float
        Estimated diameter of the masked pupil, so if there is a Lyot stop or
        SPM make sure to account for the reduced OD of that mask. Used only
        when the useCoarse flag is True.
    nPhaseSteps : int
        How many phase steps in x and y to perform when fitting the phases of
        the FFTed measured and reference pupils/masks.
    dPixel : float
        The step size in pixels (pupil plane) or phase ramp (Fourier plane)
        when performing the fit of the two arrays. The value can be relatively
        coarse (e.g., 0.5 pixels) because a quadratic fit is performed to find
        the best value rather than using the best value directly.
    nPadFFT : int
        How many points across to pad the pupil-containing arrays before
        FFTing. Should be a power of 2 for for efficient FFTs.
    nFocusCrop : int
        How many points across to crop the FFTed pupil representations.
        Recommend an odd number so that the quadratic fit isn't biased toward
        one direction. nFocusCrop depends on the value of nPadFFT.
    nIter : int
        How many times to iterate the quadratic fit. The default is 1. Setting
        to two recenters the 2nd fit on value from the 1st fit, which helps get
        rid of biases off to the side because the cost function being fitted is
        not a true parabola. More than 2 iterations usually doesn't help
        because that bias has already been removed.
    useCoarse : bool
        Whether to do a coarse offset calculation first. The default is True.
        Can set to False when the pupil center is already well known so that
        the coarse centering algorithm doesn't accidentally move the pupil a
        little farther off the true center.
    xOffsetCoarse, yOffsetCoarse: float
        Initial x- and y-offsets from the array center. Used only when
        useCoarse==False. Units of pixels.

    Returns
    -------
    yOffset1, xOffset1 : float
        Estimate of lateral offsets of the measured mask compared to the
        reference mask.

    """
    check.twoD_array(arrayMeas, 'arrayMeas', TypeError)
    check.real_array(arrayMeas, 'arrayMeas', TypeError)
    check.twoD_array(arrayRef, 'arrayRef', TypeError)
    check.real_array(arrayRef, 'arrayRef', TypeError)
    check.real_positive_scalar(diamEst, 'diamEst', TypeError)
    check.positive_scalar_integer(nPhaseSteps, 'nPhaseSteps', TypeError)
    check.real_positive_scalar(dPixel, 'dPixel', TypeError)
    check.positive_scalar_integer(nPadFFT, 'nPadFFT', TypeError)
    check.positive_scalar_integer(nFocusCrop, 'nFocusCrop', TypeError)
    check.boolean(useCoarse, 'useCoarse', TypeError)
    check.positive_scalar_integer(nIter, 'nIter', TypeError)

    # Get first estimates of x- and y-offsets
    if useCoarse:
        xOffsetCoarse, yOffsetCoarse = coarsely_locate_pupil_offset(
            arrayMeas, diamEst)

    # Need to round the estimates for consistency in usage
    xOffsetCoarse = int(np.round(xOffsetCoarse))
    yOffsetCoarse = int(np.round(yOffsetCoarse))

    # windowWidth = 10
    # nIterSmooth = 100
    # nIterDilateErode = 5#10
    # struct = generate_binary_structure(2, 2)

    # Refine the x- and y-offset estimates
    # arrayRef = ampthresh(arrayRef).astype(float)
    # arrayRef = binary_dilation(arrayRef, structure=struct,
    #                            iterations=nIterDilateErode)
    # for ii in range(nIterSmooth):
    #     arrayRef = smooth_surface(arrayRef, windowWidth)
    arrayRefPad = pad_crop(arrayRef, (nPadFFT, nPadFFT))
    EfocRef = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(arrayRefPad))) /\
        nPadFFT
    EfocRefCrop = pad_crop(EfocRef, (nFocusCrop, nFocusCrop))

    # arrayMeas = ampthresh(arrayMeas).astype(float)
    # arrayMeas = binary_dilation(arrayMeas, structure=struct,
    #                             iterations=nIterDilateErode)
    # for ii in range(nIterSmooth):
    #     arrayMeas = smooth_surface(arrayMeas, windowWidth)
    arrayMeasRecenter = offcenter_crop(
        arrayMeas, nPadFFT,
        (arrayMeas.shape[0]//2 + yOffsetCoarse),
        (arrayMeas.shape[1]//2 + xOffsetCoarse))
    Efoc = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(arrayMeasRecenter))) /\
        nPadFFT
    EfocCrop = pad_crop(Efoc, (nFocusCrop, nFocusCrop))

    # plt.figure(1)
    # plt.imshow(pad_crop(np.abs(arrayRefPad), (400, 400)))
    # plt.title('Reference')
    # plt.gca().invert_yaxis()
    # # plt.colorbar()
    # plt.pause(1e-2)

    # plt.figure(2)
    # plt.imshow(pad_crop(np.abs(arrayMeasRecenter), (400, 400)))
    # plt.title('Measured and Recentered')
    # plt.gca().invert_yaxis()
    # # plt.colorbar()
    # plt.pause(1e-2)

    # # plt.figure(1)
    # # plt.imshow(np.abs(EfocRefCrop))
    # # plt.title('Reference')
    # # plt.gca().invert_yaxis()
    # # # plt.colorbar()
    # # plt.pause(1e-2)

    # # plt.figure(2)
    # # plt.imshow(np.abs(EfocCrop))
    # # plt.title('Measured and Recentered')
    # # plt.gca().invert_yaxis()
    # # # plt.colorbar()
    # # plt.pause(1e-2)

    # plt.figure(11)
    # plt.imshow(arrayRefPad)
    # plt.title('Reference')
    # plt.gca().invert_yaxis()
    # # plt.colorbar()
    # plt.pause(1e-2)

    # plt.figure(12)
    # plt.imshow(arrayMeasRecenter)
    # plt.title('Measured and Recentered')
    # plt.gca().invert_yaxis()
    # # plt.colorbar()
    # plt.pause(1e-2)

    xis = np.arange(-nPadFFT/2, nPadFFT/2)/nPadFFT
    etas = xis
    [XIS, ETAS] = np.meshgrid(xis, etas)
    XIScrop = pad_crop(XIS, (nFocusCrop, nFocusCrop))
    ETAScrop = pad_crop(ETAS, (nFocusCrop, nFocusCrop))

    deltaVec0 = np.linspace(-(nPhaseSteps-1)/2, (nPhaseSteps-1)/2,
                            nPhaseSteps)*dPixel
    deltaX = 0
    deltaY = 0
    dPixelNew = dPixel

    for iter_ in range(nIter):

        if iter_ > 1:
            dPixelNew = dPixelNew/2.

        deltaVec0 = np.linspace(-(nPhaseSteps-1)/2, (nPhaseSteps-1)/2,
                                nPhaseSteps)*dPixelNew

        deltaVecX = deltaVec0 + deltaX
        deltaVecY = deltaVec0 + deltaY
        costMat = np.zeros((nPhaseSteps, nPhaseSteps))
        for ix in range(nPhaseSteps):
            for iy in range(nPhaseSteps):
                phaseTT = np.exp(2 * np.pi * 1j * (deltaVecX[ix] * XIScrop +
                                                   deltaVecY[iy] * ETAScrop))
                costMat[iy, ix] = np.sum(
                    np.abs(EfocCrop * phaseTT - EfocRefCrop)**2)

        # plt.figure(3)
        # plt.imshow(costMat)
        # plt.title('Iter = %d' % (iter_))
        # plt.gca().invert_yaxis()
        # # plt.colorbar()
        # plt.pause(1e-2)

        # Find the values that minimize the cost function.
        # At the first iteration, uses the minimum instead of a
        # quadratic fit in order to get a larger capture range.
        if iter_ == 0 and nIter > 1:
            bestInds = np.unravel_index(np.argmin(costMat), costMat.shape)
            xSol = deltaVecX[bestInds[1]]
            ySol = deltaVecY[bestInds[0]]
        else:
            # Fit to a 3-D paraboloid to find best (x,y) center
            xSol, ySol = findopt.find_optimum_2d(
                deltaVecX, deltaVecY, costMat, np.ones_like(costMat))

        xOffsetFine = xOffsetCoarse + xSol
        yOffsetFine = yOffsetCoarse + ySol

        deltaX = xSol
        deltaY = ySol

    return yOffsetFine, xOffsetFine


def coarsely_locate_pupil_offset(pupil_image, n_points_beam, shrink_factor=3.):
    """
    Iterate coarsely_locate_pupil_offset_once to find pupil center offset.

    This function calls the coarse pupil center locator function in a
    loop. Each iteration narrows down the window size and the step size
    used until after the step size has reached 1 pixel. The pupil
    center location should be accurate to within a pixel or two.

    Parameters
    ----------
    pupil_image : numpy ndarray
        2-D image of the pupil.
    n_points_beam : float
        Width of the pupil in detector pixels.
    shrink_factor : float
        Amount to reduce the search radius and search step size each
        iteration.  Must be > 1.0 and this is enforced, as the search
        will not converge otherwise.

    Returns
    -------
    x_offset_estimate : int
        Estimate of the pupil's x-axis offset the from image center pixel.
        Units of detector pixels.
    y_offset_estimate : int
        Estimate of the pupil's y-axis offset from the image center pixel.
        Units of detector pixels.

    Notes
    -----
    All alignment units are in detector pixels.
    """
    check.twoD_array(pupil_image, 'pupil_image', TypeError)
    check.real_array(pupil_image, 'pupil_image', TypeError)
    check.real_positive_scalar(n_points_beam, 'n_points_beam', TypeError)
    check.real_positive_scalar(shrink_factor, 'shrink_factor', TypeError)
    if shrink_factor <= 1.0:
        raise TypeError('shrink_factor must be > 1.0')

    pixel_count_y, pixel_count_x = pupil_image.shape
    search_radius = np.max((pixel_count_x//2, pixel_count_y//2))
    search_step_size = max(1, int(np.floor(n_points_beam/shrink_factor)))
    x_offset_start = 0
    y_offset_start = 0

    while not search_step_size == 0:

        x_offset_est, y_offset_est, _ = coarsely_locate_pupil_offset_once(
            pupil_image, n_points_beam, x_offset_start,
            y_offset_start, search_radius, search_step_size)
        x_offset_start = x_offset_est
        y_offset_start = y_offset_est

        search_radius = 2*search_step_size

        if search_step_size == 1:  # end condition for loop
            search_step_size = 0
        else:
            search_step_size = int(search_step_size/float(shrink_factor))
            if search_step_size < 1:
                search_step_size = 1

    return x_offset_est, y_offset_est


def coarsely_locate_pupil_offset_once(pupil_image, n_points_beam,
             x_offset_start, y_offset_start, search_radius, search_step_size):
    """
    Roughly calculate the offset of a pupil image from the center pixel.

    This function looks for the approximate center of the pupil by
    summing the image times a region of interest (ROI) the size of the
    pupil being scanned over a square grid. The pixel location giving
    the highest sum is taken as the center pixel. This method is good
    for a coarse fit only and should be followed up with a more accurate
    method.

    Parameters
    ----------
    pupil_image : numpy ndarray
        2-D image of the pupil.
    n_points_beam : float
        Width of the pupil in detector pixels.
    x_offset_start, y_offset_start : int
        Initial guess for center location of the pupil compared to the
        center pixel of the image.
    search_radius : int
        Half-width of square box in which to do a grid search for the
        center of the pupil. Must be a positive scalar integer.
    search_step_size : int
        Number of pixels in between samples in the grid search box.
        Must be a positive scalar integer.

    Returns
    -------
    x_offset_estimate : int
        Estimate of pupil's x-axis misalignment from image center pixel
        in detector pixels.
    y_offset_estimate : int
        Estimate of pupil's y-axis misalignment from image center pixel
        in detector pixels.
    overlap_matrix : numpy ndarray
        2-D array showing the sum of pixels lying within the ROI at each
        specified center pixel. Is an output for diagnostic purposes.

    Notes
    -----
    All alignment units are in detector pixels.

    Nominally 300 (formerly 386) pixels expected across the pupil.
    """
    check.twoD_array(pupil_image, 'pupil_image', TypeError)
    check.real_array(pupil_image, 'pupil_image', TypeError)
    check.real_positive_scalar(n_points_beam, 'n_points_beam', TypeError)
    check.scalar_integer(x_offset_start, 'x_offset_start', TypeError)
    check.scalar_integer(y_offset_start, 'y_offset_start', TypeError)
    check.positive_scalar_integer(search_radius, 'search_radius', TypeError)
    check.positive_scalar_integer(search_step_size, 'search_step_size',
                                  TypeError)

    # Coordinates and indexing for the image
    pixel_count_y, pixel_count_x = pupil_image.shape
    center_index_x = pixel_count_x//2
    center_index_y = pixel_count_y//2
    xs = np.arange(np.ceil(-pixel_count_x/2.), np.ceil(pixel_count_x/2.),
                   dtype=int)
    ys = np.arange(np.ceil(-pixel_count_y/2.), np.ceil(pixel_count_y/2.),
                   dtype=int)

    x_offsets = np.arange(-search_radius, search_radius+1, search_step_size,
                          dtype=int)
    y_offsets = x_offsets
    n_offsets = x_offsets.size
    overlap_matrix = np.zeros((n_offsets, n_offsets))
    for ix in range(n_offsets):
        x_offset = x_offsets[ix]
        if ((x_offset + x_offset_start + center_index_x >= 0) or
        (x_offset + x_offset_start + center_index_x <= (pixel_count_x-1))):

            for iy in range(n_offsets):
                y_offset = y_offsets[iy]
                if ((y_offset + y_offset_start + center_index_y >= 0) or
                    (y_offset + y_offset_start + center_index_y <=
                     (pixel_count_y-1))):

                    [XS, YS] = np.meshgrid(xs-x_offset-x_offset_start,
                                           ys-y_offset-y_offset_start)
                    RS = np.sqrt(XS**2 + YS**2)
                    ROI = np.zeros((pixel_count_y, pixel_count_x), dtype=int)
                    ROI[RS <= n_points_beam/2.0] = 1
                    overlap_matrix[iy, ix] = np.sum(pupil_image[ROI == 1])

    indices_of_max = np.unravel_index(np.argmax(overlap_matrix, axis=None),
                                      overlap_matrix.shape)

    # Initial (x,y) misalignment estimate.
    x_offset_estimate = x_offsets[indices_of_max[1]] + x_offset_start
    y_offset_estimate = y_offsets[indices_of_max[0]] + y_offset_start

    return x_offset_estimate, y_offset_estimate, overlap_matrix


def compute_norm_factor(unnorm_image, software_mask, percentile_for_norm):
    """
    Get the normalization factor for a pupil image.

    Parameters
    ----------
    unnorm_image : numpy ndarray
        2-D pupil map to normalize
    software_mask : numpy ndarray
        2-D array of booleans (0 or 1) indicating where to compare the
        imaged pupil and simulated pupil
    percentile_for_norm : float
        Percentile between 0 and 100 inclusive at which to take the value for
        normalization from the measured image.

    Returns
    -------
    norm_factor : float
        Normalization value for the input pupil image
    """
    check.twoD_array(unnorm_image, 'unnorm_image', TypeError)
    check.real_array(unnorm_image, 'unnorm_image', TypeError)
    check.twoD_array(software_mask, 'software_mask', TypeError)
    check.real_array(software_mask, 'software_mask', TypeError)

    if unnorm_image.shape != software_mask.shape:
        raise TypeError('unnorm_image and software_mask must have the ' +
                        'same shape')

    if np.logical_and(software_mask != 0, software_mask != 1).any():
        raise TypeError('software_mask can only contain zeros or ones')

    check.real_scalar(percentile_for_norm, 'percentile_for_norm', TypeError)
    if percentile_for_norm < 0 or percentile_for_norm > 100:
        raise TypeError('Percentiles must be in the range [0, 100]')

    is_mask_all_zeros = np.sum(software_mask) == 0
    if not is_mask_all_zeros:
        norm_factor = np.percentile(unnorm_image[software_mask == 1],
                                    percentile_for_norm)
    else:
        raise TypeError('software_mask cannot be all zeros')

    return norm_factor
