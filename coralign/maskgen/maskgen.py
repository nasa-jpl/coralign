"""Module to hold functions for the MASKGEN package."""
import os
import numpy as np
from astropy.io import fits
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import rotate
from scipy.interpolate import RectBivariateSpline

from coralign.maskgen.util.thinfilm import calc_complex_occulter
from coralign.util.pad_crop import pad_crop
from coralign.util.loadyaml import loadyaml
from coralign.util import math
from coralign.util import check

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))


def rotate_shift_downsample_amplitude_mask(
        maskIn, rotDeg, mag, xOffset, yOffset, padFac=1.2, flipx=False):
    """
    Flip, rotate, translate, and downsample a real-valued, 2-D mask.

    Parameters
    ----------
    maskIn : np.ndarray
        2-D array representing the amplitude mask to adjust.
    rotDeg : float
        Amount to rotate the array counterclockwise about the center pixel.
        Units of degrees.
    mag : float
        Magnification of the output mask compared to the input mask. Must be
        0 < mag <= 1. That is, the mask cannot be upsampled.
    xOffset, yOffset : float
        x- and y-offsets of the mask (after rotation) in output-sized pixels.
    padFac : float
        Factor by which to increase the array dimensions with zero padding.
        Array is zero padded in case the mask goes to the edge of the array;
        otherwise, artifacts can appear in the output array. padFac must be
        >= 1. Default value is 1.2.
    flipx : bool
        Flag to flip the mask left-right before doing the other operations
        (rotation, translation, and downsampling).

    Returns
    -------
    maskOut : np.ndarray
        2-D, even-sized, square array containing the altered mask.
    """
    check.twoD_array(maskIn, 'maskIn', TypeError)
    check.real_scalar(rotDeg, 'rotDeg', TypeError)
    check.real_positive_scalar(mag, 'mag', TypeError)
    if mag > 1:
        raise TypeError('This function does not perform upsampling.')
    check.real_scalar(xOffset, 'xOffset', TypeError)
    check.real_scalar(yOffset, 'yOffset', TypeError)
    check.real_positive_scalar(padFac, 'padFac', TypeError)
    if padFac < 1:
        raise TypeError('padFac must be >= 1.')
    check.boolean(flipx, 'flipx', TypeError)

    # Pad the input array to be a square with odd side lengths
    newSideLength = math.ceil_odd(padFac*np.max([maskIn.shape]))
    maskIn = pad_crop(maskIn, (newSideLength, newSideLength))

    # Flip
    if flipx:
        maskIn = np.fliplr(maskIn)

    # Array sizes
    dxIn = 1.0
    dxOut = 1.0 / mag
    nMaskIn = maskIn.shape[0]
    nMaskOut = math.ceil_odd(
        nMaskIn*mag + 2 + 2.*np.max((np.abs(xOffset), np.abs(yOffset)))
    )
    # 2 pixels added to guarantee the offset mask is fully contained in
    # the output array.

    # array-centered coordinates of input matrix [pupil diameters]
    x0 = np.arange(-(nMaskIn-1.)/2., (nMaskIn)/2., 1)*dxIn
    [X0, Y0] = np.meshgrid(x0, x0)
    R0 = np.sqrt(X0**2 + Y0**2)
    Window = 0*R0
    Window[R0 <= dxOut/2.] = 1
    Window = Window/np.sum(Window)

    # To get good grayscale edges, convolve with the correct window
    # before downsampling.
    f_window = ifft2(ifftshift(Window))*nMaskIn
    f_maskIn = ifft2(ifftshift(maskIn))*nMaskIn
    A = fftshift(fft2(f_window*f_maskIn))
    A = np.real(A)

    if not rotDeg == 0:
        A = rotate(A, -rotDeg, reshape=False)

    x1 = (np.arange(-(nMaskOut-1.)/2., nMaskOut/2., 1) - xOffset)*dxOut
    y1 = (np.arange(-(nMaskOut-1.)/2., nMaskOut/2., 1) - yOffset)*dxOut

    # RectBivariateSpline is faster in 2-D than interp2d
    interp_spline = RectBivariateSpline(x0, x0, A)
    Atemp = interp_spline(y1, x1)

    maskOut = np.zeros((nMaskOut+1, nMaskOut+1))
    maskOut[1::, 1::] = np.real(Atemp)

    return maskOut


def gen_hlc_occulter(lam, scaleWithWavelength, shapeOut, fnCalibData,
                     fnOccData, xOffset=0., yOffset=0., data_path=LOCAL_PATH):
    """
    Generate a complex-valued HLC occulting mask.

    A complex-valued transmission of the HLC occulter is generated starting
    from a 2-D array giving the thickness of the PMGI layer and from scalars
    giving the thicknesses of the titanium and nickel layers. This assumes
    that there is no metal except underneath the PMGI.
    The order of operations is:
    1. compute complex transmission from material thicknesses
    1. rotate mask
    2. translate mask
    3. downsample mask

    To finish at the exact mask resolution desired, the downsampling is
    performed in two steps. First, rotate_shift_downsample_amplitude_mask is
    used to slightly downsample the starting, high-res mask representation.
    This intermediate resolution is chosen such that the ratio of the
    intermediate and final resolutions is exactly the ratio of two integers.
    Those two integers are used as the array side lengths (in pixels) so that
    the output of the FFT downsampling is exact.

    All filenames may be absolute or relative paths.  If relative in input
    arguments, they will be relative to the current working directory, not to
    any particular location in Coralign.  If relative within YAML files
    this will be relative to the path in the data_path argument.

    Parameters
    ----------
    lam : float
        Wavelength at which to generate the mask. Units of meters.
    scaleWithWavelength : bool
        Whether to have the FPM diameter shrink linearly with wavelength,
        as is needed for FFTs. If False, the diameter in pixels will be the
        same at each wavelength.
    shapeOut : array_like
        1-D array specifying the 2-D shape of the output array.
    fnCalibData : str
        Name of YAML file containing calibration data needed to generate
        the focal plane mask representation.
    fnOccData : str
        Name of YAML file containing data for the high-resolution occulter
        design file(s) from which to start.
    xOffset, yOffset : float
        x- and y-offsets of the mask (after rotation) in output-sized pixels.

    Returns
    -------
    occ : array_like
        complex-valued, 2-D array of the HLC occulter transmission

    """
    check.real_positive_scalar(lam, 'lam', TypeError)
    check.oneD_array(shapeOut, 'shapeOut', TypeError)
    if not len(shapeOut) == 2:
        raise TypeError('shapeOut must have length 2')
    check.positive_scalar_integer(shapeOut[0], 'shapeOut[0]', TypeError)
    check.positive_scalar_integer(shapeOut[1], 'shapeOut[1]', TypeError)
    check.boolean(scaleWithWavelength, 'scaleWithWavelength', TypeError)
    check.real_scalar(xOffset, 'xOffset', TypeError)
    check.real_scalar(yOffset, 'yOffset', TypeError)
    # All exceptions for the 2 YAML filenames are checked inside loadyaml

    # Load FPM calibtration data from YAML file
    inp = loadyaml(fnCalibData)
    ppl = inp['ppl']  # desired ending pixels per lambda_central/D
    rotDeg = inp['rotDeg']  # CCW rotation of the mask [degrees]
    aoi = inp['aoi']  # Angle of incidence of beam at FPM in instrument [deg]

    # Load occulter data from YAML file
    inp = loadyaml(fnOccData)
    rOccLamD = inp['rOccLamD']  # occulter radius in lambda_central/D
    fnPMGI = inp['fnPMGI']
    lam0 = inp['lam0']  # wavelength assumed in occulter design file [meters]
    hTi = inp['hTi']  # titanium height [meters]
    hNi = inp['hNi']  # nickel height [meters]
    dx = inp['dx']  # pixel width and height in occulter design file [meters]
    FN = inp['FN']  # F number assumed in occulter design file
    polState = inp['polState']  # 0 for s, 1 for p, 2 for mean of s and p
    # reference height away from substrate to use as the zero point for
    # phase calculation [waves]
    hRefCoef = inp['hRefCoef']

    padFac = inp['padFac']
    minOversizeFacForFFT = inp['minOversizeFacForFFT']

    # Checks on values from YAML files
    check.real_positive_scalar(ppl, 'ppl', TypeError)
    check.real_scalar(rotDeg, 'rotDeg', TypeError)
    check.real_scalar(aoi, 'aoi', TypeError)

    check.real_positive_scalar(lam0, 'lam0', TypeError)
    check.real_positive_scalar(hTi, 'hTi', TypeError)
    check.real_positive_scalar(hNi, 'hNi', TypeError)
    check.real_positive_scalar(dx, 'dx', TypeError)
    check.real_positive_scalar(FN, 'FN', TypeError)
    check.nonnegative_scalar_integer(polState, 'polState', TypeError)
    if polState not in (0, 1, 2):
        raise TypeError('polState must be 0, 1, or 2')
    check.real_positive_scalar(hRefCoef, 'hRefCoef', TypeError)
    check.real_nonnegative_scalar(aoi, 'aoi', TypeError)
    if aoi >= 90:
        raise TypeError('aoi must be <90 degrees.')
    check.real_positive_scalar(padFac, 'padFac', TypeError)
    if padFac < 1.0:
        raise TypeError('padFac must be >= 1.0')
    check.real_positive_scalar(minOversizeFacForFFT, 'minOversizeFacForFFT',
                               TypeError)
    if minOversizeFacForFFT < 2.0:
        raise TypeError('minOversizeFacForFFT must be >= 2.0')

    # Load high-res occulter PMGI thickness from Dwight
    pmgi = fits.getdata(os.path.join(data_path, fnPMGI))
    nFPM0 = math.ceil_odd(np.max(pmgi.shape))
    pmgi = pad_crop(pmgi, (nFPM0, nFPM0))  # force to be square and odd-sized
    footprint = np.zeros_like(pmgi, dtype=int)
    footprint[pmgi > 0] = 1

    # Find indices of occulter pixels and substrate-only pixels
    indOcc = np.nonzero(footprint.flatten() == 1)[0]
    indSubstrate = np.nonzero(footprint.flatten() == 0)[0]

    # Generate complex-valued occulter transmission at file resolution
    hRef = hRefCoef*lam0
    hTiVec = hTi*np.ones_like(indOcc)
    hNiVec = hNi*np.ones_like(indOcc)
    hPMGIVec = pmgi.flatten()[indOcc]
    tCoef, rCoef = calc_complex_occulter(lam, aoi, hTiVec, hNiVec, hPMGIVec,
                                         hRef, polState)
    tCoefSub, rCoefSub = calc_complex_occulter(lam, aoi, [0], [0], [0], hRef,
                                               polState)
    occStartFlat = np.zeros(nFPM0*nFPM0, dtype=complex)
    occStartFlat[indSubstrate] = tCoefSub[0]
    occStartFlat[indOcc] = tCoef
    occStart = occStartFlat.reshape((nFPM0, nFPM0))
    occStart = occStart/occStart[0, 0]

    # Compute intermediate resolution mask to be used for FFT downsampling.
    pplStart = (FN * lam0) / dx  # starting pixels per lambda_central/D
    if scaleWithWavelength:
        pplEnd = (lam0 / lam) * ppl
    else:
        pplEnd = ppl
    nPadEnd = int(np.ceil(2*rOccLamD*pplEnd*minOversizeFacForFFT))
    nPadMiddle = int(np.floor((pplStart/pplEnd) * nPadEnd))
    pplMiddle = (nPadMiddle/nPadEnd) * pplEnd
    mag = pplMiddle / pplStart
    occMiddleReal = rotate_shift_downsample_amplitude_mask(
        np.real(occStart - occStart[0, 0]), rotDeg, mag,
        xOffset*(nPadMiddle/nPadEnd), yOffset*(nPadMiddle/nPadEnd), padFac)
    occMiddleImag = rotate_shift_downsample_amplitude_mask(
        np.imag(occStart - occStart[0, 0]), rotDeg, mag,
        xOffset*(nPadMiddle/nPadEnd), yOffset*(nPadMiddle/nPadEnd), padFac)
    occMiddle = occMiddleReal + 1j*occMiddleImag
    occMiddle = pad_crop(occMiddle, (nPadMiddle, nPadMiddle)) + occStart[0, 0]

    # Fourier downsample
    occFT = pad_crop(fftshift(ifft2(ifftshift(occMiddle))), (nPadEnd, nPadEnd))
    occEnd = fftshift(fft2(ifftshift(occFT)))
    occ = pad_crop(occEnd - occStart[0, 0], shapeOut) + occStart[0, 0]

    return occ
