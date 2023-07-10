"""
Computes Fresnel transforms and fast Fourier transforms
"""

import numpy as np

from coralign.util.pad_crop import pad_crop
from coralign.util import check


def do_fft(e):
    """
    Wrapper on FFT commands to get the shifting right

    Always starts with an array centered at the array center (rather than the
    origin in the corner) and finishes with an array at the array center.
    Does not include explicit normalization of FFT, which uses numpy default
    internally, but does guarantee that ``do_fft(do_ifft(e)) = e`` and
    ``do_ifft(do_fft(e)) = e``.

    Use this to go from pupil planes to focal planes.

    Arguments:
     e: electric field as a numpy ndarray

    Returns:
     a numpy array of the same size as e

    """
    check.twoD_array(e, 'e', TypeError)
    temp = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(e)))
    return temp


def do_ifft(e):
    """
    Wrapper on IFFT commands to get the shifting right

    Always starts with an array centered at the array center (rather than the
    origin in the corner) and finishes with an array at the array center.
    Does not include explicit normalization of FFT, which uses numpy default
    internally, but does guarantee that ``do_fft(do_ifft(e)) = e`` and
    ``do_ifft(do_fft(e)) = e``.

    Use this to go from focal planes to pupil planes.

    Arguments:
     e: electric field as a numpy ndarray

    Returns:
     a numpy array of the same size as e

    """
    check.twoD_array(e, 'e', TypeError)
    temp = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(e)))
    return temp


def fresnelprop(e, lam, z, nxfresnel, pixpermeter):
    """ Fresnel-propagate a field by a distance

    Implicitly assumes field is at or near pupil plane, as it uses an
    angular-spectrum propagator.

    Arguments:
     e: array containg input field; should be complex-valued rather than split
      into real and imaginary parts.
     lam: positive real-valued scalar wavelength in meters
     z: real-valued scalar propagation distance in meters.  may be negative
     nxfresnel: number of pixels to use in the Fresnel propagation. Must be a
      positive integer larger than either element of``e.shape``.
     pixpermeter: pixels per meter at the initial pupil plane, to give
      physical scaling units to the grid. (Fresnel propagation is one of the
      few places these are used.) May be derived e.g. from DM pitch or other
      external calibrations. Real-valued scalar.

    Returns:
     output field.  Array is same size as the input field

    Notes:
     e must be complex-valued
     output will be same size as input?  Do we need an intermediate size?
     need a check that nxfresnel is greater than/equal to nx

    """
    # Check inputs
    check.twoD_array(e, 'e', TypeError)
    check.real_positive_scalar(lam, 'lam', TypeError)
    check.real_scalar(z, 'z', TypeError)
    check.positive_scalar_integer(nxfresnel, 'nxfresnel', TypeError)
    check.real_positive_scalar(pixpermeter, 'pixpermeter', TypeError)

    # In case we got an integer
    pixpermeter = float(pixpermeter)
    lam = float(lam)
    z = float(z)

    # Skip the math for this one case
    if z == 0:
        return e

    nSamplesMin = int(np.ceil(lam * np.abs(z) * (pixpermeter**2)))
    if np.min(e.shape) < nSamplesMin:
        raise ValueError('Angular spectrum sampling criterion violated. '
                         'Input array must be at least %d points across.'
                         % nSamplesMin)

    fresnelrange = (np.arange(nxfresnel) - nxfresnel//2)
    xxf, yyf = np.meshgrid(fresnelrange, fresnelrange)
    rrf = np.hypot(xxf, yyf)  # right spatial frequency units so that
    # FFT of np.exp(2*np.pi*1j*alpha*xxf/nxfresnel) has peak at alpha

    # ang spec = e^(-pi*i*lam*z*R^2), with R in cycles per meter if all
    # lengths are in meters; also see Goodman sec 4.2.3
    fp = np.exp(-1j*np.pi*lam*z*(rrf*(pixpermeter/nxfresnel))**2)

    # FFT, multiply by ang spec quad phase, inverse FFT
    efresnel = pad_crop(e, (nxfresnel, nxfresnel))
    fefresnel = do_fft(efresnel)
    fefresnel *= fp
    efresnel2 = do_ifft(fefresnel)
    return pad_crop(efresnel2, e.shape)


def get_fp(lam, z, nxfresnel, pixpermeter):
    """
    Compute angular spectrum quadratic phase factor

    The quadratic phase array in fp =e^(-pi i lam z R^2), with R in cycles per
    meter, depends on 4 inputs which have only a limited number of inputs for
    most coronagraphic models.  This makes it advantageous in some cases to
    precompute this factor rather than recomputing it many, many times.
    Jacobian calculation is one place where this can be beneficial.

    This function does no optical propagation on its own.

    Arguments:
     lam: positive real-valued scalar wavelength in meters
     z: real-valued scalar propagation distance in meters.  may be negative
     nxfresnel: number of pixels to use in the Fresnel propagation. Must be a
      positive integer larger than either element of``e.shape``.
     pixpermeter: pixels per meter at the initial pupil plane, to give
      physical scaling units to the grid. (Fresnel propagation is one of the
      few places these are used.) May be derived e.g. from DM pitch or other
      external calibrations. Real-valued scalar.

    Returns
     complex-valued 2D array containing quadratic phase factor to be used in a
      subsequent angular spectrum calculation

    """
    check.real_positive_scalar(lam, 'lam', TypeError)
    check.real_scalar(z, 'z', TypeError)
    check.positive_scalar_integer(nxfresnel, 'nxfresnel', TypeError)
    check.real_positive_scalar(pixpermeter, 'pixpermeter', TypeError)

    # In case we got an integer
    pixpermeter = float(pixpermeter)
    lam = float(lam)
    z = float(z)

    nSamplesMin = int(np.ceil(lam * np.abs(z) * (pixpermeter**2)))
    if nxfresnel < nSamplesMin:
        raise ValueError('Angular spectrum sampling criterion violated. '
                         'nxfresnel is %d but must be >= %d.' %
                         (nxfresnel, nSamplesMin))

    fresnelrange = (np.arange(nxfresnel) - nxfresnel//2)
    xxf, yyf = np.meshgrid(fresnelrange, fresnelrange)
    rrf = np.hypot(xxf, yyf)  # right spatial frequency units so that
    # FFT of np.exp(2*np.pi*1j*alpha*xxf/nxfresnel) has peak at alpha

    # ang spec = e^(-pi*i*lam*z*R^2), with R in cycles per meter if all
    # lengths are in meters; also see Goodman sec 4.2.3
    return np.exp(-1j*np.pi*lam*z*(rrf*(pixpermeter/nxfresnel))**2)


def fresnelprop_fp(e, z, nxfresnel, fp):
    """ Fresnel-propagate a field by a distance with a precomputed phase

    Implicitly assumes field is at or near pupil plane, as it uses an
    angular-spectrum propagator.

    One of the inputs is a precomputed quadratic phase factor; this saves the
    repeated effort of computation with the function, which can be a boon
    during Jacobian calculation.

    Arguments:
     e: array containg input field; should be complex-valued rather than split
      into real and imaginary parts.
     z: real-valued scalar propagation distance in meters.  may be negative
     nxfresnel: number of pixels to use in the Fresnel propagation. Must be a
      positive integer larger than either element of``e.shape``.
     fp: complex-valued quadratic phase factor precalculated for speed, such as
      may be output by get_fp()

    Returns:
     output field.  Array is same size as the input field

    Notes:
     e must be complex-valued
     output will be same size as input?  Do we need an intermediate size?
     need a check that nxfresnel is greater than/equal to nx

    """
    # Check inputs
    check.twoD_array(e, 'e', TypeError)
    check.real_scalar(z, 'z', TypeError)
    check.positive_scalar_integer(nxfresnel, 'nxfresnel', TypeError)
    check.twoD_array(fp, 'fp', TypeError)

    # Skip the math for this one case
    if z == 0:
        return e

    # FFT, multiply by ang spec quad phase, inverse FFT
    efresnel = pad_crop(e, (nxfresnel, nxfresnel))
    fefresnel = do_fft(efresnel)
    fefresnel *= fp
    efresnel2 = do_ifft(fefresnel)
    return pad_crop(efresnel2, e.shape)
