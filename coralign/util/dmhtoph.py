"""
Functions to create a surface profile from an array of actuator heights.
"""

import numpy as np
import scipy.ndimage
import scipy.signal
import scipy.interpolate

from coralign.util import check
from coralign.util.pad_crop import pad_crop


def dmhtoph(nrow, ncol, dmin, nact, inf_func, ppact_d, ppact_cx, ppact_cy, dx,
            dy, thact, flipx):
    """
    Take a map of actuator pokes, in radians, and produce a surface in radians.

    Actuator pokes correspond to the commanded motion of the actuator
    underlying the facesheet.  These are internal, notional numbers---the
    actuators are buried in the device and cannot be measured directly.  We
    use this intermediate level, rather than just feeding voltages in, to
    allowing voltage-to-actuation nonlinearities in the device to be accounted
    for separately.  This function does not need to know those details; they
    may be different per DM or per DM architecture.

    The surface will, under most circumstances, move following an influence
    function, whose peak is slightly less than the poke height and whose wings
     raise the facesheet in the vicinity of neighboring actuators.

    Our normalization convention ("inffix") sets the influence function
    scaling such that an actuator poke of all actuators by 1 radian moves the
    facesheet uniformly by one radian.  This convention is conserved by this
    function if the input influence function is properly normalize, but is not
     required (e.g. if it is desired to put a quilting order in the model).

    This function gives surface, not OPD.  If you want OPD, multiply the output
    by 2.

    By default the DM grid is
     - centered in the array on the central pixel when the output array size is
      odd, e.g. [-, -, x, -, -]
     - centered on the pixel rightward of the center when the output array size
      is even, e.g. [-, -, x, -].  (FFT convention)
     - Aligned with the Cartesian axes in the absence of rotation.
    The (0,0) actuator is in the lower-left corner of the array.  x- and
    y-directions are relative to the DS9 display of the arrays, which displays
    the second axis along the x-direction.

    Arguments:
     nrow: number of rows in output array. Integer > 0.
     ncol: number of cols in output array. Integer > 0.
     dmin: a 2D ``nact`` x ``nact`` matrix of real numbers representing DM
      poke heights
     nact: number of actuators across the DM.  Should be an positve integer
      and match both ``dmin.shape[0]`` and ``dmin.shape[1]``.  This is not
      being taken directly from dmin to catch potential invalid inputs (as
      this function will happily work with whichever DM setting it gets).
     inf_func: 2D square array with the centered representation on what the
      facesheet deformation looks like for a single poke of unit height.  The
      function must be smaller than the DM (nact*ppact_d).   The edge size must
      be odd-sized so that the central pixel of the array coincides with the
      array peak.
     ppact_d: design pixels per actuator in ``inf_func``.  Expected to be a
      positive scalar integer. ``inf_func`` will in general cover several
      actuators, as influence functions are not perfectly localized;
      ``ppact_d`` is required to denote the underlying grid.  Must be sampled
      better than the camera (ppact_cx and ppact_cy).
     ppact_cx: pixels per actuator in the x-direction on the camera.  Expected
      to be a real positive scalar.  ``ppact_d`` will be downsampled to this.
       Measured by DM registration.
     ppact_cy: pixels per actuator in the y-direction on the camera.  Expected
      to be a real positive scalar.  ``ppact_d`` will be downsampled to this.
       Measured by DM registration.
     dx: number of pixels to shift the DM grid off center in the x-direction
      on the camera. Expected to be a real scalar.  Measured by DM
      registration.
     dy: number of pixels to shift the DM grid off center in the y-direction
      on the camera. Expected to be a real scalar.  Measured by DM
      registration.
     thact: number of degrees to rotate the grid in a counterclockwise
      direction about the center of the array.  Expected to be a real scalar.
      Measured by DM registration.
     flipx: boolean whether to flip the output in the x-direction, leaving the
      y-direction unchanged.  This will only be used if required to
      accommodate DM electronics being wired with a parity flip relative to
      the camera.

    Returns:
     an nrow x ncol array, matched to camera orientation and representing a DM
       surface in radians.

    """
    # Check inputs
    check.positive_scalar_integer(nrow, 'nrow', TypeError)
    check.positive_scalar_integer(ncol, 'ncol', TypeError)
    check.twoD_array(dmin, 'dmin', TypeError)
    check.positive_scalar_integer(nact, 'nact', TypeError)
    check.positive_scalar_integer(ppact_d, 'ppact_d', TypeError)
    check.twoD_array(inf_func, 'inf_func', TypeError)
    check.real_positive_scalar(ppact_cx, 'ppact_cx', TypeError)
    check.real_positive_scalar(ppact_cy, 'ppact_cy', TypeError)
    check.real_scalar(dx, 'dx', TypeError)
    check.real_scalar(dy, 'dy', TypeError)
    check.real_scalar(thact, 'thact', TypeError)
    # No check on flipx since every Python object can be used for truth tests

    if not np.isreal(dmin).all():
        raise TypeError('DM settings must be real-valued')
    if (np.array(inf_func.shape) >= nact*ppact_d).any():
        raise TypeError('Influence function must be smaller than DM')
    if (np.array(dmin.shape) != nact).any():
        raise TypeError('dmin must be nact x nact')
    if (ppact_d < ppact_cx) or (ppact_d < ppact_cy):
        raise TypeError('Design influence function must be sampled '
                        'better than camera')
    if inf_func.shape[0] != inf_func.shape[1]:
        raise TypeError('inf_func must be square')
    if inf_func.shape[0] % 2 != 1:
        raise TypeError('inf_func must be odd-sized')

    # Make sparse array, fill with dmin, size so we get both edges
    arr0 = np.zeros(((nact - 1)*ppact_d + 1, (nact - 1)*ppact_d + 1))
    arr0[::ppact_d, ::ppact_d] = dmin

    carr0 = scipy.signal.convolve(arr0, inf_func, mode='full')
    npix = carr0.shape[0] # (nact - 1)*ppact_d + N, conv of two squares is sq.

    # doing balanced, not FFT convention, for this side of interpolation so
    # the peaks are exactly on integers where possible
    xyin = (np.arange(npix) - (npix-1)/2)/ppact_d
    interpolator = scipy.interpolate.RectBivariateSpline(xyin, xyin, carr0,
                             bbox=[min(xyin), max(xyin), min(xyin), max(xyin)])

    # Make output array for interpolation onto
    mppa = max(ppact_cx, ppact_cy)
    N = inf_func.shape[0]
    # > sqrt(2) to cover 45deg rot
    nxyres = int(np.ceil(np.sqrt(2)*(nact + N/ppact_d)*mppa))
    if nxyres % 2 == 0:
        nxyres += 1  # force odd

    # Use FFT convention for outputs as the rest of the repo is using this
    # convention. Fine since we're picking the output points.
    xout = (np.arange(nxyres) - nxyres//2)/ppact_cx
    yout = (np.arange(nxyres) - nxyres//2)/ppact_cy
    X, Y = np.meshgrid(xout, yout)

    # Do interpolation over a subarea as RectBivariateSpline extrapolates
    # edge points, which is not desired behavior
    interp_inds = np.logical_and(
        np.logical_and(X >= min(xyin), X <= max(xyin)),
        np.logical_and(Y >= min(xyin), Y <= max(xyin)))

    # Expects rows then columns, which in our convention is y then x
    sind = interpolator(Y[interp_inds], X[interp_inds], grid=False)
    s0 = np.zeros((nxyres, nxyres))
    s0[interp_inds] = sind

    # parity, rotation, translation
    if flipx:
        s0 = np.fliplr(s0)
        pass
    dmrot = scipy.ndimage.rotate(s0, -thact, reshape=False)
    surface = scipy.ndimage.shift(dmrot, [dy, dx])

    return pad_crop(surface, (nrow, ncol))



def dmh_to_volts(gainmap, dmh, lam):
    """
    Convert poke heights in radians to voltages with a linear gainmap.

    ``gainmap`` is in meters of poke height (surface) per volt.  (Most
    pokes will be nanometer scale, with large and small excursions to
    micron and picometer scales.)

    Arguments:
     gainmap: a 2D array of actuator gains
     dmh: a 2D array of poke heights.  This function by default does no input
      checking on the range of validity of heights, or of the output voltage
      array.
     lam: wavelength of light to use for radian conversion, in meters

    Returns:
     2D array of voltages, of the same array size as dmh

    """
    check.twoD_array(gainmap, 'gainmap', TypeError)
    check.real_array(gainmap, 'gainmap', TypeError)
    check.twoD_array(dmh, 'dmh', TypeError)
    check.real_array(dmh, 'dmh', TypeError)
    check.real_positive_scalar(lam, 'lam', TypeError)
    if dmh.shape != gainmap.shape:
        raise TypeError('Array of poke heights must be the same ' +
                        'size as the gainmap')

    return dmh*lam/(2.0*np.pi)/gainmap


def volts_to_dmh(gainmap, volts, lam):
    """
    Convert delta volts to poke heights in radians with a linear gainmap.

    ``gainmap`` is in meters of poke height (surface) per volt.  (Most
    pokes will be nanometer scale, with large and small excursions to
    micron and picometer scales.)

    Arguments:
     gainmap: a 2-D array of actuator gains
     volts: a 2D array of voltages, of the same size as ``self.gainmap``.
      This function by default does no input checking on the range of
      validity of voltages
     lam: wavelength of light to use for radian conversion, in meters

    Returns:
     2D array of heights in radians, of the same array size as dmh

    """
    check.twoD_array(gainmap, 'gainmap', TypeError)
    check.real_array(gainmap, 'gainmap', TypeError)
    check.twoD_array(volts, 'volts', TypeError)
    check.real_array(volts, 'volts', TypeError)
    check.real_positive_scalar(lam, 'lam', TypeError)
    if volts.shape != gainmap.shape:
        raise TypeError('Array of voltages must be the same size ' +
                        'as the gainmap')

    return volts/lam*(2.0*np.pi)*gainmap
