"""
Function to unwrap a 2D pupil phase given a phase and amplitude map
"""

import numpy as np
import skimage.morphology
from skimage.restoration import unwrap_phase
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import binary_erosion

from coralign.util.ampthresh import ampthresh
import coralign.util.check as check


def unwrap(phase, amplitude, nbin=21, fill_value=0, min_size=64,
           use_mask=True, bMask=None):
    """
    2D-unwrap a pupil-plane phase

    Intended use is to unwrap phase retrieval outputs at the pupil plane, which
    has a well-defined region of interest (open areas of pupil) and is
    nominally zero elsewhere.  Other uses (e.g. for phases and amplitudes
    which don't have clear regions where the data is good and bad) are
    off-label; reuse this at your own risk.

    Unwraps the phase using the Herraez phase unwrapping algorithm implemented
    in scikit-image.

    Arguments:
     phase: 2D real ndarray with a map of phase at a pupil in radians.
     amplitude: 2D real ndarray > 0 with a map of amplitude at a pupil.  Must
      be the same dimensions as phase.

    Keyword Arguments:
     nbin: number of bins to pass into ampthresh, the routine which creates the
      mask. There must be at least 3 bins, as the algorithm is looking for a
      minimum in the histogram.  Defaults to 21, which is the ampthresh
      default.
     fill_value: constant value to assign to the phase mask in masked-out
      regions.  Defaults to 0.
     min_size: number of pixels in a disconnected region that must be present
      to keep that region from being removed from the mask.  This keeps noisy
      hot pixels and pupil-edge garbage from being included in the mask.
      Defaults to 64, which is the default for remove_small_objects, the
      underlying routine--if a region has 64 pixels or more, it will survive.
     use_mask: default = True, if True, use ampthresh to create a mask, and use
      the mask to define the regions for unwrapping the phase. if False, no
      mask, and attempt to unwrap phase for the whole image
     bMask: default = None, if given, use this mask, overrides mask generated
      by ampthresh if use_mask = True

    Returns:
     two arrays: the first of the unwrapped phase and the second of the mask
      (1 in regions used for unwrap)

    """

    # Check inputs
    check.twoD_array(phase, 'phase', TypeError)
    check.twoD_array(amplitude, 'amplitude', TypeError)
    if phase.shape != amplitude.shape:
        raise TypeError('phase and amplitude must be the same shape')
    if (phase.imag != 0).any():
        raise TypeError('phase must be real-valued')
    # should we also check that -pi <= phase < pi ??
    if (amplitude.imag != 0).any():
        raise TypeError('amplitude must be real-valued')
    if (amplitude.real < 0).any():
        raise TypeError('amplitude must be non-negative')

    check.positive_scalar_integer(nbin, 'nbin', TypeError)
    if nbin < 3:
        raise ValueError('nbin must be at least 3')
    check.real_scalar(fill_value, 'fill_value', TypeError)
    check.positive_scalar_integer(min_size, 'min_size', TypeError)
    if not isinstance(use_mask, bool):
        raise TypeError('use_mask must be a bool')

    if not bMask is None:
        check.twoD_array(bMask, 'bMask', TypeError)
        if not bMask.shape == phase.shape:
            raise TypeError('bMask must be same shape as phase')

    # if bMask is given, use it; otherwise check use_mask option
    if bMask is None:
        if use_mask:
            # Make a mask by thresholding the pupil image, return bMask = bool mask
            bMaskThresh = ampthresh(amplitude, nbin)

            # remove open regions of pupil smaller than some user specified
            # threshold
            bMask = skimage.morphology.remove_small_objects(bMaskThresh, min_size=min_size)

        else:
            # mask is True everywhere
            bMask = np.ones(phase.shape, dtype=bool)

    # numpy.ma is a module that supports numpy arrays with masks
    # Mask is True to mask out a pixel
    # Mask is False (Zero) for good pixels
    # Then use inverse for the mask array
    #invmask = np.invert(bMask)
    invmask = np.logical_not(bMask)

    # Pass a masked array to the unwrapper.
    # Note the unwrapper operates on a complex number
    phase_mask = np.ma.array(phase, mask=invmask)
    phase_unwrapped = unwrap_phase(phase_mask)

    # Fill in the masked regions with a constant
    #uwphase = np.ma.filled(im_unwrapped, fill_value=fill_value)
    phase_unwrapped[invmask] = fill_value

    # phase_unwrap is type numpy.ma.core.MaskedArray, make it a numpy ndarray
    # to return same type as input
    phase_unwrapped = np.array(phase_unwrapped)

    return phase_unwrapped, bMask

SELEM_DEFAULT = np.array([
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0],
])

def unwrap_segments(phase, amplitude, nbin=21, fill_value=0, min_size=64,
                    use_mask=True, bMask=None, selem=SELEM_DEFAULT):
    """
    calls unwrap with some pre-steps to ensure isolated segments unwrap together
    without integer*2*pi offsets between isolated segments.

    arguments are the same as unwrap

    return values are the same as unwrap

    pre-steps are:
      creating a mask from amplitude (if bMask not given as argument)
      erode the boolean mask by a pixel or two because sometimes the edge pixel at
        the strut is noisy phase
      use nearest neighbor interpolation on phase to fill in the masked strut region. This
      connects the disjoint segments, and thus prevents 2*pi jumps between between them.
      call unwrap() with new phase map and no mask
      apply mask to returned, unwrapped phase

    Arguments:
     phase: 2D real ndarray with a map of phase at a pupil in radians.
     amplitude: 2D real ndarray > 0 with a map of amplitude at a pupil.  Must
      be the same dimensions as phase.

    Keyword Arguments:
     nbin: number of bins to pass into ampthresh, the routine which creates the
      mask. There must be at least 3 bins, as the algorithm is looking for a
      minimum in the histogram.  Defaults to 21, which is the ampthresh
      default.
     fill_value: constant value to assign to the phase mask in masked-out
      regions.  Defaults to 0.
     min_size: number of pixels in a disconnected region that must be present
      to keep that region from being removed from the mask.  This keeps noisy
      hot pixels and pupil-edge garbage from being included in the mask.
      Defaults to 64, which is the default for remove_small_objects, the
      underlying routine--if a region has 64 pixels or more, it will survive.
     use_mask: default = True, if True, use ampthresh to create a mask, and use
      the mask to define the regions for unwrapping the phase. if False, no
      mask, and attempt to unwrap phase for the whole image
     bMask: default = None, if given, use this mask, overrides mask generated
      by ampthresh if use_mask = True
     selem: structure element used for morphological erosion of the binary mask.
      see docs for scipy.ndimage.morphology
      default = SELEM_DEFAULT (two pixel erosion), None = skip morph erosion step

    Returns:
     two arrays: the first of the unwrapped phase and the second of the mask
      (1 in regions used for unwrap)

    """

    # Check inputs
    check.twoD_array(phase, 'phase', TypeError)
    check.twoD_array(amplitude, 'amplitude', TypeError)
    if phase.shape != amplitude.shape:
        raise TypeError('phase and amplitude must be the same shape')
    if (phase.imag != 0).any():
        raise TypeError('phase must be real-valued')
    # should we also check that -pi <= phase < pi ??
    if (amplitude.imag != 0).any():
        raise TypeError('amplitude must be real-valued')
    if (amplitude.real < 0).any():
        raise TypeError('amplitude must be non-negative')

    check.positive_scalar_integer(nbin, 'nbin', TypeError)
    if nbin < 3:
        raise ValueError('nbin must be at least 3')
    check.real_scalar(fill_value, 'fill_value', TypeError)
    check.positive_scalar_integer(min_size, 'min_size', TypeError)
    if not isinstance(use_mask, bool):
        raise TypeError('use_mask must be a bool')

    if not bMask is None:
        check.twoD_array(bMask, 'bMask', TypeError)
        if not bMask.shape == phase.shape:
            raise TypeError('bMask must be same shape as phase')

    if not selem is None:
        check.twoD_array(selem, 'selem', TypeError)


    # if bMask is given, use it; otherwise check use_mask option
    if bMask is None:
        if use_mask:
            # Make a mask by thresholding the pupil image, return bMask = bool mask
            bMaskThresh = ampthresh(amplitude, nbin)

            # remove open regions of pupil smaller than some user specified
            # threshold
            bMask = skimage.morphology.remove_small_objects(bMaskThresh, min_size=min_size)

        else:
            # mask is True everywhere
            bMask = np.ones(phase.shape, dtype=bool)

    # erode mask edges in case of phase noise at strut edges
    if not selem is None:
        bMaskErode = binary_erosion(bMask, structure=selem)

    else:
        bMaskErode = bMask.copy()

    # nearest neighbor to fill in the struts
    ny, nx = phase.shape
    x = (np.arange(nx) - nx//2)
    y = (np.arange(ny) - ny//2)
    X, Y = np.meshgrid(x, y)
    interp = NearestNDInterpolator(list(zip(X[bMaskErode], Y[bMaskErode])), phase[bMaskErode])

    # do the interpolation in the masked out regions (i.e. struts)
    phase_interp = phase.copy()
    invmask = np.logical_not(bMaskErode)
    phase_interp[invmask] = interp(X[invmask], Y[invmask])

    # now call unwrap on the phase map with struts filled in and no mask
    phase_unwrap, _ = unwrap(phase_interp, amplitude, use_mask=False)

    # unwrapped phase inside the struts is meaningless, apply the mask
    phase_unwrap[np.logical_not(bMaskErode)] = fill_value

    return phase_unwrap, bMaskErode
