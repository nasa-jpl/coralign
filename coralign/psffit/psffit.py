"""
Functions to locate an unocculted PSF.

References
----------
Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
     "Efficient subpixel image registration algorithms,"
     Optics Letters 33, 156-158 (2008). :DOI:`10.1364/OL.33.000156`

Code is partially based on scikit/skimage function 'register_translation':
scikit-image.org/docs/dev/api/skimage.feature.html
"""

import os
from pathlib import Path

import numpy as np
from scipy import ndimage

from coralign.util.loadyaml import loadyaml

here = Path(os.path.dirname(os.path.abspath(__file__)))
default_config_file = Path(here, 'defaults', 'psffit_parms.yaml')


def compute_upsampled_dft(image_dft, upsampled_region_size,
                          upsample_factor, axis_offsets=None):
    """
    Compute upsampled DFT via matrix multiplication (see reference).

    Loosely based on 'register_translation' function in scikit-image.

    Parameters
    ----------
    image_dft : :obj:'arr' of :obj:`float`
        The input data array (DFT of original image) to upsample.

    upsampled_region_size : :obj:'tuple' of :obj:`int`
        The size of the region to be sampled.  If one integer is provided, it
        is duplicated up to the dimensionality of ``image_dft``.

    upsample_factor : :obj:'int'
        The upsampling factor (interpolation of FFT frequencies).

    axis_offsets : :obj:'arr' of :obj:`int`
        The offset of the region to be sampled relative to image center.

    Returns
    -------
    output : ndarray
            The upsampled DFT of the specified region.
    """
    # assume center of image for upsampling, unless otherwise specified
    if axis_offsets is None:
        axis_offsets = [0, ] * image_dft.ndim

    # If integer is passed for upsampled region, rather than sub-image range
    # list, expand to a list of equal-sized sections.
    # Check if has more than one element.
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [upsampled_region_size, ] * image_dft.ndim

    # check inputs
    compute_upsampled_dft_input_check(image_dft, upsampled_region_size,
                                      upsample_factor, axis_offsets)

    # get relevant dimensions categorized
    dim_properties = list(zip(image_dft.shape, upsampled_region_size,
                              axis_offsets))

    # calculate upsampled dft
    for (n_items, ups_size, ax_offset) in dim_properties[::-1]:
        kernel = ((np.arange(ups_size) - ax_offset)[:, None]
                  * np.fft.fftfreq(n_items, upsample_factor))
        kernel = np.exp(-1. * (2. * np.pi * 1j) * kernel)
        image_dft = np.tensordot(kernel, image_dft, axes=(1, -1))
    return image_dft


def psffit(source_image, target_image, config_file=default_config_file):
    """
    Sub-pixel image alignment via FFT-based cross-correlation.

    Obtains initial estimate of image offset via FFT-based cross-correlation,
    then refines estimate by upsampling the DFT only in a small
    neighborhood of that estimate by means of a matrix-multiply DFT.
    Also computes estimate of amplitude difference between images.

    Loosely based on 'register_translation' function in scikit-image.

    Parameters
    ----------
    source_image : :obj:'arr' of :obj:`float`
        Reference image / array

    target_image : :obj:'arr' of :obj:`float`
        Image/array to register.  Must have same dimensions as source_image.

    config_file : :obj:`str`
        YAML configuration file that contains relevant constants.
        YAML file must contain constants named:
            upsample_factor:  DFT upsamlping factor
            pad_factor: sub-image padding factor for DFT interpolation
            amp_thresh: threshold factor for estimating PSF amplitude
        Defaults to default_config_file, which is delivered with the repository

    Returns
    -------
    shifts :  :obj:'arr' of :obj:`float`
        Computed shift between target_image and source_image [pixels]
        Axis ordering is consistent with numpy (Y, X)

    amplitude : :obj:`float`
        Estimated amplitude ratio (target_image/source_image)

    S Halverson - JPL - 15-Mar-2020
    """
    # check inputs
    psffit_input_check(source_image, target_image, config_file)

    # get relevant constants from config_file
    constants_config = loadyaml(config_file)
    upsample_factor = constants_config['upsample_factor']

    # compute DFTs of source and target image
    src_freq = np.fft.fftn(source_image)
    target_freq = np.fft.fftn(target_image)

    # compute coarse cross-correlation by multiplying DFTs, inverting
    image_product = src_freq * target_freq.conj()
    cross_correlation = np.fft.ifftn(image_product)

    # locate maximum in pixel-precision CCF
    maxima = np.unravel_index(np.argmax(np.abs(cross_correlation)),
                              cross_correlation.shape)
    midpoints = np.array([np.fix(axis_size / 2)
                          for axis_size in src_freq.shape])

    # get pixel-precision shift estimate
    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(src_freq.shape)[shifts > midpoints]

    # if upsampling > 1 (assume this will almost always be the case)
    # redo estimate with matrix multiply DFT
    if upsample_factor != 1:

        # initial shift estimate in upsampled grid pixel units
        shifts = np.round(shifts * upsample_factor) / upsample_factor

        # select the region size with ~50% padding
        upsampled_region_size = np.ceil(upsample_factor *
                                        constants_config['pad_factor'])

        # center of output array at dftshift + 1
        dftshift = np.fix(upsampled_region_size / 2.0)
        upsample_factor = np.array(upsample_factor, dtype=np.float64)
        normalization = (src_freq.size * upsample_factor ** 2.)

        # matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts*upsample_factor
        cross_correlation = compute_upsampled_dft(image_product.conj(),
                                                  upsampled_region_size,
                                                  upsample_factor,
                                                  sample_region_offset).conj()
        # normalize to preserve flux
        cross_correlation /= normalization

        # locate maximum and map back to original pixel grid
        maxima = np.unravel_index(np.argmax(np.abs(cross_correlation)),
                                  cross_correlation.shape)
        maxima = np.array(maxima, dtype=np.float64) - dftshift
        shifts = shifts + maxima / upsample_factor

    # now shift reference image to computed offset
    guess_img = ndimage.shift(source_image, (-1.) * shifts)

    # only use pixels where the the target image has some flux
    # use fraction of max as proxy (externally configured)
    threshmap = target_image > (constants_config['amp_thresh']*
                                np.amax(target_image))

    # estimate amplitude difference from a correlation of target and guess
    # if we want 'c' such that ||targ - c*guess||_2 is minimized,
    # c = pinv(guess)*targ or (g^T@g)^{-1}@g^T@t.  (Inverse is well-defined
    # unless guess = zero vec, and if guess is zero, this was broken anyway...)
    tvec = target_image[threshmap].ravel()
    gvec = guess_img[threshmap].ravel()
    amplitude = (gvec.T @ tvec)/(gvec.T @ gvec)

    # negate the shifts so that they now represent the shifts of the target
    # from the reference
    shifts = [-shifts[0], -shifts[1]]

    return shifts, amplitude


def compute_upsampled_dft_input_check(image_dft, upsampled_region_size,
                                      upsample_factor, axis_offsets):
    """
    Check inputs to compute_upsampled_dft.

    Parameters
    ----------
    image_dft : :obj:'arr' of :obj:`float`

    upsampled_region_size : :obj:'tuple' of :obj:`int`

    upsample_factor : :obj:'int'

    config_file : :obj:'arr' of :obj:`int`
    """
    # input variable type checks
    if not isinstance(image_dft, (list, np.ndarray)):
        raise TypeError('Input DFT image is not array')

    if not isinstance(upsampled_region_size, (list, float, int)):
        raise TypeError('upsampled_region_size is not a number')

    if not isinstance(upsample_factor, (float, int, np.ndarray)):
        raise TypeError('upsample_factor is not a number')

    if not isinstance(axis_offsets, (list, np.ndarray)):
        raise TypeError('axis_offsets is not a number')

    # input variable dimension checks
    if hasattr(upsampled_region_size, "__iter__"):
        if len(upsampled_region_size) != image_dft.ndim:
            raise ValueError("Number of dimensions of upsampled_region_size"
                             "different from image_dft.")

    # assume center of image for upsampling, unless otherwise specified
    if len(axis_offsets) != image_dft.ndim:
        raise ValueError("Number of dimensions of axis_offsets"
                         "different from image_dft.")


def psffit_input_check(source_image, target_image, config_file):
    """
    Check inputs to psffit.

    Parameters
    ----------
    source_image : :obj:'arr' of :obj:`float`

    target_image : :obj:'arr' of :obj:`float`

    upsample_factor : :obj:`int`

    """
    # input variable type checks
    if not isinstance(source_image, (list, np.ndarray)):
        raise TypeError('Input image is not array')

    if not isinstance(target_image, (list, np.ndarray)):
        raise TypeError('Input image is not array')

    # load in config file
    master_files = loadyaml(config_file)

    # check pointer yaml file
    if 'upsample_factor' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'pad_factor' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')
    if 'amp_thresh' not in master_files:
        raise ValueError('Missing parameter in directory pointer YAML file.')

    if not isinstance(master_files['upsample_factor'], (float, int)):
        raise TypeError('upsample_factor is not a number')

    if not isinstance(master_files['pad_factor'], (float, int)):
        raise TypeError('pad_factor is not a number')

    if not isinstance(master_files['amp_thresh'], (float, int)):
        raise TypeError('amp_thresh is not a number')

    # input variable dimension checks
    if source_image.shape != target_image.shape:
        raise ValueError("Images are not the same dimensions")

# # if called from terminal, run with provided parameters
# if __name__ == "__main__":
#     psffit_input_check(sys.argv[1], sys.argv[2], sys.argv[3])
#     # psffit_input_check(*sys.argv)
