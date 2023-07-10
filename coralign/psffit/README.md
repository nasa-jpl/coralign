# PSFFIT

Package for fitting offset between two images at sub-pixel precision, as well as amplitude ratio.

psffit requires the following packages to be installed for use:

* numpy
* scipy
* yaml

Code is partially based on scikit/skimage function 'register_translation':

scikit-image.org/docs/dev/api/skimage.feature.html

Uses methodology described in:
Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup, "Efficient subpixel image registration algorithms," Optics Letters 33, 156-158 (2008). :DOI:`10.1364/OL.33.000156`

### Usage
If running in a separate python function:

  from psffit import psffit

  psffit(source_image, target_image, config_file)

Description of inputs:

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

    Returns
    -------
    shifts :  :obj:'arr' of :obj:`float`
        Computed shift between target_image and source_image [pixels]
        Axis ordering is consistent with numpy (Y, X)

    amplitude : :obj:`float`
        Estimated amplitude ratio (target_image/source_image)

    sigma_amp : :obj:`float`
        Estimated error in amplitude ratio (standard deviation of ratio image)
## Authors

* Sam Halverson
