# PLATESOLV

Package for fitting plate scale (lambda/D/pixel) between two images - one being a 'recorded' image and the other being a 'guess' image with an associated plate scale. The function generates simulated images spanning a variety of test plate scales, and estimates the optimum value by interpolating over the difference image residuals.

platesolv requires the following packages to be installed for use:

* numpy
* scipy
* astropy (unit test only)

### Usage
If running in a separate python function:

  from platesolv import platesolv

  platescale, pixel_periods, residuals = platesolv(img, guess_img, pixperiod_guess)


Description of inputs:

    Parameters
    ----------
    img : :obj:'arr' of :obj:`float`
        Recorded image for which to solve for platescale

    guess_img : :obj:'arr' of :obj:`float`
        Simulated image with known platescale (e.g. from CGISIM)

    pixperiod_guess : :obj:`float`
        Platescale (lambda/D/pixel) assumed for guess_img

    Returns
    -------
    best_fit : :obj:`float`
        Best fit platescale for img, based on comparison to img_sim

    pix_per_arr : :obj:'arr' of :obj:`float`
        Array of trial plate scales attempted to estimate best_fit

    resid_norm_arr : :obj:'arr' of :obj:`float`
        Array of standard deviations of residual images generated
        at each plate scale in pix_per_arr

    S Halverson - JPL - 11-Sep-2020
## Authors

* Sam Halverson
