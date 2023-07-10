"""Functions to solve for plate scale ratio between two images."""
import os
import numpy as np
from scipy.interpolate import interp1d

from coralign.util.mft import do_mft, do_imft
import coralign.util.check as check
from coralign.util.loadyaml import loadyaml

# sort out paths
LOCAL_PATH = os.path.dirname(os.path.realpath(__file__))


class PlateSolvException(Exception):
    """Empty exception class."""

    pass


def platesolv(img, guess_img, pixper_guess, yaml_param_file=None):
    """
    Solve for relative plate scale between two images.

    Solves for relative plate scale between two images, one 'recorded' (img)
    with unknown platescale, and the other 'simulated' (img_sim) with a known
    platescale (lambda/D/pixel). Function uses an 'blind' iterative approach
    to calculate the best-fit plate scale by simulating images at a range of
    platescales centered about the best guess (pixper_sim).

    The residuals as a function of trial platescale are calculated and
    interpolated over to determine the best fit value.

    Function assumes recorded and simulated images are aligned to each other.

    Parameters
    ----------
    img : :obj:'arr' of :obj:`float`
        Recorded image for which to solve for platescale

    guess_img : :obj:'arr' of :obj:`float`
        Simulated image with known platescale (e.g. from CGISIM)

    pixper_guess : :obj:`float`
        Platescale (lambda/D/pixel) assumed for img_sim

    yaml_param_file : string
        YAML file that contains relevant constants for exploring different
        platescale solution. YAML file must contain tags named:

            ps_wid:             search width (fraction of nominal plate scale)
                                If 0.1, algorithm will try plate scales at
                                +/-10%

            n_trials:           number of trial plate scales to attempt within
                                ps_wid

            percentile_val:     percentile filter value for image scaling
                                between simulated and recorded image

            interp_step_size:   interpolated plate scale step size
                                sets precision of estimated plate scale

    Returns
    -------
    best_fit : :obj:`float`
        Best fit platescale for img, based on comparison to img_sim

    pix_per_arr : :obj:'arr' of :obj:`float`
        Array of trial plate scales attempted to estimate best_fit

    resid_norm_arr : :obj:'arr' of :obj:`float`
        Array of standard deviations of residual images generated
        at each plate scale in pix_per_arr
    """
    # input checks
    check.twoD_array(img, 'img', PlateSolvException)
    check.twoD_array(guess_img, 'guess_img', PlateSolvException)
    check.real_positive_scalar(pixper_guess, 'pixperlod', PlateSolvException)

    # check that the two images are the same size
    if img.shape != guess_img.shape:
        raise PlateSolvException('img and guess_img are not the same size')

    # get relevant constants from yaml file
    if yaml_param_file is None:
        yaml_param_file = os.path.join(LOCAL_PATH, 'defaults',
                                       'platesolv_parms.yaml')
    consts = loadyaml(yaml_param_file)

    # check that yaml file has correct tags
    if 'ps_wid' not in consts:
        raise PlateSolvException('ps_wid missing from YAML file.')
    if 'n_trials' not in consts:
        raise PlateSolvException('n_trials missing from YAML file.')
    if 'percentile_val' not in consts:
        raise PlateSolvException('percentile_val missing from YAML file.')
    if 'interp_step_size' not in consts:
        raise PlateSolvException('interp_step_size missing from YAML file.')

    # setup bounds for checking plate scale
    # currently +/-20% about expected value
    wid = pixper_guess * consts['ps_wid']

    # number of plate scale values to try about the initial guess
    n_samp = consts['n_trials']

    # array of 'test' plate scales
    pix_per_arr = np.arange(pixper_guess - wid,
                            pixper_guess + wid,
                            wid / (n_samp / 2.))

    # pixperpupil value does not need to be accurate, just the same for all
    # mft and ifmt calls
    pixperpupil = np.max(guess_img.shape)

    # get inverse MFT of simulated image using assumed platescale
    imft_image_reference = do_imft(guess_img, guess_img.shape,
                                   pixper_guess, pixperpupil)

    # normalized residual array
    resid_norm_arr = []

    # for each trial plate scale, generate PSF image and calculate residuals
    for pix_per in pix_per_arr:

        # generate comparison image using 'ith' plate scale in test array
        trial_img = np.real(do_mft(imft_image_reference, guess_img.shape,
                                   pix_per, pixperpupil))

        # # scale the simulated img to match SNR of recorded img
        # filter out top value
        max_val = np.percentile(np.real(img), consts['percentile_val'])
        trial_img *= max_val / np.percentile(np.real(trial_img),
                                             consts['percentile_val'])

        # scale the simulated img area to match the recorded image area
        # area_img = np.sum(np.real(img)) # filter out top value
        # area_trial_img = np.sum(np.real(trial_img))
        # trial_img *= area_img / area_trial_img

        # calculate residuals
        resid_norm = np.std(np.real((trial_img) - img) / np.amax((guess_img)))
        resid_norm_arr.append(resid_norm)

    # interpolate sigma as function of trial plate scale to find minimum
    pix_interp = np.arange(np.amin(pix_per_arr), np.amax(pix_per_arr),
                           consts['interp_step_size'])
    func = interp1d(pix_per_arr, resid_norm_arr, kind='cubic')
    func_eval = func(pix_interp)

    # find out where the interpolated curve is minimum (lowest residuals)
    best_fit = pix_interp[np.where(func_eval == np.amin(func_eval))]

    return best_fit, pix_per_arr, resid_norm_arr
