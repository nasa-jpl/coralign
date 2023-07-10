"""Set of functions to add fixed shapes on top of existing DM patterns."""

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import convolve
import scipy.sparse
import scipy.sparse.linalg

from coralign.util import check


def fit_surf_to_dm(surf, act_effect):
    """
    Determine DM commands that best match a surface at actuator resolution.

    Given a pre-computed mapping of the effect of each actuator at the location
    of each other A, and a target surf shape b, solve Ax=b to find the DM
    setting x which best reproduces the shape.

    Parameters
    ----------
    surf : numpy ndarray
        should be a nrow x ncol array of surface heights in meters

    act_effect : CSR-type sparse matrix
        should be of size nrow*ncol x nrow*ncol; the output of
        build_prefilter() is a suitable input here

    Returns
    -------
    nrow x ncol ndarray of DM poke heights

    """
    # Check inputs
    check.twoD_array(surf, 'surf', TypeError)
    if not scipy.sparse.issparse(act_effect):
        raise TypeError('act_effect must be a sparse matrix')
    sr, sc = surf.shape
    ar, ac = act_effect.shape
    if not (ar, ac) == (sr*sc, sr*sc):
        raise TypeError('surf and act_effect must be sized to the same DM')

    # solve and re-square
    x = scipy.sparse.linalg.spsolve(act_effect, surf.ravel())
    return np.reshape(x, surf.shape)


def build_prefilter(nrow, ncol, inf_func, ppa_in):
    """
    Build a prefilter.

    The influence function of a DM actuator has a finite extent, and so we can
    map the effect of each actuator on the others by brute force.  For an NxN
    matrix, we can assemble an N^2xN^2 sparse matrix which has the effect of
    poking each actuator on all others in the row.  (Same principle for
    rectangular arrays.)

    This approach is identical to the prefiltering step used when fitting to
    higher-order B-splines for interpolation, although the exact shape of the
    B-spline can be used to make that much faster than the step here.

    Parameters
    ----------
    nrow : int
        Number of rows along one edge of the DM

    ncol : int
        Number of columns along one edge of the DM

    inf_func : numpy ndarray
        2D array with nominal influence function

    ppa_in : float
        Pixels per actuator for inf_func, must be > 0


    Returns
    -------
    CSR-type sparse matrix of size nrow*ncol x nrow*ncol

    """
    check.positive_scalar_integer(nrow, 'nrow', TypeError)
    check.positive_scalar_integer(ncol, 'ncol', TypeError)
    check.twoD_array(inf_func, 'inf_func', TypeError)
    check.real_positive_scalar(ppa_in, 'ppa_in', TypeError)

    # lil_matrix is a good sparse format for incremental build; switch to
    # CSR for operations
    act_effect = scipy.sparse.lil_matrix((nrow*ncol, nrow*ncol))

    # Influence function resampled to actuator map resolution
    ppa_out = 1.  # pixels per actuator; by def'n DM map is 1 pixel/actuator
    inf_func_actres = resample_inf_func(inf_func, ppa_in, ppa_out)

    single_poke = np.zeros((nrow, ncol))

    for j in range(nrow*ncol):
        single_poke.ravel()[j] = 1
        dm_surface = convolve(single_poke, inf_func_actres, mode='constant',
                              cval=0.0)
        single_poke.ravel()[j] = 0  # prep for next
        act_effect[j, :] = dm_surface.ravel()
        pass

    return act_effect.tocsr()  # Want CSR for fast matrix solve later


def resample_inf_func(inf_func, ppa_in, ppa_out):
    """
    Resample an influence function at a new pixels-per-actuator sampling.

    Uses spline interpolation to do the job.

    Parameters
    ----------
    inf_func : numpy ndarray
        2D array with nominal influence function

    ppa_in : float
        Pixels per actuator for inf_func, must be > 0

    ppa_out : float
        Target pixels per actuator in resampled influence function, must be > 0


    Returns
    -------
    2D array with resampled influence function

    """
    check.twoD_array(inf_func, 'inf_func', TypeError)
    check.real_positive_scalar(ppa_in, 'ppa_in', TypeError)
    check.real_positive_scalar(ppa_out, 'ppa_out', TypeError)

    if not ppa_in == ppa_out:
        # Get coords for pixels centers along rows/cols
        nr0, nc0 = inf_func.shape
        r0 = np.linspace(-(nr0-1.)/2., (nr0-1.)/2., nr0)/ppa_in
        c0 = np.linspace(-(nc0-1.)/2., (nc0-1.)/2., nc0)/ppa_in

        # Make output coords, possibly undersized
        # Make odd to have peak of 1
        nr1 = int(2*np.floor(0.5*nr0*ppa_out/ppa_in)+1)
        nc1 = int(2*np.floor(0.5*nc0*ppa_out/ppa_in)+1)

        r1 = np.linspace(-(nr1-1)/2., (nr1-1)/2., nr1)/ppa_out
        c1 = np.linspace(-(nc1-1)/2., (nc1-1)/2., nc1)/ppa_out
        interp_spline = RectBivariateSpline(r0, c0, inf_func)
        inf_func_actres = interp_spline(r1, c1)
    else:
        inf_func_actres = inf_func

    return inf_func_actres
