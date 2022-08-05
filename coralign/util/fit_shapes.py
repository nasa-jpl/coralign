"""Functions to generate 2-D geometric shapes."""
import numpy as np

from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import binary_dilation, binary_erosion
import cv2

from coralign.util.ampthresh import ampthresh
from coralign.util.pad_crop import pad_crop
from coralign.util import check, loadyaml


def fit_circle(image, min_radius, max_radius, edge_sigma=1.0):
    """
    Fit a circle to an image using a hough transform.

    Acts as a wrapper around skimage.feature.canny(), which detects edges, and
    skimage.transform.hough_circle(). The circle fitting algorithm only fits
    radii of integer values and returns integer center locations.

    Parameters
    ----------
    image : array_like
        2-D, real-valued array containing the image to fit.
    min_radius : int
        Minimum allowed radius with which to fit a circle. Must be an integer
        because that is what hough_circle() requires. Units of pixels.
    max_radius : int
        Maximum allowed radius with which to fit a circle. Must be an integer
        because that is what hough_circle() requires. Units of pixels.
    edge_sigma : float, optional
        Gaussian filter width used in the canny edge detection algorithm.
        Larger values accommodate more noise along the edges, but too large of
        a value allows pure noise to be fitted with a circle.
        Must be >0. The default is 1.0. Units of pixels.
        Refer to this website for more info:
        https://scikit-image.org/docs/stable/auto_examples/edges/plot_canny.html

    Returns
    -------
    xOffsetEst, yOffsetEst : int
        Estimated offsets of the fitted circle's center from the array's center
        pixel. Units of pixels.
    radiusEst : int
        Estimated radius of the fitted circle. Units of pixels.

    """
    check.twoD_array(image, 'image', TypeError)
    check.real_array(image, 'image', TypeError)
    check.positive_scalar_integer(min_radius, 'min_radius', TypeError)
    check.positive_scalar_integer(max_radius, 'max_radius', TypeError)
    check.real_positive_scalar(edge_sigma, 'edge_sigma', TypeError)
    if not max_radius > min_radius:
        raise ValueError('max_radius must be larger than min_radius.')

    # Detect edges in a binary image
    # The low_threshold and high_threshold values are hard-coded because
    # they only need to be between 0 and 1 when fitting a thresholded image
    # of all zeros and ones.
    image = ampthresh(image)
    edges = canny(image, sigma=edge_sigma, low_threshold=0.2,
                  high_threshold=0.8)

    # Detect one radius from the range of allowed values.
    # hough_circle() only allows integer radii values.
    hough_radii = np.arange(min_radius, max_radius+1, dtype=int)
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent 1 circle
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=1)

    radiusEst = radii[0]
    xCenterEst = cx[0]
    yCenterEst = cy[0]

    xOffsetEst = xCenterEst - image.shape[1]//2
    yOffsetEst = yCenterEst - image.shape[0]//2

    return xOffsetEst, yOffsetEst, radiusEst


def fit_ellipse(pupil_image, fn_pupil_ellipse_fitting):
    """
    Fit an ellipse to a pupil image using a Hough transform.

    Parameters
    ----------
    pupil_image : array_like
        DESCRIPTION.
    fn_pupil_ellipse_fitting : str
        Name of YAML file containing fitting parameter values. An example of
        what to put in the YAML file is included below:

        # Number of iterations to perform for binary_dilation() and
        # binary_erosion()  from scipy.ndimage.morphology.
        nIterDilateErode: 10

        # Factor by which to pad the pupil image before dilating and eroding.
        # Needs to be large enough such that the dilation and erosion do not
        # hit the edge of the array and fail.
        padFactor: 2.0

    Returns
    -------
    diamEst : float
        major diameter of the ellipse fitted to the pupil. Units of pixels.
    xOffsetEst, yOffsetEst : float
        x- and y-offsets of the fitted ellipse compared to the center pixel of
        the array. Units of pixels.

    """
    check.twoD_array(pupil_image, 'pupil_image', TypeError)
    check.real_array(pupil_image, 'pupil_image', TypeError)

    paramDict = loadyaml.loadyaml(fn_pupil_ellipse_fitting)
    nIterDilateErode = paramDict["nIterDilateErode"]
    padFactor = paramDict["padFactor"]

    nMax = int(np.ceil(padFactor*max(pupil_image.shape)))
    pupil_image = pad_crop(pupil_image, (nMax, nMax))
    pupil_image = ampthresh(pupil_image).astype(float)

    # Dilate then erode the pupil to fill in the struts
    if nIterDilateErode == 0:
        pupilOut = pupil_image
    else:
        struct = generate_binary_structure(2, 2)
        pupilDilated = binary_dilation(pupil_image, structure=struct,
                                       iterations=nIterDilateErode)
        pupilOut = binary_erosion(pupilDilated, structure=struct,
                                  iterations=nIterDilateErode)

    # # Fit an ellipse to the pupil
    # # Where (yc, xc) is the center, (a, b) the major and minor axes,
    # # respectively.
    pupilOut = np.array(pupilOut, dtype=np.uint8)
    contours, hierarchy = cv2.findContours(pupilOut, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    ellipse = cv2.fitEllipse(cnt)

    xc = ellipse[0][0]
    yc = ellipse[0][1]
    a = ellipse[1][0]/2
    b = ellipse[1][1]/2
    # theta = ellipse[2]

    diamEst = 2*np.max([a, b])
    xCenterEst = xc
    yCenterEst = yc

    xOffsetEst = xCenterEst - pupilOut.shape[1]//2
    yOffsetEst = yCenterEst - pupilOut.shape[0]//2

    return diamEst, xOffsetEst, yOffsetEst
