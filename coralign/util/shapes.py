"""Functions to generate 2-D geometric shapes with antialiased edges."""
import numpy as np

from coralign.util import check, math
from coralign.util.pad_crop import pad_crop


def simple_pupil(nx, ny, xOffset, yOffset, diamInner, diamOuter,
                 strutAngles=[], strutWidth=0,  nSubpixels=100):
    """
    Generate a simple pupil with a circular ID and OD and radial struts.

    Parameters
    ----------
    nx, ny : float
        Dimensions of the 2-D array to create.
    xOffset, yOffset : float
        Lateral offsets in pixels of the pupil's center from the array's
        center pixel.
    diamInner : float
        Inner diameter of the pupil in pixels.
    diamOuter : float
        Outer diameter of the pupil in pixels.
    strutAngles : list, optional
        Angles at which to place struts. Units of degrees. The default is [].
    strutWidth : float, optional
        Width of each strut. Units of pixels. The default is 0.
    nSubpixels : int, optional
        Each edge pixel of the pupil is subdivided into a square subarray
        nSubpixels across. The subarray is given binary values and then
        averaged to give the edge pixel a value between 0 and 1, inclusive.
        The default value is 100.

    Returns
    -------
    pupil : numpy ndarray
        2-D array containing the pupil
    """
    check.positive_scalar_integer(nx, 'nx', TypeError)
    check.positive_scalar_integer(ny, 'ny', TypeError)
    check.real_scalar(xOffset, 'xOffset', TypeError)
    check.real_scalar(yOffset, 'yOffset', TypeError)
    check.real_nonnegative_scalar(diamInner, 'diamInner', TypeError)
    check.real_positive_scalar(diamOuter, 'diamOuter', TypeError)
    check.oneD_array(strutAngles, 'strutAngles', TypeError)
    check.real_nonnegative_scalar(strutWidth, 'strutWidth', TypeError)
    check.positive_scalar_integer(nSubpixels, 'nSubpixels', TypeError)
    if diamInner >= diamOuter:
        raise TypeError('diamInner must be less than diamOuter')

    # Primary mirror
    pupil = circle(nx, ny, diamOuter/2, xOffset, yOffset,
                   nSubpixels=nSubpixels)

    # Secondary mirror
    if diamInner > 0:
        pupil *= circle(nx, ny, diamInner/2, xOffset, yOffset,
                        nSubpixels=nSubpixels, isDark=True)

    # Struts
    if len(strutAngles) > 0 and strutWidth > 0:

        # Make slightly more than half to fully cover the outer edge
        height = 0.6 * diamOuter

        for strutAngle in strutAngles:

            strutAngleRad = strutAngle * np.pi/180
            xOffsetStrut = xOffset + np.cos(strutAngleRad)*height/2
            yOffsetStrut = yOffset + np.sin(strutAngleRad)*height/2

            pupil *= rectangle(nx, ny, height, strutWidth, xOffsetStrut,
                               yOffsetStrut, rot=strutAngle,
                               nSubpixels=nSubpixels, isDark=True)

    return pupil


def circle(nx, ny, radius, xOffset, yOffset, nSubpixels=100, isDark=False):
    """
    Generate a laterally shifted circle with antialiased edges.

    Parameters
    ----------
    nx, ny : float
        Dimensions of the 2-D array to create.
    radius : float
        Radius of the circle in pixels.
    xOffset, yOffset : float
        Lateral offsets in pixels of the circle's center from the array's
        center pixel.
    nSubpixels : int, optional
        Each edge pixel of the circle is subdivided into a square subarray
        nSubpixels across. The subarray is given binary values and then
        averaged to give the edge pixel a value between 0 and 1, inclusive.
        The default value is 100. Must be a positive scalar integer.
    isDark : bool
        Flag whether to change the rectangle from being an illuminated region
        to a dark region.

    Returns
    -------
    mask : numpy ndarray
        2-D array containing the circle
    """
    check.positive_scalar_integer(nx, 'nx', TypeError)
    check.positive_scalar_integer(ny, 'ny', TypeError)
    check.real_positive_scalar(radius, 'radius', TypeError)
    check.real_scalar(xOffset, 'xOffset', TypeError)
    check.real_scalar(yOffset, 'yOffset', TypeError)
    check.positive_scalar_integer(nSubpixels, 'nSubpixels', TypeError)

    if nx % 2 == 0:
        x = np.linspace(-nx/2., nx/2. - 1, nx) - xOffset
    elif nx % 2 == 1:
        x = np.linspace(-(nx-1)/2., (nx-1)/2., nx) - xOffset

    if ny % 2 == 0:
        y = np.linspace(-ny/2., ny/2. - 1, ny) - yOffset
    elif ny % 2 == 1:
        y = np.linspace(-(ny-1)/2., (ny-1)/2., ny) - yOffset

    dx = x[1] - x[0]
    [X, Y] = np.meshgrid(x, y)
    RHO = np.sqrt(X*X + Y*Y)

    halfWindowWidth = np.sqrt(2.)*dx
    mask = -1*np.ones(RHO.shape)
    mask[np.abs(RHO) < radius - halfWindowWidth] = 1
    mask[np.abs(RHO) > radius + halfWindowWidth] = 0
    grayInds = np.array(np.nonzero(mask == -1))
    # print('Number of grayscale points = %d' % grayInds.shape[1])

    dxHighRes = 1./float(nSubpixels)
    xUp = np.linspace(-(nSubpixels-1)/2., (nSubpixels-1)/2.,
                      nSubpixels)*dxHighRes
    [Xup, Yup] = np.meshgrid(xUp, xUp)

    subpixelArray = np.zeros((nSubpixels, nSubpixels))
    # plt.figure(); plt.imshow(RHO); plt.colorbar(); plt.pause(0.1)

    # Compute the value between 0 and 1 of each edge pixel along the circle by
    # taking the mean of the binary subpixels.
    for iInterior in range(grayInds.shape[1]):

        subpixelArray = 0*subpixelArray

        xCenter = X[grayInds[0, iInterior], grayInds[1, iInterior]]
        yCenter = Y[grayInds[0, iInterior], grayInds[1, iInterior]]
        RHOHighRes = np.sqrt((Xup+xCenter)**2 + (Yup+yCenter)**2)
        # plt.figure(); plt.imshow(RHOHighRes); plt.colorbar(); plt.pause(1/20)

        subpixelArray[RHOHighRes <= radius] = 1
        pixelValue = np.sum(subpixelArray)/float(nSubpixels*nSubpixels)
        mask[grayInds[0, iInterior], grayInds[1, iInterior]] = pixelValue

    if isDark:
        mask = 1.0 - mask

    return mask


def ellipse(nx, ny, rx, ry, rot, xOffset, yOffset, nSubpixels=100,
            isDark=False):
    """
    Generate a rotated, laterally shifted ellipse with antialiased edges.

    Parameters
    ----------
    nx, ny : array_like
        Dimensions of the 2-D array to create.
    rx, ry : float
        x- and y- radii of the ellipse in pixels.
    rot : float
        Counterclockwise rotation of the ellipse in degrees.
    xOffset, yOffset : float
        Lateral offsets in pixels of the circle's center from the array's
        center pixel.
    nSubpixels : int, optional
        Each edge pixel of the ellipse is subdivided into a square subarray
        nSubpixels across. The subarray is given binary values and then
        averaged to give the edge pixel a value between 0 and 1, inclusive.
        The default value is 100. Must be a positive scalar integer.
    isDark : bool
        Flag whether to change the rectangle from being an illuminated region
        to a dark region.

    Returns
    -------
    mask : numpy ndarray
        2-D array containing the ellipse
    """
    check.positive_scalar_integer(nx, 'nx', TypeError)
    check.positive_scalar_integer(ny, 'ny', TypeError)
    check.real_positive_scalar(rx, 'rx', TypeError)
    check.real_positive_scalar(ry, 'ry', TypeError)
    check.real_scalar(rot, 'rot', TypeError)
    check.real_scalar(xOffset, 'xOffset', TypeError)
    check.real_scalar(yOffset, 'yOffset', TypeError)
    check.positive_scalar_integer(nSubpixels, 'nSubpixels', TypeError)

    rotRad = (np.pi/180.) * rot

    if nx % 2 == 0:
        x = np.linspace(-nx/2., nx/2. - 1, nx) - xOffset
    elif nx % 2 == 1:
        x = np.linspace(-(nx-1)/2., (nx-1)/2., nx) - xOffset

    if ny % 2 == 0:
        y = np.linspace(-ny/2., ny/2. - 1, ny) - yOffset
    elif ny % 2 == 1:
        y = np.linspace(-(ny-1)/2., (ny-1)/2., ny) - yOffset

    [X, Y] = np.meshgrid(x, y)
    dx = x[1] - x[0]
    radius = 0.5

    RHO = 0.5*np.sqrt(
        1/(rx)**2*(np.cos(rotRad)*X + np.sin(rotRad)*Y)**2
        + 1/(ry)**2*(np.sin(rotRad)*X - np.cos(rotRad)*Y)**2
    )

    halfWindowWidth = np.max(np.abs((RHO[1, 0] - RHO[0, 0],
                                     RHO[0, 1] - RHO[0, 0])))
    mask = -1*np.ones(RHO.shape)
    mask[np.abs(RHO) < radius - halfWindowWidth] = 1
    mask[np.abs(RHO) > radius + halfWindowWidth] = 0
    grayInds = np.array(np.nonzero(mask == -1))
    # print('Number of grayscale points = %d' % grayInds.shape[1])

    dxUp = dx/float(nSubpixels)
    xUp = np.linspace(-(nSubpixels-1)/2., (nSubpixels-1)/2., nSubpixels)*dxUp
    [Xup, Yup] = np.meshgrid(xUp, xUp)

    subpixel = np.zeros((nSubpixels, nSubpixels))

    for iInterior in range(grayInds.shape[1]):

        subpixel = 0*subpixel

        xCenter = X[grayInds[0, iInterior], grayInds[1, iInterior]]
        yCenter = Y[grayInds[0, iInterior], grayInds[1, iInterior]]
        RHOup = 0.5*np.sqrt(
            1/(rx)**2*(np.cos(rotRad)*(Xup+xCenter) +
                       np.sin(rotRad)*(Yup+yCenter))**2
            + 1/(ry)**2*(np.sin(rotRad)*(Xup+xCenter) -
                         np.cos(rotRad)*(Yup+yCenter))**2)

        subpixel[RHOup <= radius] = 1
        pixelValue = np.sum(subpixel)/float(nSubpixels**2)
        mask[grayInds[0, iInterior], grayInds[1, iInterior]] = pixelValue

    if isDark:
        mask = 1.0 - mask

    return mask


def rectangle(nx, ny, width, height, xOffset, yOffset, rot=0,
              nSubpixels=100, isDark=False):
    """
    Return an image containing a filled rectangle with antialiased edges.

    Parameters
    ----------
    nx, ny : array_like
        Dimensions of the 2-D array to create.
    width : float
       horizontal width (before rotation) of the rectangle in pixels
    height : float
       vertical width (before rotation) of the rectangle in pixels
    xOffset, yOffset : float
        (x, y) center of the rectangle relative to the center pixel of the
        array. Units of pixels
    rot : float
        Specifies the angle degrees counter-clockwise to rotate the rectangle
        about its center.
    nSubpixels : int, optional
        Each edge pixel of the recangle is subdivided into a square subarray
        nSubpixels across. The subarray is given binary values and then
        averaged to give the edge pixel a value between 0 and 1, inclusive.
        The default value is 100. Must be a positive scalar integer.
    isDark : bool
        Flag whether to change the rectangle from being an illuminated region
        to a dark region.

    Returns
    -------
    image : numpy ndarray
        Returns an image array containing an antialiased rectangular mask with
        the same dimensions as the wavefront array.
    """
    check.positive_scalar_integer(nx, 'nx', TypeError)
    check.positive_scalar_integer(ny, 'ny', TypeError)
    check.real_positive_scalar(width, 'width', TypeError)
    check.real_positive_scalar(height, 'height', TypeError)
    check.real_scalar(xOffset, 'xOffset', TypeError)
    check.real_scalar(yOffset, 'yOffset', TypeError)
    check.real_scalar(rot, 'rot', TypeError)
    check.positive_scalar_integer(nSubpixels, 'nSubpixels', TypeError)
    check.boolean(isDark, 'isDark', TypeError)

    if nx % 2 == 0:
        x = np.linspace(-nx/2., nx/2. - 1, nx) - xOffset
    elif nx % 2 == 1:
        x = np.linspace(-(nx-1)/2., (nx-1)/2., nx) - xOffset

    if ny % 2 == 0:
        y = np.linspace(-ny/2., ny/2. - 1, ny) - yOffset
    elif ny % 2 == 1:
        y = np.linspace(-(ny-1)/2., (ny-1)/2., ny) - yOffset

    dx = x[1] - x[0]
    [X, Y] = np.meshgrid(x, y)
    [RHO, THETA] = math.cart2pol(X, Y)
    # RHO = np.sqrt(X*X + Y*Y)

    halfWindowWidth = np.sqrt(2.)*dx
    mask = -1*np.ones(RHO.shape)
    rotRad = rot * np.pi/180
    inside = np.logical_and(np.logical_and(np.logical_and(
        RHO*np.cos(THETA-rotRad) <= width/2 - halfWindowWidth,
        RHO*np.cos(THETA-rotRad) >= -width/2 + halfWindowWidth),
        RHO*np.sin(THETA-rotRad) <= height/2 - halfWindowWidth),
        RHO*np.sin(THETA-rotRad) >= -height/2 + halfWindowWidth)
    outside = np.logical_or(np.logical_or(np.logical_or(
        RHO*np.cos(THETA-rotRad) >= width/2 + halfWindowWidth,
        RHO*np.cos(THETA-rotRad) <= -width/2 - halfWindowWidth),
        RHO*np.sin(THETA-rotRad) >= height/2 + halfWindowWidth),
        RHO*np.sin(THETA-rotRad) <= -height/2 - halfWindowWidth)
    mask[inside] = 1
    mask[outside] = 0

    grayInds = np.array(np.nonzero(mask == -1))
    # print('Number of grayscale points = %d' % grayInds.shape[1])

    dxHighRes = 1./float(nSubpixels)
    xUp = np.linspace(-(nSubpixels-1)/2., (nSubpixels-1)/2.,
                      nSubpixels)*dxHighRes
    [Xup, Yup] = np.meshgrid(xUp, xUp)

    # Compute the value between 0 and 1 of each edge pixel by
    # taking the mean of the binary subpixels.
    for iInterior in range(grayInds.shape[1]):

        subpixelArray = np.zeros((nSubpixels, nSubpixels))

        xCenter = X[grayInds[0, iInterior], grayInds[1, iInterior]]
        yCenter = Y[grayInds[0, iInterior], grayInds[1, iInterior]]
        [RHOHighRes, THETAHighRes] = math.cart2pol(Xup+xCenter, Yup+yCenter)
        # plt.figure(); plt.imshow(RHOHighRes); plt.colorbar(); plt.pause(1/20)

        inside = np.logical_and(np.logical_and(np.logical_and(
            RHOHighRes*np.cos(THETAHighRes-rotRad) <= width/2,
            RHOHighRes*np.cos(THETAHighRes-rotRad) >= -width/2),
            RHOHighRes*np.sin(THETAHighRes-rotRad) <= height/2),
            RHOHighRes*np.sin(THETAHighRes-rotRad) >= -height/2)
        subpixelArray[inside] = 1

        pixelValue = np.sum(subpixelArray)/float(nSubpixels*nSubpixels)
        mask[grayInds[0, iInterior], grayInds[1, iInterior]] = pixelValue

    if isDark:
        mask = 1.0 - mask

    return mask


def rect_proper(nx, ny, width, height, xOffset, yOffset, rot=0,
              nSubpixels=100, isDark=False):
    """
    Return an image containing a filled rectangle with antialiased edges.

    Modified from prop_rectangle.py from the PROPER package by John Krist.

    Parameters
    ----------
    nx, ny : array_like
        Dimensions of the 2-D array to create.
    width : float
       horizontal width (before rotation) of the rectangle in pixels
    height : float
       vertical width (before rotation) of the rectangle in pixels
    xOffset, yOffset : float
        (x, y) center of the rectangle relative to the center pixel of the
        array. Units of pixels
    rot : float
        Specifies the angle degrees counter-clockwise to rotate the rectangle
        about its center.
    nSubpixels : int, optional
        Each edge pixel of the recangle is subdivided into a square subarray
        nSubpixels across. The subarray is given binary values and then
        averaged to give the edge pixel a value between 0 and 1, inclusive.
        The default value is 100. Must be a positive scalar integer.
    isDark : bool
        Flag whether to change the rectangle from being an illuminated region
        to a dark region.

    Returns
    -------
    image : numpy ndarray
        Returns an image array containing an antialiased rectangular mask with
        the same dimensions as the wavefront array.
    """
    check.positive_scalar_integer(nx, 'nx', TypeError)
    check.positive_scalar_integer(ny, 'ny', TypeError)
    check.real_positive_scalar(width, 'width', TypeError)
    check.real_positive_scalar(height, 'height', TypeError)
    check.real_scalar(xOffset, 'xOffset', TypeError)
    check.real_scalar(yOffset, 'yOffset', TypeError)
    check.real_scalar(rot, 'rot', TypeError)
    check.positive_scalar_integer(nSubpixels, 'nSubpixels', TypeError)
    check.boolean(isDark, 'isDark', TypeError)

    sampling = 1
    ngrid = np.max([nx, ny])
    mag = nSubpixels

    # Set xcpix and ycpix values
    xcpix = xOffset/sampling + ngrid // 2
    ycpix = yOffset/sampling + ngrid // 2

    # Set xradpix and yradpix
    xradpix = 0.5 * width / sampling
    yradpix = 0.5 * height / sampling

    # Rotation angle in radians
    angle_rad = rot * np.pi / 180.

    xp0 = np.array([-xradpix, -xradpix, xradpix, xradpix])
    yp0 = np.array([-yradpix, yradpix, yradpix, -yradpix])
    nvert = 4

    xp = xp0 * np.cos(angle_rad) - yp0 * np.sin(angle_rad) + xcpix
    yp = xp0 * np.sin(angle_rad) + yp0 * np.cos(angle_rad) + ycpix

    image = np.zeros([ngrid, ngrid], dtype=float)

    left = np.where(yp == np.min(yp))
    left = left[0][np.where(xp[left] == np.min(xp[left]))[0]]
    left = left[0]

    if left != nvert-1:
        leftnext = left + 1
    else:
        leftnext = 0

    right = left
    if right != 0:
        rightnext = right - 1
    else:
        rightnext = nvert - 1

    if int(np.round(np.min(yp))) < 0:
        imin = 0
    else:
        imin = int(np.round(np.min(yp)))

    if int(np.round(np.max(yp)))+1 >= ngrid:
        imax = ngrid
    else:
        imax = int(np.round(np.max(yp))) + 1

    for ypix in range(imin, imax):
        for ysub in range(0, mag):
            y = ypix - 0.5 + (0.5 + ysub)/mag

            if y < yp[left]:
                continue
            if y > np.max(yp):
                break

            if y >= yp[leftnext]:
                left = leftnext
                if left != nvert-1:
                    leftnext = left + 1
                else:
                    leftnext = 0

            if y >= yp[rightnext]:
                right = rightnext
                if right != 0:
                    rightnext = right - 1
                else:
                    rightnext = nvert - 1

            leftdy = yp[leftnext] - yp[left]
            if leftdy != 0:
                leftdx = xp[leftnext] - xp[left]
                xleft = leftdx/leftdy * (y-yp[left]) + xp[left]
            else:
                xleft = xp[left]

            rightdy = yp[rightnext] - yp[right]
            if rightdy != 0:
                rightdx = xp[rightnext] - xp[right]
                xright = rightdx/rightdy * (y - yp[right]) + xp[right]
            else:
                xright = xp[right]

            xleftpix = int(np.round(xleft))
            xrightpix = int(np.round(xright))

            if xleftpix != xrightpix:
                if (xleftpix >= 0 and xleftpix < ngrid):
                    image[ypix, xleftpix] = image[ypix, xleftpix] +\
                        mag * ((xleftpix + 0.5) - xleft)
                if (xrightpix >= 0 and xrightpix < ngrid):
                    image[ypix, xrightpix] = image[ypix, xrightpix] +\
                        mag * (xright - (xrightpix - 0.5))
                if (xrightpix - xleftpix > 1 and xleftpix + 1 < ngrid and xrightpix > 0):
                    if xleftpix+1 < 0:
                        imin = 0
                    else:
                        imin = xleftpix+1

                    if xrightpix > ngrid:
                        imax = ngrid
                    else:
                        imax = xrightpix
                    image[ypix, imin:imax] = image[ypix, imin:imax] + mag
            else:
                if xleftpix >= 0 and xleftpix < ngrid:
                    image[ypix, xleftpix] = image[ypix, xleftpix] +\
                        mag * (xright - xleft)

    image = image / float(mag)**2
    image = pad_crop(image, (ny, nx))
    if isDark:
        image = 1.0 - image

    return image
