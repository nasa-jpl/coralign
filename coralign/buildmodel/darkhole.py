"""Functions to generate a software mask of all the dark hole pixels."""
import numpy as np

import coralign.util.check as check
from coralign.util.loadyaml import loadyaml


def gen_coords(arrayShape, xOffset, yOffset):
    """
    Generate 2-D arrays of coordinates in Cartesian and polar.

    Parameters
    ----------
    arrayShape : array_like
        Shape of the 2-D array for which to compute coordinates.
    xOffset : float
        x-offset of coordinate system from center pixel. Units of pixels.
    yOffset : float
        y-offset of coordinate system from center pixel. Units of pixels.

    Returns
    -------
    X : array_like
        2-D array of x-coordinates.
    Y : array_like
        2-D array of y-coordinates.
    R : array_like
        2-D array of radial coordinates.
    THETA : array_like
        2-D array of angular coordinates.

    """
    check.oneD_array(arrayShape, 'arrayShape', TypeError)
    if len(arrayShape) != 2:
        raise TypeError('arrayShape must contain exactly two values')
    check.positive_scalar_integer(arrayShape[0], 'arrayShape[0]', TypeError)
    check.positive_scalar_integer(arrayShape[1], 'arrayShape[1]', TypeError)
    check.real_scalar(xOffset, 'xOffset', TypeError)
    check.real_scalar(yOffset, 'yOffset', TypeError)

    nx = arrayShape[1]
    if nx % 2 == 0:
        x = np.linspace(-nx/2, nx/2 - 1, nx) - xOffset
    else:
        x = np.linspace(-(nx-1)/2, (nx-1)/2, nx) - xOffset

    ny = arrayShape[0]
    if ny % 2 == 0:
        y = np.linspace(-ny/2, ny/2 - 1, ny) - yOffset
    else:
        y = np.linspace(-(ny-1)/2, (ny-1)/2, ny) - yOffset

    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X*X + Y*Y)
    THETA = np.arctan2(Y, X)

    return X, Y, R, THETA


def gen_rectangle(arrayShape, xOffset, yOffset, width, height):
    """
    Generate a rectangle in a 2-D boolean array.

    Parameters
    ----------
    arrayShape : array_like
        Shape of the 2-D array for which to compute coordinates.
    xOffset : float
        x-offset of shape's center from center pixel. Units of pixels.
    yOffset : float
        y-offset of shape's center from center pixel. Units of pixels.
    width : float
        Width of the rectangle. Units of pixels.
    height : float
        Height of the rectangle. Units of pixels.

    Returns
    -------
    rectangle : array_like
        2-D boolean array containing the specified rectangle.

    """
    check.oneD_array(arrayShape, 'arrayShape', TypeError)
    if len(arrayShape) != 2:
        raise TypeError('arrayShape must contain exactly two values')
    check.positive_scalar_integer(arrayShape[0], 'arrayShape[0]', TypeError)
    check.positive_scalar_integer(arrayShape[1], 'arrayShape[1]', TypeError)
    check.real_scalar(xOffset, 'xOffset', TypeError)
    check.real_scalar(yOffset, 'yOffset', TypeError)
    check.real_positive_scalar(width, 'width', TypeError)
    check.real_positive_scalar(height, 'height', TypeError)

    X, Y, _, _ = gen_coords(arrayShape, xOffset, yOffset)
    rectangle = (np.abs(X) <= width/2) & (np.abs(Y) <= height/2)

    return rectangle


def gen_annulus(arrayShape, xOffset, yOffset, radiusInner, radiusOuter):
    """
    Generate an annulus in a 2-D boolean array.

    Parameters
    ----------
    arrayShape : array_like
        Shape of the 2-D array for which to compute coordinates.
    xOffset : float
        x-offset of shape's center from center pixel. Units of pixels.
    yOffset : float
        y-offset of shape's center from center pixel. Units of pixels.
    radiusInner : float
        Inner radius of the annulus. Units of pixels.
    radiusOuter : float
        Outer radius of the annulus. Units of pixels.

    Returns
    -------
    annulus : array_like
        2-D boolean array containing the specified annulus.

    """
    check.oneD_array(arrayShape, 'arrayShape', TypeError)
    if len(arrayShape) != 2:
        raise TypeError('arrayShape must contain exactly two values')
    check.positive_scalar_integer(arrayShape[0], 'arrayShape[0]', TypeError)
    check.positive_scalar_integer(arrayShape[1], 'arrayShape[1]', TypeError)
    check.real_scalar(xOffset, 'xOffset', TypeError)
    check.real_scalar(yOffset, 'yOffset', TypeError)
    check.real_nonnegative_scalar(radiusInner, 'radiusInner', TypeError)
    check.real_nonnegative_scalar(radiusOuter, 'radiusOuter', TypeError)
    if radiusInner > radiusOuter:
        raise ValueError('radiusInner cannot be greater than radiusOuter')

    _, _, R, _ = gen_coords(arrayShape, xOffset, yOffset)
    annulus = (R >= radiusInner) & (R <= radiusOuter)

    return annulus


def gen_annular_sector(arrayShape, xOffset, yOffset, radiusInner, radiusOuter,
                       openingAngle, clocking):
    """
    Generate an annular sector in a 2-D boolean array.

    Parameters
    ----------
    arrayShape : array_like
        Shape of the 2-D array for which to compute coordinates.
    xOffset : float
        x-offset of annulus's center from center pixel. Units of pixels.
    yOffset : float
        y-offset of annulus's center from center pixel. Units of pixels.
    radiusInner : float
        Inner radius of the annulus. Units of pixels.
    radiusOuter : float
        Outer radius of the annulus. Units of pixels.
    openingAngle : float
        Opening angle of the sector. Units of degrees.
    clocking : float
        Counterclockwise rotation of the shape from the x-axis.
        Units of degrees.

    Returns
    -------
    sector : array_like
        2-D boolean array containing the specified annular sector.

    """
    check.oneD_array(arrayShape, 'arrayShape', TypeError)
    if len(arrayShape) != 2:
        raise TypeError('arrayShape must contain exactly two values')
    check.positive_scalar_integer(arrayShape[0], 'arrayShape[0]', TypeError)
    check.positive_scalar_integer(arrayShape[1], 'arrayShape[1]', TypeError)
    check.real_scalar(xOffset, 'xOffset', TypeError)
    check.real_scalar(yOffset, 'yOffset', TypeError)
    check.real_nonnegative_scalar(radiusInner, 'radiusInner', TypeError)
    check.real_nonnegative_scalar(radiusOuter, 'radiusOuter', TypeError)
    if radiusInner > radiusOuter:
        raise ValueError('radiusInner cannot be greater than radiusOuter')
    check.real_positive_scalar(openingAngle, 'openingAngle', TypeError)
    check.real_scalar(clocking, 'clocking', TypeError)

    _, _, R, THETA = gen_coords(arrayShape, xOffset, yOffset)
    THETAROT = np.angle(np.exp(1j*(THETA - np.radians(clocking))))
    sector = ((R >= radiusInner) &
              (R <= radiusOuter) &
              (THETAROT >= -np.radians(openingAngle)/2) &
              (THETAROT <= np.radians(openingAngle)/2)
              )

    return sector


def gen_bowtie(arrayShape, xOffset, yOffset, radiusInner, radiusOuter,
               openingAngle, clocking):
    """
    Generate a bowtie (two annular sectors) in a 2-D boolean array.

    Parameters
    ----------
    arrayShape : array_like
        Shape of the 2-D array for which to compute coordinates.
    xOffset : float
        x-offset of annulus's center from center pixel. Units of pixels.
    yOffset : float
        y-offset of annulus's center from center pixel. Units of pixels.
    radiusInner : float
        Inner radius of the annulus. Units of pixels.
    radiusOuter : float
        Outer radius of the annulus. Units of pixels.
    openingAngle : float
        Opening angle of the sector. Units of degrees.
    clocking : float
        Counterclockwise rotation of the shape from the x-axis.
        Units of degrees.

    Returns
    -------
    bowtie : array_like
        2-D boolean array containing the specified bowtie.

    """
    check.oneD_array(arrayShape, 'arrayShape', TypeError)
    if len(arrayShape) != 2:
        raise TypeError('arrayShape must contain exactly two values')
    check.positive_scalar_integer(arrayShape[0], 'arrayShape[0]', TypeError)
    check.positive_scalar_integer(arrayShape[1], 'arrayShape[1]', TypeError)
    check.real_scalar(xOffset, 'xOffset', TypeError)
    check.real_scalar(yOffset, 'yOffset', TypeError)
    check.real_nonnegative_scalar(radiusInner, 'radiusInner', TypeError)
    check.real_nonnegative_scalar(radiusOuter, 'radiusOuter', TypeError)
    if radiusInner > radiusOuter:
        raise ValueError('radiusInner cannot be greater than radiusOuter')
    check.real_positive_scalar(openingAngle, 'openingAngle', TypeError)
    check.real_scalar(clocking, 'clocking', TypeError)

    bowtie = (gen_annular_sector(arrayShape, xOffset, yOffset, radiusInner,
                                 radiusOuter, openingAngle, clocking) |
              gen_annular_sector(arrayShape, xOffset, yOffset, radiusInner,
                                 radiusOuter, openingAngle, clocking+180)
              )

    return bowtie


def gen_dark_hole_from_yaml(fnSpecs):
    """
    Generate a boolean software mask for a dark hole from specs in a YAML file.

    Parameters
    ----------
    fnSpecs : str
        Name of the YAML file with all the parameters needed to define
        one or more dark hole shapes that will be combined by OR'ing.

    Returns
    -------
    mask : array_like
        2-D boolean array containing the combination of all specified shapes.

    """
    allowedShapes = ('rectangle', 'annulus', 'annular_sector', 'bowtie')

    inputs = loadyaml(fnSpecs, custom_exception=IOError)
    nRows = inputs["nRows"]
    nCols = inputs["nCols"]
    check.positive_scalar_integer(nRows, 'nRows', TypeError)
    check.positive_scalar_integer(nCols, 'nCols', TypeError)

    # Combine shapes in a loop
    mask = np.zeros((nRows, nCols))
    for _, nestedDict in inputs["shapes"].items():

        xOffset = nestedDict["xOffset"]
        yOffset = nestedDict["yOffset"]

        shapeName = nestedDict["shape"]
        if shapeName not in allowedShapes:
            raise ValueError(('Invalid name of shape. Must be one of '
                              '{}'.format(allowedShapes)))

        if shapeName == 'rectangle':
            mask = np.logical_or(
                gen_rectangle((nRows, nCols),
                              xOffset,
                              yOffset,
                              nestedDict["width"],
                              nestedDict["height"],
                              ),
                mask)
        elif shapeName == 'annulus':
            mask = np.logical_or(
                gen_annulus((nRows, nCols),
                            xOffset,
                            yOffset,
                            nestedDict["radiusInner"],
                            nestedDict["radiusOuter"],
                            ),
                mask)
        elif shapeName == 'annular_sector':
            mask = np.logical_or(
                gen_annular_sector((nRows, nCols),
                                   xOffset,
                                   yOffset,
                                   nestedDict["radiusInner"],
                                   nestedDict["radiusOuter"],
                                   nestedDict["openingAngle"],
                                   nestedDict["clocking"],
                                   ),
                mask)
        elif shapeName == 'bowtie':
            mask = np.logical_or(
                gen_bowtie((nRows, nCols),
                           xOffset,
                           yOffset,
                           nestedDict["radiusInner"],
                           nestedDict["radiusOuter"],
                           nestedDict["openingAngle"],
                           nestedDict["clocking"],
                           ),
                mask)

    return mask
