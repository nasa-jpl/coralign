"""
Set of functions to add fixed shapes on top of existing DM patterns
"""

import numpy as np

from coralign.util import check
from coralign.util import nollzernikes

def probe(nact, dact, xcenter, ycenter, clock, height, ximin, ximax,
          etamin, etamax, phase):
    """
    Create a sinc*sinc*sin probe following Give'on et al. 2011

    Nominal DM centration with even-numbered DM counts places the center of the
     DM (and thus the center of the probe pattern) at the gap between the two
     central actuators in both axes.  The centers of the adjacent actuators are
     at (+/- 0.5 act, +/- 0.5 act).

    The probe pattern is normalized such that the Fourier transform of the
    probe pattern creates two rectangles, each with amplitude "height"

    When the sinc pattern is both moved and clocked, the clocking will occur
     about the shifted pattern center, so the pattern remains centered at the
     same point as an unclocked case.

    If building a cosine probe (phase = 90), it is strongly recommended to use
    ximin=0.  This will make the probe area a giant rectangle without any
    low-amplitude null regions.

    Arguments:
     nact: number of actuators along one side of the DM (assumes square DM)
     dact: diameter of pupil, in actuators
     xcenter: number of actuators to move the center of the DM pattern along
      the positive x-axis, as seen from the camera.  Negative and fractional
      inputs are acceptable.
     ycenter: number of actuators to move the center of the DM pattern along
      the positive y-axis, as seen from the camera.  Negative and fractional
      inputs are acceptable.
     clock: angle in degrees to rotate the DM pattern about its center.  This
      rotation is clockwise and is applied before the xcenter/ycenter
      shifts are applied.
     height: height of sinc peak, in meters; actual shape may not reach this
      value depending on centration
     ximin: min lambda/D along the x-axis in the focal plane for the probe
      rectangle, must be less than ximax
     ximax: max lambda/D along the x-axis in the focal plane for the probe
      rectangle
     etamin: min lambda/D along the y-axis in the focal plane for the probe
      rectangle, must be less than etamax
     etamax: max lambda/D along the y-axis in the focal plane for the probe
      rectangle
     phase: phase angle in degrees to shift this particular probe; at phase = 0
      the modulation will be a sine and at phase = 90 it will be a cosine.

    Returns
     a nact x nact 2D array of heights in meters for each actuator

    """

    # Check inputs
    check.positive_scalar_integer(nact, 'nact', TypeError)
    check.real_positive_scalar(dact, 'dact', TypeError)
    check.real_scalar(xcenter, 'xcenter', TypeError)
    check.real_scalar(ycenter, 'ycenter', TypeError)
    check.real_scalar(clock, 'clock', TypeError)
    check.real_positive_scalar(height, 'height', TypeError)
    check.real_scalar(ximin, 'ximin', TypeError)
    check.real_scalar(ximax, 'ximax', TypeError)
    check.real_scalar(etamin, 'etamin', TypeError)
    check.real_scalar(etamax, 'etamax', TypeError)
    check.real_scalar(phase, 'phase', TypeError)
    if ximin >= ximax:
        raise ValueError('ximin must be strictly less than ximax')
    if etamin >= etamax:
        raise ValueError('etamin must be strictly less than etamax')

    # Set up grids with translation and rotation
    xx0, yy0 = np.meshgrid(np.arange(nact)-(nact-1.)/2.-xcenter,
                           np.arange(nact)-(nact-1.)/2.-ycenter)
    rclock = clock*np.pi/180.
    xx = xx0*np.cos(rclock) - yy0*np.sin(rclock)
    yy = xx0*np.sin(rclock) + yy0*np.cos(rclock)

    # Build probe shapes, using dact instead of nact so we get the right
    # lambda/D when the pupil undersizes the DM
    wx = dact/float(ximax-ximin)
    wy = dact/float(etamax-etamin)
    fx = (ximin+ximax)/2.
    fy = (etamin+etamax)/2.

    ddm = height*np.sinc(xx/wx)*np.sinc(yy/wy)
    # this normalizes FFT to 1 across one rectangle, leaving aside "height"
    ddm /= wx*wy
    # 2x because the sine turns one amp 1 rectangle into two amp 0.5 rects
    ddm *= 2*np.sin(2.*np.pi*(xx*fx + yy*fy)/dact + phase*np.pi/180.)
    return ddm


def sine(nact, dact, amplitude, phase, freqx, freqy):
    """
    Create a sinusoid with a given spatial frequency

    freqx creates speckles located at freqx lam/D, similarly for freqy.
     Total radial distance is sqrt(freqx**2 + freqy**2).

    Arguments:
     nact: number of actuators along one side of the DM (assumes square DM)
     dact: diameter of pupil, in actuators
     amplitude: amplitude of sinusoid, in meters. Peak-to-valley range will be
      <= 2*amplitude (only == if you got the sampling just right)
     phase: phase angle in degrees to shift the sinusoid phase; implemented at
      +phase, so phase = 0 will give a sine and phase = 90 will give a cosine.
     freqx: Component of spatial frequency in x direction.
     freqy: Component of spatial frequency in y direction.

    Returns
     a nact x nact 2D array of heights in meters for each actuator

    """

    # Check inputs
    check.positive_scalar_integer(nact, 'nact', TypeError)
    check.real_positive_scalar(dact, 'dact', TypeError)
    check.real_positive_scalar(amplitude, 'amplitude', TypeError)
    check.real_scalar(phase, 'phase', TypeError)
    check.real_scalar(freqx, 'freqx', TypeError)
    check.real_scalar(freqy, 'freqy', TypeError)

    # Set up grids
    xx, yy = np.meshgrid(np.arange(nact)-(nact-1.)/2.,
                           np.arange(nact)-(nact-1.)/2.)
    # Build sine argument, using dact instead of nact so we get the right
    # lambda/D when the pupil undersizes the DM
    sphase = 2.*np.pi*(xx*freqx + yy*freqy)/dact
    ddm = amplitude*np.sin(sphase + phase*np.pi/180.)
    return ddm


def grid(nact, xspace, yspace, xoffset, yoffset, pokesize):
    """
    Create a DM pattern where one out of every X actuators is raised or lowered
     by a constant amplitude.

    These are used to do aligment/registration/characterization in pupil plane
     in conjunction with phase retrieval.  xspace/offset are applied along
     columns, and yspace/offset are applied along rows.

    Example of poke pattern for xspace = yspace = 4, xoffset = yoffset = 0,
     with poked actuators as 'x' and unpoked as '.'.

     ......
     x...x.
     ......
     ......
     ......
    0x...x.
     0

    Example for xoffset = 0, yoffset = 1, xspace = 2, yspace = 3:

     x.x.x.x.
     ........
     ........
     x.x.x.x.
     ........
     ........
     x.x.x.x.
    0........
     0

    Arguments:
     nact: number of actuators along one side of the DM (assumes square DM).
      If the grid number and spacing does not align perfectly with the number
      of actuators, it will be truncated at the high end of the array rather
      than the side near zero.
     xspace: spacing in integer number of actuators for pokes along the x-axis.
      If xspace=5, you might see the 0th, 5th, 10th, etc. actuator poked.  Must
      be greater than zero and <= than nact.
     yspace: spacing in integer number of actuators for pokes along the y-axis.
      Must be greater than zero and <= than nact.
     xoffset: integer number of actuators to offset a grid from 0 along x.
      Given xspace=3, xoffset=0 would poke 0th, 3rd, 6th, etc. while
      xoffset=2 would poke 2nd, 5th, 8th, etc.  xoffset must be < xspace and
      >= 0.
     yoffset: integer number of actuators to offset a grid from 0 along y.
      yoffset must be < yspace and >= 0.
     pokesize: amplitude of grid pokes, in meters.  All poked actuators will
      be the same amplitude and all unpoked actuators will be 0.  May be
      negative or positive.

    Returns
     a nact x nact 2D array of heights in meters for each actuator

    """

    # Check inputs
    check.positive_scalar_integer(nact, 'nact', TypeError)
    check.positive_scalar_integer(xspace, 'xspace', TypeError)
    check.positive_scalar_integer(yspace, 'yspace', TypeError)
    check.nonnegative_scalar_integer(xoffset, 'xoffset', TypeError)
    check.nonnegative_scalar_integer(yoffset, 'yoffset', TypeError)
    check.real_scalar(pokesize, 'pokesize', TypeError)

    if xoffset >= xspace:
        raise ValueError('xoffset must be < xspace')
    if yoffset >= yspace:
        raise ValueError('yoffset must be < yspace')
    if xspace > nact:
        raise ValueError('xspace must be < nact')
    if yspace > nact:
        raise ValueError('yspace must be < nact')

    # Set up grids
    ddm = np.zeros((nact, nact))
    ddm[yoffset::yspace, xoffset::xspace] += pokesize
    return ddm


def zernike(nact, dact, zernsize, order):
    """
    Create a Zernike polynomial overlaid on a DM

    Future implementation note: it would not be hard to extend this to overlay
     several Zernikes, but I am not implementing this at the moment as the use
     case is not there.   Consider updating this function if for some reason we
     do end up putting on several at once.

    Arguments:
     nact: number of actuators along one side of the DM (assumes square DM)
     dact: diameter of pupil, in actuators
     zernsize: magnitude of Zernike in meters rms.  Real-valued scalar, may be
      negative.
     order: Noll order of requested Zernike.  See description in
      nollzernikes.py for more information about the ordering.  This will be a
      positive integer.

    Returns
     a nact x nact 2D array of heights in meters for each actuator

    """

    # Check inputs
    check.positive_scalar_integer(nact, 'nact', TypeError)
    check.real_positive_scalar(dact, 'dact', TypeError)
    check.real_scalar(zernsize, 'zernsize', TypeError)
    check.positive_scalar_integer(order, 'order', TypeError)

    # Set up grids
    xx, yy = np.meshgrid(np.arange(nact)-(nact-1.)/2.,
                           np.arange(nact)-(nact-1.)/2.)
    ddm = zernsize*nollzernikes.xyzern(xx, yy, dact/2., [order])
    return ddm[0] # xyzern returns 3D array
