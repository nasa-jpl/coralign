"""Unit tests for the MASKGEN module."""
import unittest
import os
from math import isclose
import numpy as np

from coralign.util import shapes
from coralign.util.pad_crop import pad_crop as inin
from coralign.util.math import ceil_odd
from coralign.maskgen.maskgen import (
    rotate_shift_downsample_amplitude_mask, gen_hlc_occulter,
)


class TestAmplitudeMaskCases(unittest.TestCase):
    """Test the algorithm for amplitude masks as it will be used."""

    def setUp(self):
        """Initialize variables used in all the functional tests."""
        self.localpath = os.path.dirname(os.path.abspath(__file__))

        self.mag = 4/25.

        res = 25
        nSeg = 2
        rInPix = 2.6*res
        rOutPix = 9.4*res
        nx = 500
        ny = 500
        xOffset = 0
        yOffset = 0
        angOpen = 65
        angRot = 0
        self.maskIn = shapes.annular_segments(
            nSeg, rInPix, rOutPix, nx, ny, xOffset, yOffset,
            angOpen, angRot, nSubpixels=11, dTheta=5)

        self.xOffset = -15
        self.yOffset = 50
        self.padFac = 1.1

    def test_area_conservation(self):
        """Test downsampling by comparing the area before and after."""
        rotDeg = 90
        maskOut = rotate_shift_downsample_amplitude_mask(
            self.maskIn, rotDeg, self.mag, self.xOffset,
            self.yOffset, self.padFac)
        ratioArea = np.sum(maskOut)/np.sum(self.maskIn)
        self.assertTrue(isclose(self.mag**2, ratioArea, rel_tol=0.001))

    def test_rotation_translation(self):
        """Test rotation and translation using np.rot90 and np.roll."""
        rotDeg = 90
        maskOut = rotate_shift_downsample_amplitude_mask(
            self.maskIn, rotDeg, self.mag, self.xOffset,
            self.yOffset, self.padFac)
        maskJustDemag = rotate_shift_downsample_amplitude_mask(
            self.maskIn, 0, self.mag, 0, 0, self.padFac)
        maskJustDemag = inin(maskJustDemag, (maskOut.shape))
        maskRotRoll = np.roll(np.rot90(maskJustDemag[1::, 1::], 1),
                              (self.yOffset, self.xOffset), axis=(0, 1))
        maskRotRoll = inin(maskRotRoll, (maskOut.shape))

        maxAbsDiff = np.max(np.abs(maskOut - maskRotRoll))
        abs_tol = 100*np.finfo(float).eps
        self.assertTrue(maxAbsDiff < abs_tol)

    def test_flipx_rotation(self):
        """Test flipx and rotation at the same time."""
        maskExpected = np.zeros((5, 5))
        maskExpected[:, 1] = 1
        maskExpected[4, 1:5] = 1
        maskExpected[2, 1:4] = 1
        maskExpected = np.fliplr(maskExpected)
        maskExpected = np.rot90(maskExpected, -1)
        maskExpected = inin(maskExpected, (15, 15))

        maskIn = np.zeros((15, 15))
        maskIn[:, 3:6] = 1
        maskIn[12:15, 4:15] = 1
        maskIn[6:9, 4:12] = 1

        rotDeg = 90
        mag = 1/3
        xOffset = 0
        yOffset = 0
        padFac = 3
        flipx = True
        maskOut = rotate_shift_downsample_amplitude_mask(
            maskIn, rotDeg, mag, xOffset, yOffset, padFac, flipx=flipx)
        maskOut = inin(maskOut, maskExpected.shape)

        maxAbsDiff = np.max(np.abs(maskOut - maskExpected))
        abs_tol = 10*np.finfo(float).eps
        self.assertTrue(maxAbsDiff < abs_tol)

    def test_padFac_default(self):
        """Test that the default padding factor gives the expected result."""
        maskIn = np.ones((9, 9))
        maskIn = inin(maskIn, (11, 11))
        outShape = (5, 5)

        rotDeg = 0
        mag = 1/3
        xOffset = 0
        yOffset = 0

        maskOutDefault = rotate_shift_downsample_amplitude_mask(
            maskIn, rotDeg, mag, xOffset, yOffset)
        maskOutDefault = inin(maskOutDefault, outShape)

        padFac = 1.2
        maskOutB = rotate_shift_downsample_amplitude_mask(
            maskIn, rotDeg, mag, xOffset, yOffset, padFac=padFac)
        maskOutB = inin(maskOutB, outShape)

        padFac = 4.0
        maskOutC = rotate_shift_downsample_amplitude_mask(
            maskIn, rotDeg, mag, xOffset, yOffset, padFac=padFac)
        maskOutC = inin(maskOutC, outShape)

        abs_tol = 10*np.finfo(float).eps

        # Default value of padFac gives same answer as setting padFac=1.2
        maxAbsDiff1pt2 = np.max(np.abs(maskOutB - maskOutDefault))
        self.assertTrue(maxAbsDiff1pt2 < abs_tol)

        # Default padFac value gives same (correct) answer as a large padding
        maxAbsDiff4pt0 = np.max(np.abs(maskOutC - maskOutDefault))
        self.assertTrue(maxAbsDiff4pt0 < abs_tol)


class TestComplexMaskCases(unittest.TestCase):
    """Test the algorithm for complex-valued masks as it will be used."""

    def setUp(self):
        """Initialize variables used in all the functional tests."""
        self.localpath = os.path.dirname(os.path.abspath(__file__))

        self.fnCalibData = os.path.join(
            self.localpath, 'testdata', 'calib_data_for_hlc_fpm_gen.yaml')
        self.fnRotCalibData = os.path.join(
            self.localpath, 'testdata',
            'calib_data_for_rot90_hlc_fpm_gen.yaml')

        # Use mask_file_data_HLC_Band1.yaml for FLT.
        # Can use ut_mask_file_data_HLC_Band1.yaml for faster testing.
        self.fnOccData = os.path.join(self.localpath, 'testdata',
                                      'ut_mask_file_data_HLC_Band1.yaml')

        self.shapeOut = (111, 100)
        self.scale = False
        self.lam = 575e-9
        self.yOffset = 0
        self.xOffset = 0

    def test_scaling_with_wavelength(self):
        """Test scaling with wavelength."""
        scaleWithWavelength = False
        lam = 575e-9
        mask = gen_hlc_occulter(lam,
                                scaleWithWavelength,
                                self.shapeOut,
                                self.fnCalibData,
                                self.fnOccData,
                                data_path=self.localpath)
        lamFac = 0.95
        scaleWithWavelength = True
        maskScale = gen_hlc_occulter(lamFac*lam,
                                     scaleWithWavelength,
                                     self.shapeOut,
                                     self.fnCalibData,
                                     self.fnOccData,
                                     data_path=self.localpath)
        I0 = np.sum(np.abs(mask-1)**2)
        I1 = np.sum(np.abs(maskScale-1)**2)
        # Make sure less than 0.25% difference in expected area. Can't be more
        # precise because the complex-valued transmission changes with
        # wavelength and doesn't make the energy scaling exact.
        self.assertTrue(np.abs(I1*lamFac**2 - I0)/I0 < 0.0025)

    def test_rotation(self):
        """Test rotation by comparing to np.rot90."""
        scaleWithWavelength = False
        lam = 575e-9
        mask = gen_hlc_occulter(lam,
                                scaleWithWavelength,
                                self.shapeOut,
                                self.fnCalibData,
                                self.fnOccData,
                                data_path=self.localpath)
        maskRot90 = gen_hlc_occulter(lam,
                                     scaleWithWavelength,
                                     self.shapeOut,
                                     self.fnRotCalibData,
                                     self.fnOccData,
                                     data_path=self.localpath)

        # need to make smaller to avoid zero padding:
        n = ceil_odd(np.min(mask.shape)) - 2
        mask = inin(mask, (n, n))
        maskRot90 = inin(maskRot90, (n, n))
        sumAbsDiff = np.sum(np.abs(maskRot90 - np.rot90(mask, -1)))
        self.assertTrue(sumAbsDiff < 1e-8)

    def test_centering(self):
        """Make sure the occulters are centered on the array's center pixel."""
        scaleWithWavelength = False
        lam = 575e-9
        mask = gen_hlc_occulter(lam,
                                scaleWithWavelength,
                                self.shapeOut,
                                self.fnCalibData,
                                self.fnOccData,
                                data_path=self.localpath)
        lamFac = 0.95
        scaleWithWavelength = True
        maskScale = gen_hlc_occulter(lamFac*lam,
                                     scaleWithWavelength,
                                     self.shapeOut,
                                     self.fnCalibData,
                                     self.fnOccData,
                                     data_path=self.localpath)

        # need to make smaller to avoid zero padding:
        n0 = ceil_odd(np.min(mask.shape)) - 2
        mask = inin(mask, (n0, n0))
        sumAbsDiff = np.sum(np.abs(mask - np.fliplr(mask)))
        self.assertTrue(sumAbsDiff < 1e-3)

        n1 = ceil_odd(np.min(maskScale.shape)) - 2
        maskScale = inin(maskScale, (n1, n1))
        sumAbsDiff = np.sum(np.abs(maskScale - np.fliplr(maskScale)))
        self.assertTrue(sumAbsDiff < 1e-3)

    def test_translation(self):
        """Make sure the occulters are translated as expected."""
        scaleWithWavelength = False
        lam = 575e-9
        xOffsetTrue = 1
        yOffsetTrue = -2
        mask = gen_hlc_occulter(lam,
                                scaleWithWavelength,
                                self.shapeOut,
                                self.fnCalibData,
                                self.fnOccData,
                                data_path=self.localpath)
        maskShift = np.roll(mask, (yOffsetTrue, xOffsetTrue), axis=(0, 1))
        maskOut = gen_hlc_occulter(lam,
                                   scaleWithWavelength,
                                   self.shapeOut,
                                   self.fnCalibData,
                                   self.fnOccData,
                                   xOffset=xOffsetTrue,
                                   yOffset=yOffsetTrue,
                                   data_path=self.localpath)

        lamFac = 0.95
        scaleWithWavelength = True
        maskScale = gen_hlc_occulter(lamFac*lam,
                                     scaleWithWavelength,
                                     self.shapeOut,
                                     self.fnCalibData,
                                     self.fnOccData,
                                     data_path=self.localpath)
        maskScaleShift = np.roll(maskScale, (yOffsetTrue, xOffsetTrue),
                                 axis=(0, 1))
        maskScaleOut = gen_hlc_occulter(lamFac*lam,
                                        scaleWithWavelength,
                                        self.shapeOut,
                                        self.fnCalibData,
                                        self.fnOccData,
                                        xOffset=xOffsetTrue,
                                        yOffset=yOffsetTrue,
                                        data_path=self.localpath)

        # need to make smaller to avoid zero padding:
        n0 = ceil_odd(np.min(mask.shape)) - 4
        maskShift = inin(maskShift, (n0, n0))
        maskOut = inin(maskOut, (n0, n0))
        maxAbsDiff = np.max(np.abs(maskOut - maskShift))
        abs_tol = 1e-3
        self.assertTrue(maxAbsDiff < abs_tol)

        n1 = ceil_odd(np.min(maskScale.shape)) - 4
        maskScaleShift = inin(maskScaleShift, (n1, n1))
        maskScaleOut = inin(maskScaleOut, (n1, n1))
        maxAbsDiff = np.max(np.abs(maskScaleOut - maskScaleShift))
        abs_tol = 1e-3
        self.assertTrue(maxAbsDiff < abs_tol)

    # Bad input tests
    # Tests on file names are handled by loading functions.
    def test_bad_input_0(self):
        """Test that the correct exception is called for a bad input."""
        for badVal in (-1, 0, 1j, [5, ], np.ones((5, 2, 3)), 'string'):
            with self.assertRaises(TypeError):
                gen_hlc_occulter(badVal, self.scale, self.shapeOut,
                                 self.fnCalibData, self.fnOccData)

    def test_bad_input_1(self):
        """Test that the correct exception is called for a bad input."""
        for badVal in (-1, 0, 1j, 1.5, [5, ], np.ones((5, 2, 3)), 'string'):
            with self.assertRaises(TypeError):
                gen_hlc_occulter(self.lam, badVal, self.shapeOut,
                                 self.fnCalibData, self.fnOccData)

    def test_bad_input_2(self):
        """Test that the correct exception is called for a bad input."""
        for badVal in (-1, 0, 1j, [5, ], np.ones((5, 2, 3)), 'string'):
            with self.assertRaises(TypeError):
                gen_hlc_occulter(self.lam, self.scale, badVal,
                                 self.fnCalibData, self.fnOccData)

    def test_bad_input_5(self):
        """Test that the correct exception is called for a bad input."""
        for badVal in (1j, [5, ], np.ones((5, 2, 3)), 'string'):
            with self.assertRaises(TypeError):
                gen_hlc_occulter(self.lam, self.scale, self.shapeOut,
                                 self.fnCalibData, self.fnOccData,
                                 xOffset=badVal, yOffset=self.yOffset)

    def test_bad_input_6(self):
        """Test that the correct exception is called for a bad input."""
        for badVal in (1j, [5, ], np.ones((5, 2, 3)), 'string'):
            with self.assertRaises(TypeError):
                gen_hlc_occulter(self.lam, self.scale, self.shapeOut,
                                 self.fnCalibData, self.fnOccData,
                                 xOffset=self.xOffset, yOffset=badVal)


class TestMaskgenInputFailure(unittest.TestCase):
    """Test suite for valid function inputs."""

    def test_rotate_shift_downsample_amplitude_mask_inputs(self):
        """Test incorrect inputs of rotate_shift_downsample_amplitude_mask."""

        # localpath = os.path.dirname(os.path.abspath(__file__))

        # # Working inputs
        mag = 0.4
        rotDeg = 10.5  # degrees CCW
        xOffset = 52.4
        yOffset = -25
        padFac = 1.2
        nArray = 500
        radius = 200
        maskIn = shapes.circle(nArray, nArray, radius, 0, 0)

        # First verify that function runs as expected.
        rotate_shift_downsample_amplitude_mask(maskIn, rotDeg, mag,
                                               xOffset, yOffset, padFac)

        for maskInBad in (1j, np.ones((5, )), np.ones((5, 2, 3)), 'string'):
            with self.assertRaises(TypeError):
                rotate_shift_downsample_amplitude_mask(
                    maskInBad, rotDeg, mag, xOffset, yOffset, padFac)
        for rotDegBad in (1j, np.ones((5, )), 'string'):
            with self.assertRaises(TypeError):
                rotate_shift_downsample_amplitude_mask(
                    maskIn, rotDegBad, mag, xOffset, yOffset, padFac)
        for magBad in (-0.1, 0, 1.00001, 1j, np.ones((5, )), 'string'):
            with self.assertRaises(TypeError):
                rotate_shift_downsample_amplitude_mask(
                    maskIn, rotDeg, magBad, xOffset, yOffset, padFac)
        for xOffsetBad in (1j, np.ones((5, )), 'string'):
            with self.assertRaises(TypeError):
                rotate_shift_downsample_amplitude_mask(
                    maskIn, rotDeg, mag, xOffsetBad, yOffset, padFac)
        for yOffsetBad in (1j, np.ones((5, )), 'string'):
            with self.assertRaises(TypeError):
                rotate_shift_downsample_amplitude_mask(
                    maskIn, rotDeg, mag, xOffset, yOffsetBad, padFac)
        for padFacBad in (-0.1, 0, 0.999, 1j, np.ones((5, )), 'string'):
            with self.assertRaises(TypeError):
                rotate_shift_downsample_amplitude_mask(
                    maskIn, rotDeg, mag, xOffset, yOffset, padFacBad)
        for flipxBad in (-0.1, 0, 1, 1j, np.ones((5, )), 'string'):
            with self.assertRaises(TypeError):
                rotate_shift_downsample_amplitude_mask(
                    maskIn, rotDeg, mag, xOffset, yOffset, padFac,
                    flipx=flipxBad)


if __name__ == '__main__':
    unittest.main()
