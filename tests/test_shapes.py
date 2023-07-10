"""Test suite for shapes.py."""
import unittest
from math import isclose
import numpy as np

from coralign.util import shapes
from coralign.util.pad_crop import pad_crop


class TestCircle(unittest.TestCase):
    """Unit tests for shapes.circle()."""

    def setUp(self):
        self.nx = 51
        self.ny = 60
        self.radius = 5.0
        self.xOffset = 10
        self.yOffset = -8.2
        self.nSubpixels = 101

    def test_input_failures(self):
        with self.assertRaises(TypeError):
            shapes.circle(2.5, self.ny, self.radius, self.xOffset,
                          self.yOffset, nSubpixels=self.nSubpixels)
        with self.assertRaises(TypeError):
            shapes.circle(self.nx, 2.5, self.radius, self.xOffset,
                          self.yOffset, nSubpixels=self.nSubpixels)
        with self.assertRaises(TypeError):
            shapes.circle(self.nx, self.ny, -1, self.xOffset,
                          self.yOffset, nSubpixels=self.nSubpixels)
        with self.assertRaises(TypeError):
            shapes.circle(self.nx, self.ny, self.radius, 1j,
                          self.yOffset, nSubpixels=self.nSubpixels)
        with self.assertRaises(TypeError):
            shapes.circle(self.nx, self.ny, self.radius, self.xOffset,
                          [2, ], nSubpixels=self.nSubpixels)
        with self.assertRaises(TypeError):
            shapes.circle(self.nx, self.ny, self.radius, self.xOffset,
                          self.yOffset, nSubpixels=2.5)

    def test_area(self):
        """Test that circle area matches analytical value."""
        out = shapes.circle(self.nx, self.ny, self.radius,
                            self.xOffset, self.yOffset,
                            nSubpixels=self.nSubpixels)

        roiSum = np.sum(out)
        expectedSum = np.pi*self.radius**2
        self.assertTrue(isclose(roiSum, expectedSum, rel_tol=1e-3),
                        msg='Area of circle incorrect')

    def test_shift(self):
        """Test that circle shifts as expected."""
        xOffset = -10
        yOffset = 5
        maskCentered = shapes.circle(self.nx, self.ny, self.radius, 0, 0)
        maskShifted = shapes.circle(self.nx, self.ny, self.radius,
                                    xOffset, yOffset)
        maskRecentered = np.roll(maskShifted, (-yOffset, -xOffset),
                                 axis=(0, 1))
        diffSum = np.sum(np.abs(maskCentered - maskRecentered))
        self.assertTrue(isclose(diffSum, 0.),
                        msg='Shear incorrectly applied.')

    def test_outside(self):
        """Test that circle isn't in the array."""
        xOffset = 2*self.nx
        yOffset = -2*self.ny
        out = shapes.circle(self.nx, self.ny, self.radius,
                            xOffset, yOffset)
        self.assertTrue(isclose(np.sum(out), 0.),
                        msg='Circle should be outside array.')

    def test_centering(self):
        """Test that center doesn't shift for different array sizes."""
        out0 = shapes.circle(self.nx, self.ny, self.radius,
                             self.xOffset, self.yOffset,
                             nSubpixels=self.nSubpixels)
        out1 = shapes.circle(self.nx+1, self.ny+1, self.radius,
                             self.xOffset, self.yOffset,
                             nSubpixels=self.nSubpixels)
        out1 = pad_crop(out1, out0.shape)

        self.assertTrue(np.allclose(out0, out1, rtol=1e-3),
                        msg='Centering changed with array size.')


class TestEllipse(unittest.TestCase):
    """Unit tests for shapes.ellipse()."""

    def setUp(self):
        self.nx = 51
        self.ny = 60
        self.rx = 5.0
        self.ry = 6.5
        self.rot = 0.0
        self.xOffset = 10
        self.yOffset = -8.2
        self.nSubpixels = 101

    def test_input_failures(self):
        with self.assertRaises(TypeError):
            shapes.circle(2.5, self.ny, self.rx, self.ry, self.rot,
                          self.xOffset, self.yOffset,
                          nSubpixels=self.nSubpixels)
        with self.assertRaises(TypeError):
            shapes.circle(self.nx, 2.5, self.rx, self.ry, self.rot,
                          self.xOffset, self.yOffset,
                          nSubpixels=self.nSubpixels)
        with self.assertRaises(TypeError):
            shapes.circle(self.nx, self.ny, -1, self.ry, self.rot,
                          self.xOffset, self.yOffset,
                          nSubpixels=self.nSubpixels)
        with self.assertRaises(TypeError):
            shapes.circle(self.nx, self.ny, self.rx, 1j, self.rot,
                          self.xOffset, self.yOffset,
                          nSubpixels=self.nSubpixels)
        with self.assertRaises(TypeError):
            shapes.circle(self.nx, self.ny, self.rx, self.ry, [1, ],
                          self.xOffset, self.yOffset,
                          nSubpixels=self.nSubpixels)
        with self.assertRaises(TypeError):
            shapes.circle(self.nx, self.ny, self.rx, self.ry, self.rot,
                          1j, self.yOffset, nSubpixels=self.nSubpixels)
        with self.assertRaises(TypeError):
            shapes.circle(self.nx, self.ny, self.rx, self.ry, self.rot,
                          self.xOffset, [2, ], nSubpixels=self.nSubpixels)
        with self.assertRaises(TypeError):
            shapes.circle(self.nx, self.ny, self.rx, self.ry, self.rot,
                          self.xOffset, self.yOffset, nSubpixels=10.2)

    def test_area(self):
        """Test that ellipse area matches analytical value."""
        out = shapes.ellipse(self.nx, self.ny, self.rx, self.ry, self.rot,
                             self.xOffset, self.yOffset,
                             nSubpixels=self.nSubpixels)

        shapeSum = np.sum(out)
        expectedSum = np.pi*self.rx*self.ry
        self.assertTrue(isclose(shapeSum, expectedSum, rel_tol=1e-3),
                        msg='Area of ellipse incorrect')

    def test_shift(self):
        """Test that ellipse shifts as expected."""
        xOffset = -10
        yOffset = 5
        maskCentered = shapes.ellipse(self.nx, self.ny, self.rx, self.ry,
                                      self.rot, 0, 0)
        maskShifted = shapes.ellipse(self.nx, self.ny, self.rx, self.ry,
                                     self.rot, xOffset, yOffset)
        maskRecentered = np.roll(maskShifted, (-yOffset, -xOffset),
                                 axis=(0, 1))
        diffSum = np.sum(np.abs(maskCentered - maskRecentered))
        self.assertTrue(isclose(diffSum, 0.),
                        msg='Shear incorrectly applied.')

    def test_rotation(self):
        """Test that ellipse rotates as expected."""
        mask0 = shapes.ellipse(51, 51, self.rx, self.ry, 0, 0, 0)
        maskRot = shapes.ellipse(51, 51, self.rx, self.ry, 90, 0, 0)
        maskDerot = np.rot90(maskRot, 1)
        diffSum = np.sum(np.abs(mask0 - maskDerot))

        self.assertTrue(isclose(diffSum, 0.),
                        msg='Rotation incorrect.')

    def test_outside(self):
        """Test that the ellipse isn't in the array."""
        xOffset = 2*self.nx
        yOffset = -2*self.ny
        out = shapes.ellipse(self.nx, self.ny, self.rx, self.ry, self.rot,
                             xOffset, yOffset)
        self.assertTrue(isclose(np.sum(out), 0.),
                        msg='Ellipse should be outside array.')

    def test_centering(self):
        """Test that center doesn't shift for different array sizes."""
        out0 = shapes.ellipse(self.nx, self.ny, self.rx, self.ry, self.rot,
                              self.xOffset, self.yOffset,
                              nSubpixels=self.nSubpixels)
        out1 = shapes.ellipse(self.nx+1, self.ny+1, self.rx, self.ry, self.rot,
                              self.xOffset, self.yOffset,
                              nSubpixels=self.nSubpixels)
        out1 = pad_crop(out1, out0.shape)

        self.assertTrue(np.allclose(out0, out1, rtol=1e-3),
                        msg='Centering changed with array size.')


class TestRectangle(unittest.TestCase):
    """Unit tests for shapes.rectangle()."""

    def setUp(self):
        self.nx = 51
        self.ny = 60
        self.width = 5.0
        self.height = 6.5
        self.rot = 0.0
        self.xOffset = 10
        self.yOffset = -8.2
        self.nSubpixels = 101
        self.isDark = False

    def test_input_failures(self):
        for badVal in (-1, 0, 1j, 1.5, [1, ], 'string'):
            with self.assertRaises(TypeError):
                shapes.rectangle(
                    badVal, self.ny, self.width, self.height,
                    self.xOffset, self.yOffset, rot=self.rot,
                    nSubpixels=self.nSubpixels, isDark=self.isDark)
        for badVal in (-1, 0, 1j, 1.5, [1, ], 'string'):
            with self.assertRaises(TypeError):
                shapes.rectangle(
                    self.nx, badVal, self.width, self.height,
                    self.xOffset, self.yOffset, rot=self.rot,
                    nSubpixels=self.nSubpixels, isDark=self.isDark)
        for badVal in (-1, 0, 1j, [1, ], 'string'):
            with self.assertRaises(TypeError):
                shapes.rectangle(
                    self.nx, self.ny, badVal, self.height,
                    self.xOffset, self.yOffset, rot=self.rot,
                    nSubpixels=self.nSubpixels, isDark=self.isDark)
        for badVal in (-1, 0, 1j, [1, ], 'string'):
            with self.assertRaises(TypeError):
                shapes.rectangle(
                    self.nx, self.ny, self.width, badVal,
                    self.xOffset, self.yOffset, rot=self.rot,
                    nSubpixels=self.nSubpixels, isDark=self.isDark)
        for badVal in (1j, [1, ], 'string'):
            with self.assertRaises(TypeError):
                shapes.rectangle(
                    self.nx, self.ny, self.width, self.height,
                    badVal, self.yOffset, rot=self.rot,
                    nSubpixels=self.nSubpixels, isDark=self.isDark)
        for badVal in (1j, [1, ], 'string'):
            with self.assertRaises(TypeError):
                shapes.rectangle(
                    self.nx, self.ny, self.width, self.height,
                    self.xOffset, badVal, rot=self.rot,
                    nSubpixels=self.nSubpixels, isDark=self.isDark)
        for badVal in (1j, [1, ], 'string'):
            with self.assertRaises(TypeError):
                shapes.rectangle(
                    self.nx, self.ny, self.width, self.height,
                    self.xOffset, self.yOffset, rot=badVal,
                    nSubpixels=self.nSubpixels, isDark=self.isDark)
        for badVal in (-1, 0, 1j, 1.5, [1, ], 'string'):
            with self.assertRaises(TypeError):
                shapes.rectangle(
                    self.nx, self.ny, self.width, self.height,
                    self.xOffset, self.yOffset, rot=self.rot,
                    nSubpixels=badVal, isDark=self.isDark)
        for badVal in (-1, 1j, [1, ], 'string'):
            with self.assertRaises(TypeError):
                shapes.rectangle(
                    self.nx, self.ny, self.width, self.height,
                    self.xOffset, self.yOffset, rot=self.rot,
                    nSubpixels=self.nSubpixels, isDark=badVal)

    def test_area(self):
        """Test that rectangle area matches analytical value."""
        out = shapes.rectangle(self.nx, self.ny, self.width, self.height,
                               self.xOffset, self.yOffset, rot=self.rot,
                               nSubpixels=self.nSubpixels, isDark=self.isDark)

        shapeSum = np.sum(out)
        expectedSum = self.width*self.height
        self.assertTrue(isclose(shapeSum, expectedSum, rel_tol=1e-3),
                        msg='Area of rectangle incorrect')

    def test_shift(self):
        """Test that rectangle shifts as expected."""
        xOffset = 0
        yOffset = 0
        maskCentered = shapes.rectangle(
            self.nx, self.ny, self.width, self.height, xOffset, yOffset,
            rot=self.rot, nSubpixels=self.nSubpixels, isDark=self.isDark)

        xOffset = -10
        yOffset = 5
        maskShifted = shapes.rectangle(
            self.nx, self.ny, self.width, self.height, xOffset, yOffset,
            rot=self.rot, nSubpixels=self.nSubpixels, isDark=self.isDark)
        maskRecentered = np.roll(maskShifted, (-yOffset, -xOffset),
                                 axis=(0, 1))

        diffSum = np.sum(np.abs(maskCentered - maskRecentered))
        self.assertTrue(isclose(diffSum, 0.),
                        msg='Offsets incorrectly applied.')

    def test_rotation(self):
        """Test that rectangle rotates as expected."""
        xOffset = 0
        yOffset = 0
        nx = 51
        ny = 51

        rot = 0
        mask0 = shapes.rectangle(
            nx, ny, self.width, self.height, xOffset, yOffset,
            rot=rot, nSubpixels=self.nSubpixels, isDark=self.isDark)

        rot = 90
        maskRot = shapes.rectangle(
            nx, ny, self.width, self.height, xOffset, yOffset,
            rot=rot, nSubpixels=self.nSubpixels, isDark=self.isDark)
        maskDerot = np.rot90(maskRot, 1)

        diffMax = np.max(np.abs(mask0 - maskDerot))

        abs_tol = 10 * np.finfo(float).eps
        self.assertTrue(isclose(diffMax, 0., abs_tol=abs_tol),
                        msg='Rotation incorrect.')

    def test_outside(self):
        """Test that the rectangle isn't in the array."""
        xOffset = 2*self.nx
        yOffset = -2*self.ny
        out = shapes.rectangle(
            self.nx, self.ny, self.width, self.height, xOffset, yOffset,
            rot=self.rot, nSubpixels=self.nSubpixels, isDark=self.isDark)
        self.assertTrue(isclose(np.sum(out), 0.),
                        msg='Rectangle should be outside array.')

    def test_centering(self):
        """Test that center doesn't shift for different array sizes."""
        out0 = shapes.rectangle(
            self.nx, self.ny, self.width, self.height, self.xOffset,
            self.yOffset, rot=self.rot, nSubpixels=self.nSubpixels,
            isDark=self.isDark)
        out1 = shapes.rectangle(
            self.nx+1, self.ny+1, self.width, self.height, self.xOffset,
            self.yOffset, rot=self.rot, nSubpixels=self.nSubpixels,
            isDark=self.isDark)
        out1 = pad_crop(out1, out0.shape)

        self.assertTrue(np.allclose(out0, out1, rtol=1e-3),
                        msg='Centering changed with array size.')


class TestSimplePupil(unittest.TestCase):
    """Unit tests for shapes.simple_pupil()."""

    def setUp(self):
        self.nx = 51
        self.ny = 60
        self.xOffset = 10
        self.yOffset = -8.2
        self.diamInner = 10.1
        self.diamOuter = 20.5
        self.strutAngles = [-45.0, 0, 90.0]
        self.strutWidth = 3.4
        self.nSubpixels = 101

    def test_input_failures(self):
        for badVal in (-1, 0, 1j, 1.5, [1, ], 'string'):
            with self.assertRaises(TypeError):
                shapes.simple_pupil(
                    badVal, self.ny, self.xOffset, self.yOffset,
                    self.diamInner, self.diamOuter,
                    strutAngles=self.strutAngles, strutWidth=self.strutWidth,
                    nSubpixels=self.nSubpixels)
        for badVal in (-1, 0, 1j, 1.5, [1, ], 'string'):
            with self.assertRaises(TypeError):
                shapes.simple_pupil(
                    self.nx, badVal, self.xOffset, self.yOffset,
                    self.diamInner, self.diamOuter,
                    strutAngles=self.strutAngles, strutWidth=self.strutWidth,
                    nSubpixels=self.nSubpixels)
        for badVal in (1j, [1, ], 'string'):
            with self.assertRaises(TypeError):
                shapes.simple_pupil(
                    self.nx, self.ny, badVal, self.yOffset,
                    self.diamInner, self.diamOuter,
                    strutAngles=self.strutAngles, strutWidth=self.strutWidth,
                    nSubpixels=self.nSubpixels)
        for badVal in (1j, [1, ], 'string'):
            with self.assertRaises(TypeError):
                shapes.simple_pupil(
                    self.nx, self.ny, self.xOffset, badVal,
                    self.diamInner, self.diamOuter,
                    strutAngles=self.strutAngles, strutWidth=self.strutWidth,
                    nSubpixels=self.nSubpixels)
        for badVal in (-1, 1j, [1, ], 'string', self.diamOuter+1):
            with self.assertRaises(TypeError):
                shapes.simple_pupil(
                    self.nx, self.ny, self.xOffset, self.yOffset,
                    badVal, self.diamOuter,
                    strutAngles=self.strutAngles, strutWidth=self.strutWidth,
                    nSubpixels=self.nSubpixels)
        for badVal in (-1, 0, 1j, [1, ], 'string', self.diamInner-1):
            with self.assertRaises(TypeError):
                shapes.simple_pupil(
                    self.nx, self.ny, self.xOffset, self.yOffset,
                    self.diamInner, badVal,
                    strutAngles=self.strutAngles, strutWidth=self.strutWidth,
                    nSubpixels=self.nSubpixels)
        for badVal in (-1, 0, 1j, 1.5, np.eye(2), 'string'):
            with self.assertRaises(TypeError):
                shapes.simple_pupil(
                    self.nx, self.ny, self.xOffset, self.yOffset,
                    self.diamInner, self.diamOuter,
                    strutAngles=badVal, strutWidth=self.strutWidth,
                    nSubpixels=self.nSubpixels)
        for badVal in (-1, 1j, [1, ], 'string'):
            with self.assertRaises(TypeError):
                shapes.simple_pupil(
                    self.nx, self.ny, self.xOffset, self.yOffset,
                    self.diamInner, self.diamOuter,
                    strutAngles=self.strutAngles, strutWidth=badVal,
                    nSubpixels=self.nSubpixels)
        for badVal in (-1, 0, 1j, 1.5, [1, ], 'string'):
            with self.assertRaises(TypeError):
                shapes.simple_pupil(
                    self.nx, self.ny, self.xOffset, self.yOffset,
                    self.diamInner, self.diamOuter,
                    strutAngles=self.strutAngles, strutWidth=self.strutWidth,
                    nSubpixels=badVal)

    def test_area(self):
        """Test that pupil area matches analytical value."""
        out = shapes.simple_pupil(
                    self.nx, self.ny, self.xOffset, self.yOffset,
                    self.diamInner, self.diamOuter,
                    strutAngles=[], strutWidth=self.strutWidth,
                    nSubpixels=self.nSubpixels)
        shapeSum = np.sum(out)
        expectedSum = np.pi/4*(self.diamOuter**2 - self.diamInner**2)
        self.assertTrue(isclose(shapeSum, expectedSum, rel_tol=1e-3),
                        msg='Area of pupil incorrect')

    def test_shift(self):
        """Test that pupil shifts as expected."""
        xOffset = 0
        yOffset = 0
        maskCentered = shapes.simple_pupil(
                    self.nx, self.ny, xOffset, yOffset,
                    self.diamInner, self.diamOuter,
                    strutAngles=self.strutAngles, strutWidth=self.strutWidth,
                    nSubpixels=self.nSubpixels)

        xOffset = -10
        yOffset = 5
        maskShifted = shapes.simple_pupil(
                    self.nx, self.ny, xOffset, yOffset,
                    self.diamInner, self.diamOuter,
                    strutAngles=self.strutAngles, strutWidth=self.strutWidth,
                    nSubpixels=self.nSubpixels)
        maskRecentered = np.roll(maskShifted, (-yOffset, -xOffset),
                                 axis=(0, 1))

        diffMax = np.max(np.abs(maskCentered - maskRecentered))
        abs_tol = 100 * np.finfo(float).eps
        self.assertTrue(isclose(diffMax, 0., abs_tol=abs_tol),
                        msg='Offsets incorrectly applied.')

    def test_strut_angles(self):
        """Test that strut angles are as expected."""
        xOffset = 0
        yOffset = 0
        nx = 51
        ny = 51

        strutAngles = [10, 150]
        mask0 = shapes.simple_pupil(
                    nx, ny, xOffset, yOffset,
                    self.diamInner, self.diamOuter,
                    strutAngles=strutAngles, strutWidth=self.strutWidth,
                    nSubpixels=self.nSubpixels)

        strutAngles = [10+90, 150+90]
        maskRot = shapes.simple_pupil(
                    nx, ny, xOffset, yOffset,
                    self.diamInner, self.diamOuter,
                    strutAngles=strutAngles, strutWidth=self.strutWidth,
                    nSubpixels=self.nSubpixels)
        maskDerot = np.rot90(maskRot, 1)

        diffMax = np.max(np.abs(mask0 - maskDerot))

        abs_tol = 10 * np.finfo(float).eps
        self.assertTrue(isclose(diffMax, 0., abs_tol=abs_tol),
                        msg='Rotation incorrect.')

    def test_outside(self):
        """Test that the rectangle isn't in the array."""
        xOffset = 2*self.nx
        yOffset = -2*self.ny
        out = shapes.simple_pupil(
                    self.nx, self.ny, xOffset, yOffset,
                    self.diamInner, self.diamOuter,
                    strutAngles=self.strutAngles, strutWidth=self.strutWidth,
                    nSubpixels=self.nSubpixels)
        self.assertTrue(isclose(np.sum(out), 0.),
                        msg='Pupil should be outside array.')

    def test_centering(self):
        """Test that center doesn't shift for different array sizes."""
        out0 = shapes.simple_pupil(
                    self.nx, self.ny, self.xOffset, self.yOffset,
                    self.diamInner, self.diamOuter,
                    strutAngles=self.strutAngles, strutWidth=self.strutWidth,
                    nSubpixels=self.nSubpixels)
        out1 = shapes.simple_pupil(
                    self.nx+1, self.ny+1, self.xOffset, self.yOffset,
                    self.diamInner, self.diamOuter,
                    strutAngles=self.strutAngles, strutWidth=self.strutWidth,
                    nSubpixels=self.nSubpixels)
        out1 = pad_crop(out1, out0.shape)

        self.assertTrue(np.allclose(out0, out1, rtol=1e-3),
                        msg='Centering changed with array size.')


class TestAnnularSegments(unittest.TestCase):
    """Test that masks are generated as expected."""

    def test_number_of_segments(self):
        """Verify the 2-segment mask is composed of two 1-segment masks."""
        rInPix = 7.2
        rOutPix = 19.5
        nx = 60
        ny = 71
        xOffset = -5
        yOffset = 2
        angOpen = 65
        angRot = 15
        nsp = 100
        dTheta = 5

        nSeg = 1
        maskRight = shapes.annular_segments(nSeg, rInPix, rOutPix, nx, ny,
                                            xOffset, yOffset, angOpen, angRot,
                                            nSubpixels=nsp, dTheta=dTheta)
        maskLeft = shapes.annular_segments(
            nSeg, rInPix, rOutPix, nx, ny, xOffset, yOffset, angOpen,
            angRot+180, nSubpixels=nsp, dTheta=dTheta)
        nSeg = 2
        maskBoth = shapes.annular_segments(nSeg, rInPix, rOutPix, nx, ny,
                                           xOffset, yOffset, angOpen, angRot,
                                           nSubpixels=nsp, dTheta=dTheta)

        sumAbsDiff = np.sum(np.abs(maskBoth - maskRight - maskLeft))
        self.assertTrue(isclose(sumAbsDiff, 0, abs_tol=1e-8))

    def test_summed_area(self):
        """Test that the summed area matches the analytical value."""
        nSeg = 1
        rInPix = 7.2
        rOutPix = 19.5
        nx = 60
        ny = 71
        xOffset = -5
        yOffset = 2
        angOpen = 65
        angRot = 15
        nsp = 100
        dTheta = 5

        mask1 = shapes.annular_segments(
            nSeg, rInPix, rOutPix, nx, ny, xOffset,
            yOffset, angOpen, angRot, nSubpixels=nsp, dTheta=dTheta)
        area1 = np.sum(mask1)
        nSeg = 2
        mask2 = shapes.annular_segments(
            nSeg, rInPix, rOutPix, nx, ny, xOffset,
            yOffset, angOpen, angRot, nSubpixels=nsp, dTheta=dTheta)
        area2 = np.sum(mask2)

        areaTrue = np.pi * (rOutPix**2 - rInPix**2) * (angOpen/360)
        self.assertTrue(isclose(areaTrue, area1, rel_tol=1e-5))
        self.assertTrue(isclose(2*areaTrue, area2, rel_tol=1e-5))

    def test_offsets(self):
        """Test the lateral offsets by comparing to np.roll."""
        nSeg = 1
        rInPix = 7.2
        rOutPix = 19.5
        nx = 60
        ny = 71
        xOffset = -5
        yOffset = 2
        angOpen = 65
        angRot = 15
        nsp = 100
        dTheta = 5

        maskOffset = shapes.annular_segments(nSeg, rInPix, rOutPix, nx, ny,
            xOffset, yOffset, angOpen, angRot, nSubpixels=nsp, dTheta=dTheta)

        maskCentered = shapes.annular_segments(nSeg, rInPix, rOutPix, nx, ny,
            0, 0, angOpen, angRot, nSubpixels=nsp, dTheta=dTheta)

        sumAbsDiff = np.sum(np.abs(maskCentered - np.roll(maskOffset,
                                                          (-yOffset, -xOffset),
                                                          axis=(0, 1))))

        self.assertTrue(isclose(sumAbsDiff, 0, abs_tol=1e-4))

    def test_rotation(self):
        """Test the rotation by comparing to np.rot90."""
        nSeg = 1
        rInPix = 7.2
        rOutPix = 19.5
        nx = 61
        ny = 61
        xOffset = 0
        yOffset = 0
        angOpen = 65
        angRot = 15
        nsp = 100
        dTheta = 5

        mask = shapes.annular_segments(nSeg, rInPix, rOutPix, nx, ny,
            xOffset, yOffset, angOpen, angRot, nSubpixels=nsp, dTheta=dTheta)

        maskRot90 = shapes.annular_segments(nSeg, rInPix, rOutPix, nx, ny,
            xOffset, yOffset, angOpen, angRot+90,
            nSubpixels=nsp, dTheta=dTheta)

        sumAbsDiff = np.sum(np.abs(maskRot90 - np.rot90(mask, -1)))

        self.assertTrue(isclose(sumAbsDiff, 0, abs_tol=1e-4))

    def test_return_zeros(self):
        """Test the lateral offsets by comparing to np.roll."""
        nSeg = 1
        rInPix = 7.2
        rOutPix = 19.5
        nx = 61
        ny = 61
        xOffset = 0
        yOffset = 0
        angOpen = 0
        angRot = 15
        nsp = 100
        dTheta = 5

        mask = shapes.annular_segments(nSeg, rInPix, rOutPix, nx, ny,
            xOffset, yOffset, angOpen, angRot, nSubpixels=nsp, dTheta=dTheta)

        self.assertTrue(isclose(0, np.sum(mask), abs_tol=1e-8))


class TestInputFailure(unittest.TestCase):
    """Test suite for valid function inputs."""

    def test_annular_segments_inputs(self):
        """Test the inputs of shapes.annular_segments_inputs."""
        rInPix = 7.2
        rOutPix = 19.5
        nx = 60
        ny = 71
        xOffset = -5
        yOffset = 2
        angOpen = 65
        angRot = 15
        nsp = 100
        dTheta = 5
        nSeg = 1

        # Check standard inputs do not raise anything first
        shapes.annular_segments(nSeg, rInPix, rOutPix, nx, ny,
                             xOffset, yOffset, angOpen, angRot,
                             nSubpixels=nsp, dTheta=dTheta)

        for nSegBad in (-1, 0, 3, 2.1, 1j, np.ones(5), np.ones((5, 2)), 'a'):
            with self.assertRaises(TypeError):
                shapes.annular_segments(nSegBad, rInPix, rOutPix, nx, ny,
                                 xOffset, yOffset, angOpen, angRot,
                                 nSubpixels=nsp, dTheta=dTheta)

        for rInPixBad in (-1, 0, 1j, [5], np.ones(5), np.ones((5, 2)), 'a'):
            with self.assertRaises(TypeError):
                shapes.annular_segments(nSeg, rInPixBad, rOutPix, nx, ny,
                                 xOffset, yOffset, angOpen, angRot,
                                 nSubpixels=nsp, dTheta=dTheta)

        for rOutPixBad in (-1, 0, 1j, [5], np.ones(5), np.ones((5, 2)), 'a'):
            with self.assertRaises(TypeError):
                shapes.annular_segments(nSeg, rInPix, rOutPixBad, nx, ny,
                                 xOffset, yOffset, angOpen, angRot,
                                 nSubpixels=nsp, dTheta=dTheta)

        for nxBad in (-1, 0, 2.1, 1j, np.ones(5), np.ones((5, 2)), 'a'):
            with self.assertRaises(TypeError):
                shapes.annular_segments(nSeg, rInPix, rOutPix, nxBad, ny,
                                 xOffset, yOffset, angOpen, angRot,
                                 nSubpixels=nsp, dTheta=dTheta)

        for nyBad in (-1, 0, 2.1, 1j, np.ones(5), np.ones((5, 2)), 'a'):
            with self.assertRaises(TypeError):
                shapes.annular_segments(nSeg, rInPix, rOutPix, nx, nyBad,
                                 xOffset, yOffset, angOpen, angRot,
                                 nSubpixels=nsp, dTheta=dTheta)

        for xOffsetBad in (1j, [5], np.ones(5), np.ones((5, 2)), 'a'):
            with self.assertRaises(TypeError):
                shapes.annular_segments(nSeg, rInPix, rOutPix, nx, ny,
                                 xOffsetBad, yOffset, angOpen, angRot,
                                 nSubpixels=nsp, dTheta=dTheta)

        for yOffsetBad in (1j, [5], np.ones(5), np.ones((5, 2)), 'a'):
            with self.assertRaises(TypeError):
                shapes.annular_segments(nSeg, rInPix, rOutPix, nx, ny,
                                 xOffset, yOffsetBad, angOpen, angRot,
                                 nSubpixels=nsp, dTheta=dTheta)

        for angOpenBad in (-1, 1j, [5], np.ones(5), np.ones((5, 2)), 'a'):
            with self.assertRaises(TypeError):
                shapes.annular_segments(nSeg, rInPix, rOutPix, nx, ny,
                                 xOffset, yOffset, angOpenBad, angRot,
                                 nSubpixels=nsp, dTheta=dTheta)

        for angRotBad in (1j, [5], np.ones(5), np.ones((5, 2)), 'a'):
            with self.assertRaises(TypeError):
                shapes.annular_segments(nSeg, rInPix, rOutPix, nx, ny,
                                 xOffset, yOffset, angOpen, angRotBad,
                                 nSubpixels=nsp, dTheta=dTheta)

        for nspBad in (-1, 0, 2.1, 1j, [5], np.ones(5), np.ones((5, 2)), 'a'):
            with self.assertRaises(TypeError):
                shapes.annular_segments(nSeg, rInPix, rOutPix, nx, ny,
                                 xOffset, yOffset, angOpen, angRot,
                                 nSubpixels=nspBad, dTheta=dTheta)

        for dThetaBad in (-1, 1j, [5], np.ones(5), np.ones((5, 2)), 'a'):
            with self.assertRaises(TypeError):
                shapes.annular_segments(nSeg, rInPix, rOutPix, nx, ny,
                                 xOffset, yOffset, angOpen, angRot,
                                 nSubpixels=nsp, dTheta=dThetaBad)



if __name__ == '__main__':
    unittest.main()
