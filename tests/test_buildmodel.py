"""Unit tests for the DARKHOLE module."""
import os
import unittest
from math import isclose
import numpy as np

from coralign.buildmodel import darkhole as dh


class TestDarkHole(unittest.TestCase):
    """Test the methods of darkhole."""

    def test_gen_dark_hole_from_yaml(self):
        """Test that all shapes can be combined from a YAML file."""
        localpath = os.path.dirname(os.path.abspath(__file__))
        fnSpecs = os.path.join(localpath, 'testdata',
                               'ut_dark_hole_shapes.yaml')
        output = dh.gen_dark_hole_from_yaml(fnSpecs)
        self.assertTrue(np.allclose(output.shape, (500, 600)))

    def test_area_from_yaml_generated_shapes(self):
        """Verify the total area of the overlapping rectangles."""
        localpath = os.path.dirname(os.path.abspath(__file__))
        fnSpecs = os.path.join(localpath, 'testdata',
                               'ut_dark_hole_overlapping_rectangles.yaml')

        output = dh.gen_dark_hole_from_yaml(fnSpecs)
        area = np.sum(output.astype(float))

        width = 9
        height = 11
        xOffset = 3
        areaExpected = height*width*2 - height*(width-2*xOffset)

        self.assertTrue(isclose(area, areaExpected, rel_tol=1e-4))

    def test_gen_coords_output_shapes(self):
        """Test that the outputs of gen_coords have the correct shapes."""
        arrayShape = (10, 11)
        xOffset = 3
        yOffset = -2
        X, Y, R, THETA = dh.gen_coords(arrayShape, xOffset, yOffset)

        self.assertTrue(np.allclose(arrayShape, X.shape))
        self.assertTrue(np.allclose(arrayShape, Y.shape))
        self.assertTrue(np.allclose(arrayShape, R.shape))
        self.assertTrue(np.allclose(arrayShape, THETA.shape))

    def test_gen_coords_values(self):
        """Test that coordinates have the expected values."""
        arrayShape = (5, 8)
        xOffset = 2
        yOffset = -1
        X, Y, R, THETA = dh.gen_coords(arrayShape, xOffset, yOffset)

        middleRowIndex = arrayShape[0] // 2
        middleColumnIndex = arrayShape[1] // 2

        self.assertTrue(X[-1, 0] == -6)
        self.assertTrue(X[-1, -1] == 1)
        self.assertTrue(Y[0, -1] == -1)
        self.assertTrue(Y[-1, -1] == 3)
        self.assertTrue(isclose(R[middleRowIndex, middleColumnIndex],
                                np.sqrt(xOffset**2 + yOffset**2),
                                rel_tol=np.finfo(float).eps))
        self.assertTrue(isclose(THETA[1, -1], 0,
                                rel_tol=np.finfo(float).eps))
        self.assertTrue(isclose(THETA[-1, 6], np.pi/2,
                                rel_tol=np.finfo(float).eps))

    def test_gen_rectangle_area(self):
        """Verify the area of the rectangle."""
        arrayShape = (200, 180)
        xOffset = 5.
        yOffset = -10.
        width = 101.
        height = 51.

        rectangle = dh.gen_rectangle(arrayShape, xOffset, yOffset,
                                     width, height)
        area = np.sum(rectangle.astype(float))
        areaExpected = width * height

        self.assertTrue(isclose(area, areaExpected, rel_tol=1e-4))

    def test_gen_rectangle_translation(self):
        """Test translation of the rectangle with np.roll."""
        arrayShape = (200, 180)
        xOffset = 5.
        yOffset = -10.
        width = 101.
        height = 51.

        rectangleOffset = dh.gen_rectangle(arrayShape, xOffset, yOffset,
                                           width, height)
        rectangleRecentered = np.roll(rectangleOffset,
                                      [-int(yOffset), -int(xOffset)],
                                      axis=(0, 1))
        rectangle = dh.gen_rectangle(arrayShape, 0, 0, width, height)

        self.assertTrue(np.allclose(rectangle, rectangleRecentered))

    def test_gen_annulus_area(self):
        """Verify the area of the annulus."""
        arrayShape = (1000, 999)
        xOffset = 5.
        yOffset = -10.
        radiusInner = 50.
        radiusOuter = 400.

        annulus = dh.gen_annulus(arrayShape, xOffset, yOffset,
                                 radiusInner, radiusOuter)
        area = np.sum(annulus.astype(float))
        areaExpected = np.pi*(radiusOuter**2 - radiusInner**2)

        self.assertTrue(isclose(area, areaExpected, rel_tol=1e-3))

    def test_gen_annulus_translation(self):
        """Test translation of the annulus with np.roll."""
        arrayShape = (200, 180)
        xOffset = 5.
        yOffset = -10.
        radiusInner = 10.
        radiusOuter = 35.

        annulusOffset = dh.gen_annulus(arrayShape, xOffset, yOffset,
                                       radiusInner, radiusOuter)
        annulusRecentered = np.roll(annulusOffset,
                                    [-int(yOffset), -int(xOffset)],
                                    axis=(0, 1))
        annulus = dh.gen_annulus(arrayShape, 0, 0, radiusInner, radiusOuter)

        self.assertTrue(np.allclose(annulus, annulusRecentered))

    def test_gen_annular_sector_area(self):
        """Verify the area of the annular sector."""
        arrayShape = (1000, 999)
        xOffset = 5.
        yOffset = -10.
        radiusInner = 50.
        radiusOuter = 400.
        openingAngle = 80.5
        clocking = 10.

        sector = dh.gen_annular_sector(arrayShape, xOffset, yOffset,
                                       radiusInner, radiusOuter,
                                       openingAngle, clocking)
        area = np.sum(sector.astype(float))
        areaExpected = (np.pi * (radiusOuter**2 - radiusInner**2) *
                        (openingAngle/360))

        self.assertTrue(isclose(area, areaExpected, rel_tol=1e-3))

    def test_gen_annular_sector_translation(self):
        """Test translation of the annular sector with np.roll."""
        arrayShape = (200, 180)
        xOffset = 5.
        yOffset = -10.
        radiusInner = 10.
        radiusOuter = 35.
        openingAngle = 80.5
        clocking = 10.

        sectorOffset = dh.gen_annular_sector(arrayShape, xOffset, yOffset,
                                             radiusInner, radiusOuter,
                                             openingAngle, clocking)
        sectorRecentered = np.roll(sectorOffset,
                                   [-int(yOffset), -int(xOffset)],
                                   axis=(0, 1))
        sector = dh.gen_annular_sector(arrayShape, 0, 0, radiusInner,
                                       radiusOuter, openingAngle, clocking)

        self.assertTrue(np.allclose(sector, sectorRecentered))

    def test_gen_annular_sector_rotation(self):
        """Test rotation of the annular sector with np.rot90."""
        arrayShape = (201, 201)
        xOffset = 0.
        yOffset = 0.
        radiusInner = 10.
        radiusOuter = 35.
        openingAngle = 80.5
        clocking = 90.

        sectorRotated = dh.gen_annular_sector(arrayShape, xOffset, yOffset,
                                              radiusInner, radiusOuter,
                                              openingAngle, clocking)
        sectorDerotated = np.rot90(sectorRotated)
        sector = dh.gen_annular_sector(arrayShape, 0, 0, radiusInner,
                                       radiusOuter, openingAngle, 0)

        self.assertTrue(np.allclose(sector, sectorDerotated))

    def test_gen_bowtie_area(self):
        """Verify the area of the bowtie."""
        arrayShape = (1000, 999)
        xOffset = 5.
        yOffset = -10.
        radiusInner = 50.
        radiusOuter = 400.
        openingAngle = 80.5
        clocking = 10.

        sector = dh.gen_bowtie(arrayShape, xOffset, yOffset,
                               radiusInner, radiusOuter,
                               openingAngle, clocking)
        area = np.sum(sector.astype(float))
        areaExpected = (np.pi * (radiusOuter**2 - radiusInner**2) *
                        (openingAngle/360)) * 2

        self.assertTrue(isclose(area, areaExpected, rel_tol=1e-3))

    def test_gen_bowtie_translation(self):
        """Test translation of the bowtie with np.roll."""
        arrayShape = (200, 180)
        xOffset = 5.
        yOffset = -10.
        radiusInner = 10.
        radiusOuter = 35.
        openingAngle = 80.5
        clocking = 10.

        bowtieOffset = dh.gen_annular_sector(arrayShape, xOffset, yOffset,
                                             radiusInner, radiusOuter,
                                             openingAngle, clocking)
        bowtieRecentered = np.roll(bowtieOffset,
                                   [-int(yOffset), -int(xOffset)],
                                   axis=(0, 1))
        bowtie = dh.gen_annular_sector(arrayShape, 0, 0, radiusInner,
                                       radiusOuter, openingAngle, clocking)

        self.assertTrue(np.allclose(bowtie, bowtieRecentered))

    def test_gen_bowtie_rotation(self):
        """Test rotation of the bowtie with np.rot90."""
        arrayShape = (201, 201)
        xOffset = 0.
        yOffset = 0.
        radiusInner = 10.
        radiusOuter = 35.
        openingAngle = 80.5
        clocking = 90.

        bowtieRotated = dh.gen_annular_sector(arrayShape, xOffset, yOffset,
                                              radiusInner, radiusOuter,
                                              openingAngle, clocking)
        bowtieDerotated = np.rot90(bowtieRotated)
        bowtie = dh.gen_annular_sector(arrayShape, 0, 0, radiusInner,
                                       radiusOuter, openingAngle, 0)

        self.assertTrue(np.allclose(bowtie, bowtieDerotated))


class TestDarkholeYamlInputFailure(unittest.TestCase):
    """Tests for valid YAML inputs of gen_dark_hole_from_yaml()."""

    def test_nRows(self):
        """Test that an exception is raised when a bad YAML value is read."""
        localpath = os.path.dirname(os.path.abspath(__file__))
        fnSpecs = os.path.join(localpath, 'testdata',
                               'ut_dark_hole_shapes_bad_input_0.yaml')
        with self.assertRaises(TypeError):
            dh.gen_dark_hole_from_yaml(fnSpecs)

    def test_nCols(self):
        """Test that an exception is raised when a bad YAML value is read."""
        localpath = os.path.dirname(os.path.abspath(__file__))
        fnSpecs = os.path.join(localpath, 'testdata',
                               'ut_dark_hole_shapes_bad_input_1.yaml')
        with self.assertRaises(TypeError):
            dh.gen_dark_hole_from_yaml(fnSpecs)

    def test_shape_name(self):
        """Test that an exception is raised when a bad YAML value is read."""
        localpath = os.path.dirname(os.path.abspath(__file__))
        fnSpecs = os.path.join(localpath, 'testdata',
                               'ut_dark_hole_shapes_bad_input_2.yaml')
        with self.assertRaises(ValueError):
            dh.gen_dark_hole_from_yaml(fnSpecs)

    def test_no_yaml(self):
        """
        Test that an exception is raised when the YAML file cannot be read in
        """
        localpath = os.path.dirname(os.path.abspath(__file__))
        fndne = os.path.join(localpath, 'testdata',
                             'does_not_exist')
        with self.assertRaises(IOError):
            dh.gen_dark_hole_from_yaml(fndne)


class TestDarkholeInputFailure(unittest.TestCase):
    """Test suite for valid function inputs."""

    def setUp(self):
        """Initialize valid variable values for all the unit tests."""
        self.arrayShape = (100, 99)
        self.xOffset = -5.5
        self.yOffset = 3.2
        self.width = 9.0
        self.height = 8.5
        self.radiusInner = 2.8
        self.radiusOuter = 9.7
        self.openingAngle = 65.0
        self.clocking = -0.5

    def test_inputs_of_gen_coords(self):
        """Test the inputs of gen_coords."""
        for arrayShapeBad in (1, 1j, np.ones(3), np.eye(2), 's'):
            with self.assertRaises(TypeError):
                dh.gen_coords(arrayShapeBad, self.xOffset, self.yOffset)

        for xOffsetBad in (1j, np.ones(3), (-1, 2), (2, 1.1), np.eye(2), 's'):
            with self.assertRaises(TypeError):
                dh.gen_coords(self.arrayShape, xOffsetBad, self.yOffset)

        for yOffsetBad in (1j, np.ones(3), (-1, 2), (2, 1.1), np.eye(2), 's'):
            with self.assertRaises(TypeError):
                dh.gen_coords(self.arrayShape, self.xOffset, yOffsetBad)

    def test_inputs_of_gen_rectangle(self):
        """Test the inputs of gen_rectangle."""
        for arrayShapeBad in (1, 1j, np.ones(3), np.eye(2), 's'):
            with self.assertRaises(TypeError):
                dh.gen_rectangle(arrayShapeBad, self.xOffset, self.yOffset,
                                 self.width, self.height)

        for xOffsetBad in (1j, np.ones(3), (-1, 2), (2, 1.1), np.eye(2), 's'):
            with self.assertRaises(TypeError):
                dh.gen_rectangle(self.arrayShape, xOffsetBad, self.yOffset,
                                 self.width, self.height)

        for yOffsetBad in (1j, np.ones(3), (-1, 2), (2, 1.1), np.eye(2), 's'):
            with self.assertRaises(TypeError):
                dh.gen_rectangle(self.arrayShape, self.xOffset, yOffsetBad,
                                 self.width, self.height)

        for widthBad in (-1, 0, 1j, (2, 1.1), np.eye(2), 's'):
            with self.assertRaises(TypeError):
                dh.gen_rectangle(self.arrayShape, self.xOffset, yOffsetBad,
                                 widthBad, self.height)

        for heightBad in (-1, 0, 1j, (2, 1.1), np.eye(2), 's'):
            with self.assertRaises(TypeError):
                dh.gen_rectangle(self.arrayShape, self.xOffset, yOffsetBad,
                                 self.width, heightBad)

    def test_inputs_of_gen_annulus(self):
        """Test the inputs of gen_annulus."""
        for arrayShapeBad in (1, 1j, np.ones(3), np.eye(2), 's'):
            with self.assertRaises(TypeError):
                dh.gen_annulus(arrayShapeBad, self.xOffset, self.yOffset,
                               self.radiusInner, self.radiusOuter)

        for xOffsetBad in (1j, np.ones(3), (-1, 2), (2, 1.1), np.eye(2), 's'):
            with self.assertRaises(TypeError):
                dh.gen_annulus(self.arrayShape, xOffsetBad, self.yOffset,
                               self.radiusInner, self.radiusOuter)

        for yOffsetBad in (1j, np.ones(3), (-1, 2), (2, 1.1), np.eye(2), 's'):
            with self.assertRaises(TypeError):
                dh.gen_annulus(self.arrayShape, self.xOffset, yOffsetBad,
                               self.radiusInner, self.radiusOuter)

        for radiusInnerBad in (-1, 1j, (2, 1.1), np.eye(2), 's'):
            with self.assertRaises(TypeError):
                dh.gen_annulus(self.arrayShape, self.xOffset, self.yOffset,
                               radiusInnerBad, self.radiusOuter)

        for radiusOuterBad in (-1, 1j, (2, 1.1), np.eye(2), 's'):
            with self.assertRaises(TypeError):
                dh.gen_annulus(self.arrayShape, self.xOffset, self.yOffset,
                               self.radiusInner, radiusOuterBad)

    def test_rads_of_gen_annulus(self):
        """Test the input radius relationship of gen_annulus."""

        for radiusOuterBad in [0.5*self.radiusInner]:
            with self.assertRaises(ValueError):
                dh.gen_annulus(self.arrayShape, self.xOffset, self.yOffset,
                               self.radiusInner, radiusOuterBad)

    def test_inputs_of_gen_annular_sector(self):
        """Test the inputs of gen_annular_sector."""
        for arrayShapeBad in (1, 1j, np.ones(3), np.eye(2), 's'):
            with self.assertRaises(TypeError):
                dh.gen_annular_sector(arrayShapeBad, self.xOffset,
                                      self.yOffset, self.radiusInner,
                                      self.radiusOuter, self.openingAngle,
                                      self.clocking)

        for xOffsetBad in (1j, np.ones(3), (-1, 2), (2, 1.1), np.eye(2), 's'):
            with self.assertRaises(TypeError):
                dh.gen_annular_sector(self.arrayShape, xOffsetBad,
                                      self.yOffset, self.radiusInner,
                                      self.radiusOuter, self.openingAngle,
                                      self.clocking)

        for yOffsetBad in (1j, np.ones(3), (-1, 2), (2, 1.1), np.eye(2), 's'):
            with self.assertRaises(TypeError):
                dh.gen_annular_sector(self.arrayShape, self.xOffset,
                                      yOffsetBad, self.radiusInner,
                                      self.radiusOuter, self.openingAngle,
                                      self.clocking)

        for radiusInnerBad in (-1, 1j, (2, 1.1), np.eye(2), 's'):
            with self.assertRaises(TypeError):
                dh.gen_annular_sector(self.arrayShape, self.xOffset,
                                      self.yOffset, radiusInnerBad,
                                      self.radiusOuter, self.openingAngle,
                                      self.clocking)

        for radiusOuterBad in (-1, 1j, (2, 1.1), np.eye(2), 's'):
            with self.assertRaises(TypeError):
                dh.gen_annular_sector(self.arrayShape, self.xOffset,
                                      self.yOffset, self.radiusInner,
                                      radiusOuterBad, self.openingAngle,
                                      self.clocking)

        for openingAngleBad in (-1, 0, 1j, (2, 1.1), np.eye(2), 's'):
            with self.assertRaises(TypeError):
                dh.gen_annular_sector(self.arrayShape, self.xOffset,
                                      self.yOffset, radiusInnerBad,
                                      self.radiusOuter, openingAngleBad,
                                      self.clocking)

        for clockingBad in (1j, (2, 1.1), np.eye(2), 's'):
            with self.assertRaises(TypeError):
                dh.gen_annular_sector(self.arrayShape, self.xOffset,
                                      self.yOffset, radiusInnerBad,
                                      self.radiusOuter, self.openingAngle,
                                      clockingBad)

    def test_rads_of_gen_annular_sector(self):
        """Test the input radius relationship of gen_annular_sector."""
        for radiusOuterBad in [0.5*self.radiusInner]:
            with self.assertRaises(ValueError):
                dh.gen_annular_sector(self.arrayShape, self.xOffset,
                                      self.yOffset, self.radiusInner,
                                      radiusOuterBad, self.openingAngle,
                                      self.clocking)

    def test_inputs_of_gen_bowtie(self):
        """Test the inputs of gen_bowtie."""
        for arrayShapeBad in (1, 1j, np.ones(3), np.eye(2), 's'):
            with self.assertRaises(TypeError):
                dh.gen_bowtie(arrayShapeBad, self.xOffset,
                              self.yOffset, self.radiusInner,
                              self.radiusOuter, self.openingAngle,
                              self.clocking)

        for xOffsetBad in (1j, np.ones(3), (-1, 2), (2, 1.1), np.eye(2), 's'):
            with self.assertRaises(TypeError):
                dh.gen_bowtie(self.arrayShape, xOffsetBad,
                              self.yOffset, self.radiusInner,
                              self.radiusOuter, self.openingAngle,
                              self.clocking)

        for yOffsetBad in (1j, np.ones(3), (-1, 2), (2, 1.1), np.eye(2), 's'):
            with self.assertRaises(TypeError):
                dh.gen_bowtie(self.arrayShape, self.xOffset,
                              yOffsetBad, self.radiusInner,
                              self.radiusOuter, self.openingAngle,
                              self.clocking)

        for radiusInnerBad in (-1, 1j, (2, 1.1), np.eye(2), 's'):
            with self.assertRaises(TypeError):
                dh.gen_bowtie(self.arrayShape, self.xOffset,
                              self.yOffset, radiusInnerBad,
                              self.radiusOuter, self.openingAngle,
                              self.clocking)

        for radiusOuterBad in (-1, 1j, (2, 1.1), np.eye(2), 's'):
            with self.assertRaises(TypeError):
                dh.gen_bowtie(self.arrayShape, self.xOffset,
                              self.yOffset, self.radiusInner,
                              radiusOuterBad, self.openingAngle,
                              self.clocking)

        for openingAngleBad in (-1, 0, 1j, (2, 1.1), np.eye(2), 's'):
            with self.assertRaises(TypeError):
                dh.gen_bowtie(self.arrayShape, self.xOffset,
                              self.yOffset, radiusInnerBad,
                              self.radiusOuter, openingAngleBad,
                              self.clocking)

        for clockingBad in (1j, (2, 1.1), np.eye(2), 's'):
            with self.assertRaises(TypeError):
                dh.gen_bowtie(self.arrayShape, self.xOffset,
                              self.yOffset, radiusInnerBad,
                              self.radiusOuter, self.openingAngle,
                              clockingBad)

    def test_rads_of_gen_bowtie(self):
        """Test the input radius relationship of gen_bowtie."""
        for radiusOuterBad in [0.5*self.radiusInner]:
            with self.assertRaises(ValueError):
                dh.gen_bowtie(self.arrayShape, self.xOffset,
                              self.yOffset, self.radiusInner,
                              radiusOuterBad, self.openingAngle,
                              self.clocking)


if __name__ == '__main__':
    unittest.main()
