"""Test suite for OCCASTRO module."""
import unittest
import os
import numpy as np
from astropy.io import fits

from coralign.occastro.occastro import (
    calc_star_location_from_spots,
    calc_spot_separation,
)


class TestOccastroInputFailure(unittest.TestCase):
    """Test suite for valid function inputs."""

    def test_calc_offset_from_spots_inputs(self):
        """Test the inputs of calc_star_location_from_spots."""
        localpath = os.path.dirname(os.path.abspath(__file__))
        fnTuning = os.path.join(localpath, 'testdata',
                                'occastro_offset_tuning_nfov.yaml')
        fnSpots = os.path.join(localpath, 'testdata',
                               'occastro_nfov_4spots_R6.50_no_errors.fits')
        spotArray = fits.getdata(fnSpots)

        xOffsetGuess = 0
        yOffsetGuess = 0

        # Check that standard inputs do not raise anything first
        _, _ = calc_star_location_from_spots(
            spotArray, xOffsetGuess, yOffsetGuess, fnTuning)

        # Tests of bad inputs
        # except for fnTuning which uses the tests of util.loadyaml()
        with self.assertRaises(TypeError):
            calc_star_location_from_spots(
                np.ones((10, )), xOffsetGuess, yOffsetGuess, fnTuning)
        with self.assertRaises(TypeError):
            calc_star_location_from_spots(
                spotArray, [-1.3], yOffsetGuess, fnTuning)
        with self.assertRaises(TypeError):
            calc_star_location_from_spots(
                spotArray, xOffsetGuess, [-1.3], fnTuning)
        with self.assertRaises(TypeError):
            calc_star_location_from_spots(
                spotArray, xOffsetGuess, yOffsetGuess, 1)

    def test_calc_spot_separation_inputs(self):
        """Test the inputs of calc_spot_separation."""
        localpath = os.path.dirname(os.path.abspath(__file__))
        fnTuning = os.path.join(localpath, 'testdata',
                                'occastro_separation_tuning_nfov.yaml')
        fnSpots = os.path.join(localpath, 'testdata',
                               'occastro_nfov_4spots_R6.50_no_errors.fits')
        spotArray = fits.getdata(fnSpots)

        xOffset = 0
        yOffset = 0

        # Check that standard inputs do not raise anything first
        _ = calc_spot_separation(spotArray, xOffset, yOffset, fnTuning)

        # Tests of bad inputs
        # except for fnTuning which uses the tests of util.loadyaml()
        with self.assertRaises(TypeError):
            calc_star_location_from_spots(
                np.ones((10, )), xOffset, yOffset, fnTuning)
        with self.assertRaises(TypeError):
            calc_star_location_from_spots(
                spotArray, [-1.3], yOffset, fnTuning)
        with self.assertRaises(TypeError):
            calc_star_location_from_spots(
                spotArray, xOffset, [-1.3], fnTuning)
        with self.assertRaises(TypeError):
            calc_star_location_from_spots(
                spotArray, xOffset, yOffset, 1)


class TestOccastroOffset(unittest.TestCase):
    """Integration tests of occastro's stellar offset estimation."""

    def test_occastro_offset_nfov(self):
        """Make sure NFOV astrometry is estimated to within +/- 0.1 pixels."""
        localpath = os.path.dirname(os.path.abspath(__file__))
        fnTuning = os.path.join(localpath, 'testdata',
                                'occastro_offset_tuning_nfov.yaml')
        fnSpots = os.path.join(localpath, 'testdata',
                               'occastro_nfov_4spots_R6.50_no_errors.fits')
        spotArray = fits.getdata(fnSpots)

        xOffsetGuessVec = [2, 0]
        yOffsetGuessVec = [-2, 0]
        nOffsets = len(xOffsetGuessVec)

        for iOffset in range(nOffsets):
            xOffsetGuess = xOffsetGuessVec[iOffset]
            yOffsetGuess = yOffsetGuessVec[iOffset]
            xOffsetEst, yOffsetEst = calc_star_location_from_spots(
                spotArray, xOffsetGuess, yOffsetGuess, fnTuning)

            xErrorPix = np.abs(xOffsetEst + xOffsetGuess)
            yErrorPix = np.abs(yOffsetEst + yOffsetGuess)
            self.assertTrue(xErrorPix < 0.1)
            self.assertTrue(yErrorPix < 0.1)

    def test_occastro_offset_spec(self):
        """Make sure Spec astrometry is estimated to within +/- 0.1 pixels."""
        localpath = os.path.dirname(os.path.abspath(__file__))
        fnTuning = os.path.join(localpath, 'testdata',
                                'occastro_offset_tuning_spec.yaml')
        fnSpots = os.path.join(localpath, 'testdata',
                               'occastro_spec_2spots_R6.00_no_errors.fits')
        spotArray = fits.getdata(fnSpots)

        xOffsetGuessVec = [2, 0]
        yOffsetGuessVec = [-2, 0]
        nOffsets = len(xOffsetGuessVec)

        for iOffset in range(nOffsets):
            xOffsetGuess = xOffsetGuessVec[iOffset]
            yOffsetGuess = yOffsetGuessVec[iOffset]
            xOffsetEst, yOffsetEst = calc_star_location_from_spots(
                spotArray, xOffsetGuess, yOffsetGuess, fnTuning)

            xErrorPix = np.abs(xOffsetEst + xOffsetGuess)
            yErrorPix = np.abs(yOffsetEst + yOffsetGuess)
            self.assertTrue(xErrorPix < 0.1)
            self.assertTrue(yErrorPix < 0.1)

    def test_occastro_offset_wfov(self):
        """Make sure WFOV astrometry is estimated to within +/- 0.1 pixels."""
        localpath = os.path.dirname(os.path.abspath(__file__))
        fnTuning = os.path.join(localpath, 'testdata',
                                'occastro_offset_tuning_wfov.yaml')
        fnSpots = os.path.join(localpath, 'testdata',
                               'occastro_wfov_4spots_R13.00_no_errors.fits')
        spotArray = fits.getdata(fnSpots)

        xOffsetGuessVec = [2, 0]
        yOffsetGuessVec = [-2, 0]
        nOffsets = len(xOffsetGuessVec)

        for iOffset in range(nOffsets):
            xOffsetGuess = xOffsetGuessVec[iOffset]
            yOffsetGuess = yOffsetGuessVec[iOffset]
            xOffsetEst, yOffsetEst = calc_star_location_from_spots(
                spotArray, xOffsetGuess, yOffsetGuess, fnTuning)

            xErrorPix = np.abs(xOffsetEst + xOffsetGuess)
            yErrorPix = np.abs(yOffsetEst + yOffsetGuess)
            self.assertTrue(xErrorPix < 0.1)
            self.assertTrue(yErrorPix < 0.1)


class TestOccastroSeparation(unittest.TestCase):
    """Integration tests of occastro's spot separation estimation."""

    def test_occastro_separation_nfov(self):
        """Test that NFOV spot separation is accuracte to +/- 0.1 pixels."""
        localpath = os.path.dirname(os.path.abspath(__file__))
        fnTuning = os.path.join(localpath, 'testdata',
                                'occastro_separation_tuning_nfov.yaml')
        fnSpots = os.path.join(localpath, 'testdata',
                               'occastro_nfov_4spots_R6.50_no_errors.fits')
        spotArray = fits.getdata(fnSpots)

        xOffset = 0
        yOffset = 0
        spotSepEst = calc_spot_separation(spotArray,
                                          xOffset,
                                          yOffset,
                                          fnTuning)
        spotSepTrue = 14.79
        errorPix = np.abs(spotSepEst - spotSepTrue)
        self.assertTrue(errorPix < 0.1)

    def test_occastro_separation_spec(self):
        """Test that Spec spot separation is accuracte to +/- 0.1 pixels."""
        localpath = os.path.dirname(os.path.abspath(__file__))
        fnTuning = os.path.join(localpath, 'testdata',
                                'occastro_separation_tuning_spec.yaml')
        fnSpots = os.path.join(localpath, 'testdata',
                               'occastro_spec_2spots_R6.00_no_errors.fits')
        spotArray = fits.getdata(fnSpots)

        xOffset = 0
        yOffset = 0
        spotSepEst = calc_spot_separation(spotArray,
                                          xOffset,
                                          yOffset,
                                          fnTuning)
        spotSepTrue = 17.34
        errorPix = np.abs(spotSepEst - spotSepTrue)
        self.assertTrue(errorPix < 0.1)

    def test_occastro_separation_wfov(self):
        """Test that WFOV spot separation is accuracte to +/- 0.1 pixels."""
        localpath = os.path.dirname(os.path.abspath(__file__))
        fnTuning = os.path.join(localpath, 'testdata',
                                'occastro_separation_tuning_wfov.yaml')
        fnSpots = os.path.join(localpath, 'testdata',
                               'occastro_wfov_4spots_R13.00_no_errors.fits')
        spotArray = fits.getdata(fnSpots)

        xOffset = 0
        yOffset = 0
        spotSepEst = calc_spot_separation(spotArray,
                                          xOffset,
                                          yOffset,
                                          fnTuning)
        spotSepTrue = 42.45
        errorPix = np.abs(spotSepEst - spotSepTrue)
        self.assertTrue(errorPix < 0.1)


if __name__ == '__main__':
    unittest.main()
