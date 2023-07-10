"""Test suite for coralign.pupil.pupilfit.py"""
from astropy.io import fits
from math import isclose
import numpy as np
import os
from tempfile import NamedTemporaryFile
import unittest

from coralign.pupil import pupilfit
from coralign.util import shapes
from coralign.util.loadyaml import loadyaml

from coralign.util.ampthresh import ampthresh
from coralign.util.nollzernikes import gen_zernikes
from coralign.pupil.pupilfit import fit_pupil_zernikes

LOCALPATH = os.path.dirname(os.path.abspath(__file__))

not_2d_list = [-1, 1.5, 1j, (1, ), np.ones((2, 3, 4)), 'asdf']


class TestFitUnmaskedPupil(unittest.TestCase):
    """Test suite for fit_unmasked_pupil()."""

    def setUp(self):
        """Define reused variables."""
        # self.fn_tuning = 'ut_tuning.yaml'
        self.fn_tuning = os.path.join(
            LOCALPATH, 'testdata', 'ut_fit_unmasked_pupil.yaml')
        self.fn_pupil_ellipse_fitting = os.path.join(
            LOCALPATH, 'testdata', 'ut_fit_pupil_as_ellipse.yaml')
        self.data_path = './'
        
        tuning_dict = loadyaml(self.fn_tuning)
        self.fnMaskRefHighRes = tuning_dict['fnMaskRefHighRes']
        self.diamHighRes = tuning_dict['diamHighRes']

    def test_fit_unmasked_pupil(self):
        """Test that the unmasked pupil parameters are fitted correctly."""
        diamTrue = 300
        clockDegTrue = 1.1
        xOffsetTrue = 10.1
        yOffsetTrue = -12.5
        nArray = 450

        pupilMeas = shapes.ellipse(nArray, nArray,
                                    0.5*diamTrue*0.9901, 0.5*diamTrue,
                                    clockDegTrue, xOffsetTrue, yOffsetTrue) -\
                    shapes.ellipse(nArray, nArray,
                                    0.05*diamTrue, 0.3*diamTrue,
                                    clockDegTrue, xOffsetTrue, yOffsetTrue)
        
        diamHighRes = self.diamHighRes
        nArray = int(self.diamHighRes + 50)
        pupilHighRes = shapes.ellipse(nArray, nArray,
                                    0.5*diamHighRes*0.9901, 0.5*diamHighRes,
                                    0, 0, 0) -\
                    shapes.ellipse(nArray, nArray,
                                    0.05*diamHighRes, 0.3*diamHighRes,
                                    0, 0, 0)

        # Refer to https://echorand.me/posts/named_temporary_file/ for help
        # with the temporary file usage.
        f = NamedTemporaryFile()
        # Save original name (the "name" actually is the absolute path)
        original_path = f.name
        f.name = self.fnMaskRefHighRes
        hdu = fits.PrimaryHDU(pupilHighRes)
        hdu.writeto(f.name, overwrite=True)        

        xOffsetEst, yOffsetEst, clockDegEst, diamEst = \
            pupilfit.fit_unmasked_pupil(
                pupilMeas, self.fn_tuning, self.fn_pupil_ellipse_fitting,
                data_path=self.data_path)

        self.assertTrue(np.abs(xOffsetEst-xOffsetTrue) < 0.5)  # pixels
        self.assertTrue(np.abs(yOffsetEst-yOffsetTrue) < 0.5)  # pixels
        self.assertTrue(np.abs(clockDegEst-clockDegTrue)*(1e3*np.pi/180.)
                        < 5.)  # mrad
        self.assertTrue((diamEst - diamTrue)/diamTrue*100 < 0.3)  # percent

        # Delete the temporary file
        os.unlink(f.name)


class TestFitPupilZernikes(unittest.TestCase):
    """
    Test that fit_pupil_zernikes operates correctly.

    Since fit_pupil_zernikes only calls functions that have their own unit tests,
    we just run one generated example with offsets, etc and check for correct
    result.
    """
    def setUp(self):
        """Define reused variables."""
        # self.fn_tuning = 'ut_tuning.yaml'
        self.fn_tuning = os.path.join(
            LOCALPATH, 'testdata', 'ut_fit_unmasked_pupil.yaml')
        self.fn_pupil_ellipse_fitting = os.path.join(
            LOCALPATH, 'testdata', 'ut_fit_pupil_as_ellipse.yaml')
        self.data_path = './'
        
        tuning_dict = loadyaml(self.fn_tuning)
        self.fnMaskRefHighRes = tuning_dict['fnMaskRefHighRes']
        self.diamHighRes = tuning_dict['diamHighRes']
        
        self.diamBeam = 300
        self.nArray = 302

        self.xOffset = 0.0
        self.yOffset = 0.0

        maxNollZern = 11
        self.zernCoefIn = 2*np.array(
            [0.17181050, 0.79094532, 0.94230985,
             0.41422633, 0.53925223, 0.80700204,
             0.64446254, 0.87674567, 0.21717132,
             0.78011424, 0.63288971]
        )
        self.wfe = gen_zernikes(
            np.arange(1, 1+maxNollZern), self.zernCoefIn,
            self.xOffset, self.yOffset, self.diamBeam, nArray=self.nArray)
        self.amp_pup = shapes.circle(self.nArray, self.nArray, self.diamBeam/2,
                                     self.xOffset, self.yOffset)
        self.mask = ampthresh(self.amp_pup)
        

    def test_fit_pupil_zernikes(self):
        # Switch to using a circle or ellipse to make it run faster.
        """Fabricate an example and test success for each return value."""
        # Generate a shifted, aberrated wavefront with Zernike modes and
        # standard amplitude mask

        # Generate the reference pupil
        diamHighRes = self.diamHighRes
        nArrayHighRes = int(self.diamHighRes + 50)
        pupilHighRes = shapes.circle(
            nArrayHighRes, nArrayHighRes, diamHighRes/2, 0, 0)

        # Save the pupil with the expected file name
        # Refer to https://echorand.me/posts/named_temporary_file/ for help
        # with the temporary file usage.
        f = NamedTemporaryFile()
        # Save original name (the "name" actually is the absolute path)
        original_path = f.name
        f.name = self.fnMaskRefHighRes
        hdu = fits.PrimaryHDU(pupilHighRes)
        hdu.writeto(f.name, overwrite=True)  

        # make the call
        zernCoefOut, dictResults = fit_pupil_zernikes(
            self.wfe, self.amp_pup, self.fn_tuning, self.fn_pupil_ellipse_fitting,
            bMask=self.mask, data_path=self.data_path,
        )

        # test zernike coeff
        self.assertTrue(np.all(
            [isclose(a, b, abs_tol=0.05) for a, b in zip(self.zernCoefIn,
                                                         zernCoefOut)]))

        # test parameter returns
        self.assertTrue(
            isclose(dictResults['diamEst'], self.diamBeam, rel_tol=3e-3))
        self.assertTrue(isclose(dictResults['xOffset'], self.xOffset,
                                abs_tol=0.3))
        self.assertTrue(isclose(dictResults['yOffset'], self.yOffset,
                                abs_tol=0.3))

        # Delete the temporary file
        os.unlink(f.name)


    def test_fit_zernikes_inputs(self):
        """Test inputs for fit_zernikes."""
        list_not_2drealarray = [5, np.ones((5,)), 2j*np.ones((5, 10)), 'asdf']

        # Generate the reference pupil
        diamHighRes = self.diamHighRes
        nArrayHighRes = int(self.diamHighRes + 50)
        pupilHighRes = shapes.circle(
            nArrayHighRes, nArrayHighRes, diamHighRes/2, 0, 0)

        # Save the pupil with the expected file name
        # Refer to https://echorand.me/posts/named_temporary_file/ for help
        # with the temporary file usage.
        f = NamedTemporaryFile()
        # Save original name (the "name" actually is the absolute path)
        original_path = f.name
        f.name = self.fnMaskRefHighRes
        hdu = fits.PrimaryHDU(pupilHighRes)
        hdu.writeto(f.name, overwrite=True)  

        # Make sure it runs normally first.
        fit_pupil_zernikes(self.wfe, self.amp_pup, self.fn_tuning,
                           self.fn_pupil_ellipse_fitting,
                           data_path=self.data_path)

        for wfe_bad in list_not_2drealarray:
            with self.assertRaises(TypeError):
                fit_pupil_zernikes(wfe_bad, self.amp_pup, self.fn_tuning,
                                   self.fn_pupil_ellipse_fitting)

        for amp_bad in list_not_2drealarray:
            with self.assertRaises(TypeError):
                fit_pupil_zernikes(self.wfe, amp_bad, self.fn_tuning,
                                  self.fn_pupil_ellipse_fitting)

        for mask_bad in list_not_2drealarray:
            with self.assertRaises(TypeError):
                fit_pupil_zernikes(self.wfe, self.amp_pup, self.fn_tuning,
                                   self.fn_pupil_ellipse_fitting,
                                   bMask=mask_bad)

        for maxNollZernBad in (1j, np.ones((5,)), np.ones((5, 10)), 'asdf'):
            with self.assertRaises(TypeError):
                fit_pupil_zernikes(self.wfe, self.amp_pup, self.fn_tuning,
                                   self.fn_pupil_ellipse_fitting,
                                   Z_noll_max=maxNollZernBad)

        for mask_selem_bad in list_not_2drealarray:
            with self.assertRaises(TypeError):
                fit_pupil_zernikes(self.wfe, self.amp_pup, self.fn_tuning,
                                   self.fn_pupil_ellipse_fitting,
                                   mask_selem=mask_selem_bad)

        for fn_bad in [1j, np.ones((5,)), np.ones((5, 10))]:
            with self.assertRaises(TypeError):
                fit_pupil_zernikes(self.wfe, self.amp_pup, fn_bad,
                                   self.fn_pupil_ellipse_fitting)

        for fn_bad in [1j, np.ones((5,)), np.ones((5, 10))]:
            with self.assertRaises(TypeError):
                fit_pupil_zernikes(self.wfe, self.amp_pup, self.fn_tuning,
                                   fn_bad)

        # Delete the temporary file
        os.unlink(f.name)


if __name__ == '__main__':
    unittest.main()
