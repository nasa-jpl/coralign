"""Test suite for maskfit."""
import os
from tempfile import NamedTemporaryFile
import unittest

import numpy as np
from astropy.io import fits

from coralign.pupil import maskfit
from coralign.util import math, shapes
from coralign.util.dmhtoph import dmhtoph, volts_to_dmh
from coralign.util.loadyaml import loadyaml

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))

not_2d_list = [-1, 0, 1.5, 1j, (1, ), np.ones((2, 3, 4)), 'asdf']
not_str_list = [-1, 0, 1.5, 1j, (1, ), np.ones((2, 3, 4))]
not_real_scalar_list = [1j, (10,), np.ones((10, 10, 10)), 'asdf']


class TestShapedPupilBeamShear(unittest.TestCase):
    """Unit tests for fit_z_induced_spm_shear."""

    def setUp(self):
        """Set up reused variables for testing fit_z_induced_spm_shear."""
        self.data_path = './'
        self.fnConfig = os.path.join(LOCAL_PATH, 'testdata',
                                     'ut_psffit_parms_spm_shear.yaml')

        self.badArray = np.zeros((10, 2))  # bad input with different shape

        self.xOffsetPupil = 50.80
        self.yOffsetPupil = -21.37
        
        self.xOffsetMask = 4.90
        self.yOffsetMask = -3.72
        
        self.xOffsetZinduced = 5.5
        self.yOffsetZinduced = -6.8

        self.magAtMask = 0.98
        self.clockDegMask = 0.5

        self.diamBeamAtMask = 300
        strutAnglesNominal = np.arange(45, 360, 90)

        mask_id = 0.40
        mask_od = 0.90
        mask_strut_width = 0.03

        # Generate pupil
        diamBeam = self.diamBeamAtMask
        nx = 501
        ny = 501
        nSubpixels = 10

        pupil = shapes.simple_pupil(
            nx, ny, self.xOffsetPupil, self.yOffsetPupil,  0.20*diamBeam,
            1.00*diamBeam, strutAngles=strutAnglesNominal,
            strutWidth=0.01*diamBeam, nSubpixels=nSubpixels)
        
        pupilZ = shapes.simple_pupil(
            nx, ny, self.xOffsetPupil+self.xOffsetZinduced,
            self.yOffsetPupil+self.yOffsetZinduced, 0.20*diamBeam,
            1.00*diamBeam, strutAngles=strutAnglesNominal,
            strutWidth=0.01*diamBeam, nSubpixels=nSubpixels)

        maskZ = shapes.simple_pupil(
            nx, ny, self.xOffsetPupil+self.xOffsetMask+self.xOffsetZinduced,
            self.yOffsetPupil+self.yOffsetMask+self.yOffsetZinduced,
            mask_id*diamBeam*self.magAtMask, mask_od*diamBeam*self.magAtMask,
            strutAngles=strutAnglesNominal+self.clockDegMask,
            strutWidth=mask_strut_width*diamBeam,
            nSubpixels=nSubpixels)

        nact = 48        
        deltaV1 = np.zeros((nact, nact))
        deltaV1[37:42, 24] = 5
        deltaV1[39, 22:27] = 5
        
        actGain = 4e-9  # m/V
        gainmap = actGain * np.ones((nact, nact))
        lam = 575e-9
        inf_func = np.squeeze(fits.getdata(os.path.join(
            LOCAL_PATH, 'testdata', 'influence_dm5v2.fits')))
        dmrad1 = volts_to_dmh(gainmap, deltaV1, lam)

        ppact_d = 10
        ppact_cx = self.diamBeamAtMask/46.3
        ppact_cy = ppact_cx
        flipx = False
        thact = 0

        dx = self.xOffsetPupil
        dy = self.yOffsetPupil
        phPoked0 = dmhtoph(ny, nx, dmrad1, nact, inf_func, ppact_d, ppact_cx,
                           ppact_cy, dx, dy, thact, flipx)

        dx = self.xOffsetPupil + self.xOffsetZinduced
        dy = self.yOffsetPupil + self.yOffsetZinduced
        phPoked1 = dmhtoph(ny, nx, dmrad1, nact, inf_func, ppact_d, ppact_cx,
                           ppact_cy, dx, dy, thact, flipx)

        self.ampUnpoked0 = pupil
        self.phUnpoked0 = np.zeros_like(phPoked0)
        self.phPoked0 = phPoked0

        self.ampUnpoked1 = pupilZ * maskZ
        self.phUnpoked1 = np.zeros_like(phPoked1)
        self.phPoked1 = phPoked1

    def test_performance(self):
        """Test performance of fit_z_induced_spm_shear."""
        xOffsetEst, yOffsetEst = maskfit.fit_z_induced_spm_shear(
            self.ampUnpoked0, self.phUnpoked0, self.phPoked0,
            self.ampUnpoked1, self.phUnpoked1, self.phPoked1,
            self.fnConfig,
        )
        
        abs_tol = 0.1  # pixels
        print('xOffsetEst = %.4f' % xOffsetEst)
        print('yOffsetEst = %.4f' % yOffsetEst)
        self.assertTrue(np.abs(self.xOffsetZinduced - xOffsetEst) < abs_tol)
        self.assertTrue(np.abs(self.yOffsetZinduced - yOffsetEst) < abs_tol)

    def test_bad_input_type_0(self):
        """Test with bad input type."""
        with self.assertRaises(TypeError):
            for badVal in not_2d_list:
                maskfit.fit_z_induced_spm_shear(
                    badVal, self.phUnpoked0, self.phPoked0,
                    self.ampUnpoked1, self.phUnpoked1, self.phPoked1,
                    self.fnConfig,
                )

    def test_bad_input_type_1(self):
        """Test with bad input type."""
        with self.assertRaises(TypeError):
            for badVal in not_2d_list:
                maskfit.fit_z_induced_spm_shear(
                    self.ampUnpoked0, badVal, self.phPoked0,
                    self.ampUnpoked1, self.phUnpoked1, self.phPoked1,
                    self.fnConfig,
                )

    def test_bad_input_type_2(self):
        """Test with bad input type."""
        with self.assertRaises(TypeError):
            for badVal in not_2d_list:
                maskfit.fit_z_induced_spm_shear(
                    self.ampUnpoked0, self.phUnpoked0, badVal,
                    self.ampUnpoked1, self.phUnpoked1, self.phPoked1,
                    self.fnConfig,
                )

    def test_bad_input_type_3(self):
        """Test with bad input type."""
        with self.assertRaises(TypeError):
            for badVal in not_2d_list:
                maskfit.fit_z_induced_spm_shear(
                    self.ampUnpoked0, self.phUnpoked0, self.phPoked0,
                    badVal, self.phUnpoked1, self.phPoked1,
                    self.fnConfig,
                )

    def test_bad_input_type_4(self):
        """Test with bad input type."""
        with self.assertRaises(TypeError):
            for badVal in not_2d_list:
                maskfit.fit_z_induced_spm_shear(
                    self.ampUnpoked0, self.phUnpoked0, self.phPoked0,
                    self.ampUnpoked1, badVal, self.phPoked1,
                    self.fnConfig,
                )

    def test_bad_input_type_5(self):
        """Test with bad input type."""
        with self.assertRaises(TypeError):
            for badVal in not_2d_list:
                maskfit.fit_z_induced_spm_shear(
                    self.ampUnpoked0, self.phUnpoked0, self.phPoked0,
                    self.ampUnpoked1, self.phUnpoked1, badVal,
                    self.fnConfig,
                )

    def test_bad_input_type_6(self):
        """Test with bad input type."""
        with self.assertRaises(TypeError):
            for badVal in not_str_list:
                maskfit.fit_z_induced_spm_shear(
                    self.ampUnpoked0, self.phUnpoked0, self.phPoked0,
                    self.ampUnpoked1, self.phUnpoked1, self.phPoked1,
                    badVal,
                )

    def test_bad_input_shape_0(self):
        """Test with bad input shape."""
        with self.assertRaises(ValueError):
            maskfit.fit_z_induced_spm_shear(
                self.badArray, self.phUnpoked0, self.phPoked0,
                self.ampUnpoked1, self.phUnpoked1, self.phPoked1,
                self.fnConfig,
            )

    def test_bad_input_shape_1(self):
        """Test with bad input shape."""
        with self.assertRaises(ValueError):
            maskfit.fit_z_induced_spm_shear(
                self.ampUnpoked0, self.badArray, self.phPoked0,
                self.ampUnpoked1, self.phUnpoked1, self.phPoked1,
                self.fnConfig,
            )

    def test_bad_input_shape_2(self):
        """Test with bad input shape."""
        with self.assertRaises(ValueError):
            maskfit.fit_z_induced_spm_shear(
                self.ampUnpoked0, self.phUnpoked0, self.badArray,
                self.ampUnpoked1, self.phUnpoked1, self.phPoked1,
                self.fnConfig,
            )

    def test_bad_input_shape_3(self):
        """Test with bad input shape."""
        with self.assertRaises(ValueError):
            maskfit.fit_z_induced_spm_shear(
                self.ampUnpoked0, self.phUnpoked0, self.phPoked0,
                self.badArray, self.phUnpoked1, self.phPoked1,
                self.fnConfig,
            )

    def test_bad_input_shape_4(self):
        """Test with bad input shape."""
        with self.assertRaises(ValueError):
            maskfit.fit_z_induced_spm_shear(
                self.ampUnpoked0, self.phUnpoked0, self.phPoked0,
                self.ampUnpoked1, self.badArray, self.phPoked1,
                self.fnConfig,
            )

    def test_bad_input_shape_5(self):
        """Test with bad input shape."""
        with self.assertRaises(ValueError):
            maskfit.fit_z_induced_spm_shear(
                self.ampUnpoked0, self.phUnpoked0, self.phPoked0,
                self.ampUnpoked1, self.phUnpoked1, self.badArray,
                self.fnConfig,
            )


class TestLyotStopMagClocking(unittest.TestCase):
    """Unit tests for fit_lyot_stop_mag_clocking."""

    def setUp(self):
        
        self.xOffsetPupil = 50.80
        self.yOffsetPupil = -21.37
        
        self.xOffsetMask = 20.5 #4.90
        self.yOffsetMask = -10 #-3.72
        
        self.offsetGain = 1

        self.magAtMask = 1.02
        self.clockDegMask = -1.5
        
        self.fnMaskParams = os.path.join(
            LOCAL_PATH, 'testdata',
            'ut_params_for_fitting_lyot_stop_mag_clocking.yaml')
        lyot_design_dict = loadyaml(self.fnMaskParams)
        self.fnOffsetParams = os.path.join(LOCAL_PATH, 'testdata',
                                           'fit_pupil_mask_offsets.yaml')
        self.fnLyotCalib = os.path.join(
            LOCAL_PATH, 'testdata', 'fit_lyot_stop_mag_clocking.yaml')
        lyot_calib_dict = loadyaml(self.fnLyotCalib)
        
        self.data_path = './'

        self.diamBeamAtMask = lyot_calib_dict['diamPupil']
        self.diamHighRes = lyot_design_dict['diamHighResMaskRef']
        strutAnglesNominal = np.arange(45, 360, 90)

        mask_id = 0.40
        mask_od = lyot_design_dict['OD_LS']
        mask_strut_width = 0.03
        
        pupil_strut_width = 0.01

        # Generate pupil
        diamBeam = self.diamBeamAtMask
        nx = 501
        ny = 501
        nSubpixels = 10

        pupil = shapes.simple_pupil(
            nx, ny, self.xOffsetPupil, self.yOffsetPupil,  0.20*diamBeam,
            1.00*diamBeam, strutAngles=strutAnglesNominal,
            strutWidth=pupil_strut_width*diamBeam, nSubpixels=nSubpixels)

        mask = shapes.simple_pupil(
            nx, ny, self.xOffsetPupil+self.xOffsetMask,
            self.yOffsetPupil+self.yOffsetMask,
            mask_id*diamBeam*self.magAtMask, mask_od*diamBeam*self.magAtMask,
            strutAngles=strutAnglesNominal+self.clockDegMask,
            strutWidth=mask_strut_width*diamBeam,
            nSubpixels=nSubpixels)
        
        self.imageUnmasked = pupil**2
        self.imageMasked = (pupil * mask)**2
        
        # Generate high-res reference mask
        self.fnMaskRefHighRes = lyot_design_dict['fnMaskRefHighRes']
        diamBeam = self.diamHighRes
        nx = math.ceil_odd(diamBeam)
        ny = nx

        self.maskRefHighRes = shapes.simple_pupil(
            nx, ny, 0, 0,
            mask_id*diamBeam, mask_od*diamBeam,
            strutAngles=strutAnglesNominal,
            strutWidth=mask_strut_width*diamBeam,
            nSubpixels=nSubpixels)


    def test_fit_lyot_stop_mag_clocking(self):

        # Refer to https://echorand.me/posts/named_temporary_file/ for help
        # with the temporary file usage.
        f = NamedTemporaryFile()
        f.name = self.fnMaskRefHighRes
        hdu = fits.PrimaryHDU(self.maskRefHighRes)
        hdu.writeto(f.name, overwrite=True)        

        magEst, clockEst = maskfit.fit_lyot_stop_mag_clocking(
            self.imageUnmasked, self.imageMasked, self.xOffsetPupil,
            self.yOffsetPupil, self.fnMaskParams, self.fnOffsetParams,
            self.fnLyotCalib, data_path=self.data_path)

        print('magEst = %.3f' % magEst)
        print('clockEst = %.3f' % clockEst)

        self.assertTrue((np.abs(magEst - self.magAtMask)/self.magAtMask*100)
                        <= 0.2,
                        msg=('magEst = %.3f, magTrue = %.3f' %
                             (magEst, self.magAtMask)))  # percent mag error
        self.assertTrue(np.abs(clockEst - self.clockDegMask)*(np.pi/180.)*1e3
                        < 1.5,
                        msg=('clockEst = %.3f, clockTrue = %.3f' %
                             (clockEst, self.clockDegMask)))  # mrad

        # Delete the temporary file
        os.unlink(f.name)


class TestFitPupilMaskOffsets(unittest.TestCase):
    """Unit tests for fit_pupil_mask_offsets and
    fit_shaped_pupil_mask_offsets."""

    def setUp(self):
        
        self.xOffsetPupil = 50.80
        self.yOffsetPupil = -21.37
        
        self.xOffsetMask = 4.90
        self.yOffsetMask = -3.72
        
        self.xOffsetZinduced = 5.5
        self.yOffsetZinduced = -6.8
        
        self.offsetGain = 1
        self.fnTuningParams = os.path.join(LOCAL_PATH, 'testdata',
                                           'fit_pupil_mask_offsets.yaml')

        self.magAtMask = 0.98
        self.clockDegMask = 0.5
        
        self.data_path = './'
        self.fnMaskParams = os.path.join(
            LOCAL_PATH, 'testdata',
            'ut_high_res_pupil_mask_info_dummy.yaml')
        mask_param_dict = loadyaml(self.fnMaskParams)

        self.diamBeamAtMask = 300
        self.diamHighRes = mask_param_dict['diamHighResMaskRef']
        strutAnglesNominal = np.arange(45, 360, 90)

        mask_id = 0.40
        mask_od = 0.80
        mask_strut_width = 0.03

        # Generate pupil
        diamBeam = self.diamBeamAtMask
        nx = 501
        ny = 501
        nSubpixels = 10

        pupil = shapes.simple_pupil(
            nx, ny, self.xOffsetPupil, self.yOffsetPupil,  0.20*diamBeam,
            1.00*diamBeam, strutAngles=strutAnglesNominal,
            strutWidth=0.01*diamBeam, nSubpixels=nSubpixels)

        mask = shapes.simple_pupil(
            nx, ny, self.xOffsetPupil+self.xOffsetMask,
            self.yOffsetPupil+self.yOffsetMask,
            mask_id*diamBeam*self.magAtMask, mask_od*diamBeam*self.magAtMask,
            strutAngles=strutAnglesNominal+self.clockDegMask,
            strutWidth=mask_strut_width*diamBeam,
            nSubpixels=nSubpixels)
        
        self.imageUnmasked = pupil**2
        self.imageMasked = (pupil * mask)**2
        
        pupilZ = shapes.simple_pupil(
            nx, ny, self.xOffsetPupil+self.xOffsetZinduced,
            self.yOffsetPupil+self.yOffsetZinduced, 0.20*diamBeam,
            1.00*diamBeam, strutAngles=strutAnglesNominal,
            strutWidth=0.01*diamBeam, nSubpixels=nSubpixels)

        maskZ = shapes.simple_pupil(
            nx, ny, self.xOffsetPupil+self.xOffsetMask+self.xOffsetZinduced,
            self.yOffsetPupil+self.yOffsetMask+self.yOffsetZinduced,
            mask_id*diamBeam*self.magAtMask, mask_od*diamBeam*self.magAtMask,
            strutAngles=strutAnglesNominal+self.clockDegMask,
            strutWidth=mask_strut_width*diamBeam,
            nSubpixels=nSubpixels)

        self.imageMaskedZ = (pupilZ * maskZ)**2


        # Generate high-res reference mask
        self.fnMaskRefHighRes = mask_param_dict['fnMaskRefHighRes']
        diamBeam = self.diamHighRes
        nx = math.ceil_odd(diamBeam)
        ny = nx

        self.maskRefHighRes = shapes.simple_pupil(
            nx, ny, 0, 0,
            mask_id*diamBeam, mask_od*diamBeam,
            strutAngles=strutAnglesNominal,
            strutWidth=mask_strut_width*diamBeam,
            nSubpixels=nSubpixels)

    def test_fit_shaped_pupil_mask_offsets(self):
        """Performance test of fit_shaped_pupil_mask_offsets."""
        
        abs_tol = 0.2  # pixels

        # Refer to https://echorand.me/posts/named_temporary_file/ for help
        # with the temporary file usage.
        f = NamedTemporaryFile()
        f.name = self.fnMaskRefHighRes
        hdu = fits.PrimaryHDU(self.maskRefHighRes)
        hdu.writeto(f.name, overwrite=True)    

        # Call the function
        xOffsetEst, yOffsetEst = maskfit.fit_shaped_pupil_mask_offsets(
            self.imageUnmasked,
            self.imageMaskedZ,
            self.offsetGain,
            self.fnMaskParams,
            self.fnTuningParams,
            self.xOffsetPupil,
            self.yOffsetPupil,
            self.diamBeamAtMask,
            self.magAtMask,
            self.clockDegMask,
            self.xOffsetZinduced,
            self.yOffsetZinduced,
            data_path=self.data_path,
        )
        
        xOffsetError = self.xOffsetMask - xOffsetEst
        yOffsetError = self.yOffsetMask - yOffsetEst

        self.assertTrue(np.abs(xOffsetError) < abs_tol,
                        msg=('x true / est:  %.3f  %.3f' %
                             (self.xOffsetMask, xOffsetEst)))
        self.assertTrue(np.abs(yOffsetError) < abs_tol,
                        msg=('y true / est:  %.3f  %.3f' %
                             (self.yOffsetMask, yOffsetEst)))
        
        # print('x error = %.6f pixels' % xOffsetError)
        # print('y error = %.6f pixels' % yOffsetError)
        # self.assertTrue(False)

        # Delete the temporary file
        os.unlink(f.name)


    def test_fit_pupil_mask_offsets(self):
        """Performance test of fit_pupil_mask_offsets."""
        
        abs_tol = 0.2  # pixels

        # Refer to https://echorand.me/posts/named_temporary_file/ for help
        # with the temporary file usage.
        f = NamedTemporaryFile()
        f.name = self.fnMaskRefHighRes
        hdu = fits.PrimaryHDU(self.maskRefHighRes)
        hdu.writeto(f.name, overwrite=True)    

        # Call the function
        xOffsetEst, yOffsetEst = maskfit.fit_pupil_mask_offsets(
            self.imageUnmasked,
            self.imageMasked,
            self.offsetGain,
            self.fnMaskParams,
            self.fnTuningParams,
            self.xOffsetPupil,
            self.yOffsetPupil,
            self.diamBeamAtMask,
            self.magAtMask,
            self.clockDegMask,
            data_path=self.data_path,
        )
        
        xOffsetError = self.xOffsetMask - xOffsetEst
        yOffsetError = self.yOffsetMask - yOffsetEst

        self.assertTrue(np.abs(xOffsetError) < abs_tol,
                        msg=('x true / est:  %.3f  %.3f' %
                             (self.xOffsetMask, xOffsetEst)))
        self.assertTrue(np.abs(yOffsetError) < abs_tol,
                        msg=('y true / est:  %.3f  %.3f' %
                             (self.yOffsetMask, yOffsetEst)))
        
        # print('x error = %.6f pixels' % xOffsetError)
        # print('y error = %.6f pixels' % yOffsetError)
        # self.assertTrue(False)

        # Delete the temporary file
        os.unlink(f.name)

    def test_bad_inputs_for_fit_shaped_pupil_mask_offsets(self):
        """Test that exceptions are raised for bad input types and values."""
        
        # Refer to https://echorand.me/posts/named_temporary_file/ for help
        # with the temporary file usage.
        f = NamedTemporaryFile()
        f.name = self.fnMaskRefHighRes
        hdu = fits.PrimaryHDU(self.maskRefHighRes)
        hdu.writeto(f.name, overwrite=True)    
        
        with self.assertRaises(ValueError):
            maskfit.fit_shaped_pupil_mask_offsets(
                np.eye(6),
                np.eye(5),
                self.offsetGain,
                self.fnMaskParams,
                self.fnTuningParams,
                self.xOffsetPupil,
                self.yOffsetPupil,
                self.diamBeamAtMask,
                self.magAtMask,
                self.clockDegMask,
                self.xOffsetZinduced,
                self.yOffsetZinduced,
                data_path=self.data_path,
            )

        for badImage in not_2d_list:
            with self.assertRaises(TypeError):
                maskfit.fit_shaped_pupil_mask_offsets(
                    badImage,
                    self.imageMasked,
                    self.offsetGain,
                    self.fnMaskParams,
                    self.fnTuningParams,
                    self.xOffsetPupil,
                    self.yOffsetPupil,
                    self.diamBeamAtMask,
                    self.magAtMask,
                    self.clockDegMask,
                    self.xOffsetZinduced,
                    self.yOffsetZinduced,
                    data_path=self.data_path,
                )

        for badImage in not_2d_list:
            with self.assertRaises(TypeError):
                maskfit.fit_shaped_pupil_mask_offsets(
                    self.imageUnmasked,
                    badImage,
                    self.offsetGain,
                    self.fnMaskParams,
                    self.fnTuningParams,
                    self.xOffsetPupil,
                    self.yOffsetPupil,
                    self.diamBeamAtMask,
                    self.magAtMask,
                    self.clockDegMask,
                    self.xOffsetZinduced,
                    self.yOffsetZinduced,
                    data_path=self.data_path,
                )

        with self.assertRaises(TypeError):
            maskfit.fit_shaped_pupil_mask_offsets(
                self.imageUnmasked,
                self.imageMasked,
                -2,
                self.fnMaskParams,
                self.fnTuningParams,
                self.xOffsetPupil,
                self.yOffsetPupil,
                self.diamBeamAtMask,
                self.magAtMask,
                self.clockDegMask,
                self.xOffsetZinduced,
                self.yOffsetZinduced,
                data_path=self.data_path,
            )

        with self.assertRaises(TypeError):
            maskfit.fit_shaped_pupil_mask_offsets(
                self.imageUnmasked,
                self.imageMasked,
                self.offsetGain,
                1,
                self.fnTuningParams,
                self.xOffsetPupil,
                self.yOffsetPupil,
                self.diamBeamAtMask,
                self.magAtMask,
                self.clockDegMask,
                self.xOffsetZinduced,
                self.yOffsetZinduced,
                data_path=self.data_path,
            )

        with self.assertRaises(TypeError):
            maskfit.fit_shaped_pupil_mask_offsets(
                self.imageUnmasked,
                self.imageMasked,
                self.offsetGain,
                self.fnMaskParams,
                1,
                self.xOffsetPupil,
                self.yOffsetPupil,
                self.diamBeamAtMask,
                self.magAtMask,
                self.clockDegMask,
                self.xOffsetZinduced,
                self.yOffsetZinduced,
                data_path=self.data_path,
            )

        with self.assertRaises(TypeError):
            maskfit.fit_shaped_pupil_mask_offsets(
                self.imageUnmasked,
                self.imageMasked,
                self.offsetGain,
                self.fnMaskParams,
                self.fnTuningParams,
                [1],
                self.yOffsetPupil,
                self.diamBeamAtMask,
                self.magAtMask,
                self.clockDegMask,
                self.xOffsetZinduced,
                self.yOffsetZinduced,
                data_path=self.data_path,
            )

        with self.assertRaises(TypeError):
            maskfit.fit_shaped_pupil_mask_offsets(
                self.imageUnmasked,
                self.imageMasked,
                self.offsetGain,
                self.fnMaskParams,
                self.fnTuningParams,
                self.xOffsetPupil,
                1j,
                self.diamBeamAtMask,
                self.magAtMask,
                self.clockDegMask,
                self.xOffsetZinduced,
                self.yOffsetZinduced,
                data_path=self.data_path,
            )

        with self.assertRaises(TypeError):
            maskfit.fit_shaped_pupil_mask_offsets(
                self.imageUnmasked,
                self.imageMasked,
                self.offsetGain,
                self.fnMaskParams,
                self.fnTuningParams,
                self.xOffsetPupil,
                self.yOffsetPupil,
                0,
                self.magAtMask,
                self.clockDegMask,
                self.xOffsetZinduced,
                self.yOffsetZinduced,
                data_path=self.data_path,
            )

        with self.assertRaises(TypeError):
            maskfit.fit_shaped_pupil_mask_offsets(
                self.imageUnmasked,
                self.imageMasked,
                self.offsetGain,
                self.fnMaskParams,
                self.fnTuningParams,
                self.xOffsetPupil,
                self.yOffsetPupil,
                self.diamBeamAtMask,
                0,
                self.clockDegMask,
                self.xOffsetZinduced,
                self.yOffsetZinduced,
                data_path=self.data_path,
            )
        
        with self.assertRaises(TypeError):
            maskfit.fit_shaped_pupil_mask_offsets(
                self.imageUnmasked,
                self.imageMasked,
                self.offsetGain,
                self.fnMaskParams,
                self.fnTuningParams,
                self.xOffsetPupil,
                self.yOffsetPupil,
                self.diamBeamAtMask,
                self.magAtMask,
                1j,
                self.xOffsetZinduced,
                self.yOffsetZinduced,
                data_path=self.data_path,
            )

        with self.assertRaises(TypeError):
            maskfit.fit_shaped_pupil_mask_offsets(
                self.imageUnmasked,
                self.imageMasked,
                self.offsetGain,
                self.fnMaskParams,
                self.fnTuningParams,
                self.xOffsetPupil,
                self.yOffsetPupil,
                self.diamBeamAtMask,
                self.magAtMask,
                self.clockDegMask,
                1j,
                self.yOffsetZinduced,
                data_path=self.data_path,
            )
        
        with self.assertRaises(TypeError):
            maskfit.fit_shaped_pupil_mask_offsets(
                self.imageUnmasked,
                self.imageMasked,
                self.offsetGain,
                self.fnMaskParams,
                self.fnTuningParams,
                self.xOffsetPupil,
                self.yOffsetPupil,
                self.diamBeamAtMask,
                self.magAtMask,
                self.clockDegMask,
                self.xOffsetZinduced,
                1j,
                data_path=self.data_path,
            )
        with self.assertRaises(TypeError):
            maskfit.fit_shaped_pupil_mask_offsets(
                self.imageUnmasked,
                self.imageMasked,
                self.offsetGain,
                self.fnMaskParams,
                self.fnTuningParams,
                self.xOffsetPupil,
                self.yOffsetPupil,
                self.diamBeamAtMask,
                self.magAtMask,
                self.clockDegMask,
                self.xOffsetZinduced,
                self.yOffsetZinduced,
                data_path=2,
            )

        # Delete the temporary file
        os.unlink(f.name)


    def test_bad_inputs_for_fit_pupil_mask_offsets(self):
        """Test that exceptions are raised for bad input types and values."""
        
        # Refer to https://echorand.me/posts/named_temporary_file/ for help
        # with the temporary file usage.
        f = NamedTemporaryFile()
        f.name = self.fnMaskRefHighRes
        hdu = fits.PrimaryHDU(self.maskRefHighRes)
        hdu.writeto(f.name, overwrite=True)    
        
        with self.assertRaises(ValueError):
            maskfit.fit_pupil_mask_offsets(
                np.eye(6),
                np.eye(5),
                self.offsetGain,
                self.fnMaskParams,
                self.fnTuningParams,
                self.xOffsetPupil,
                self.yOffsetPupil,
                self.diamBeamAtMask,
                self.magAtMask,
                self.clockDegMask,
                data_path=self.data_path,
            )

        with self.assertRaises(TypeError):
            maskfit.fit_pupil_mask_offsets(
                np.ones((5,)),
                self.imageMasked,
                self.offsetGain,
                self.fnMaskParams,
                self.fnTuningParams,
                self.xOffsetPupil,
                self.yOffsetPupil,
                self.diamBeamAtMask,
                self.magAtMask,
                self.clockDegMask,
                data_path=self.data_path,
            )

        with self.assertRaises(TypeError):
            maskfit.fit_pupil_mask_offsets(
                self.imageUnmasked,
                np.ones((5,)),
                self.offsetGain,
                self.fnMaskParams,
                self.fnTuningParams,
                self.xOffsetPupil,
                self.yOffsetPupil,
                self.diamBeamAtMask,
                self.magAtMask,
                self.clockDegMask,
                data_path=self.data_path,
            )

        with self.assertRaises(TypeError):
            maskfit.fit_pupil_mask_offsets(
                self.imageUnmasked,
                self.imageMasked,
                -2,
                self.fnMaskParams,
                self.fnTuningParams,
                self.xOffsetPupil,
                self.yOffsetPupil,
                self.diamBeamAtMask,
                self.magAtMask,
                self.clockDegMask,
                data_path=self.data_path,
            )

        with self.assertRaises(TypeError):
            maskfit.fit_pupil_mask_offsets(
                self.imageUnmasked,
                self.imageMasked,
                self.offsetGain,
                1,
                self.fnTuningParams,
                self.xOffsetPupil,
                self.yOffsetPupil,
                self.diamBeamAtMask,
                self.magAtMask,
                self.clockDegMask,
                data_path=self.data_path,
            )

        with self.assertRaises(TypeError):
            maskfit.fit_pupil_mask_offsets(
                self.imageUnmasked,
                self.imageMasked,
                self.offsetGain,
                self.fnMaskParams,
                1,
                self.xOffsetPupil,
                self.yOffsetPupil,
                self.diamBeamAtMask,
                self.magAtMask,
                self.clockDegMask,
                data_path=self.data_path,
            )

        with self.assertRaises(TypeError):
            maskfit.fit_pupil_mask_offsets(
                self.imageUnmasked,
                self.imageMasked,
                self.offsetGain,
                self.fnMaskParams,
                self.fnTuningParams,
                [1],
                self.yOffsetPupil,
                self.diamBeamAtMask,
                self.magAtMask,
                self.clockDegMask,
                data_path=self.data_path,
            )

        with self.assertRaises(TypeError):
            maskfit.fit_pupil_mask_offsets(
                self.imageUnmasked,
                self.imageMasked,
                self.offsetGain,
                self.fnMaskParams,
                self.fnTuningParams,
                self.xOffsetPupil,
                1j,
                self.diamBeamAtMask,
                self.magAtMask,
                self.clockDegMask,
                data_path=self.data_path,
            )

        with self.assertRaises(TypeError):
            maskfit.fit_pupil_mask_offsets(
                self.imageUnmasked,
                self.imageMasked,
                self.offsetGain,
                self.fnMaskParams,
                self.fnTuningParams,
                self.xOffsetPupil,
                self.yOffsetPupil,
                0,
                self.magAtMask,
                self.clockDegMask,
                data_path=self.data_path,
            )


        with self.assertRaises(TypeError):
            maskfit.fit_pupil_mask_offsets(
                self.imageUnmasked,
                self.imageMasked,
                self.offsetGain,
                self.fnMaskParams,
                self.fnTuningParams,
                self.xOffsetPupil,
                self.yOffsetPupil,
                self.diamBeamAtMask,
                0,
                self.clockDegMask,
                data_path=self.data_path,
            )

        with self.assertRaises(TypeError):
            maskfit.fit_pupil_mask_offsets(
                self.imageUnmasked,
                self.imageMasked,
                self.offsetGain,
                self.fnMaskParams,
                self.fnTuningParams,
                self.xOffsetPupil,
                self.yOffsetPupil,
                self.diamBeamAtMask,
                self.magAtMask,
                1j,
                data_path=self.data_path,
            )

        with self.assertRaises(TypeError):
            maskfit.fit_pupil_mask_offsets(
                self.imageUnmasked,
                self.imageMasked,
                self.offsetGain,
                self.fnMaskParams,
                self.fnTuningParams,
                self.xOffsetPupil,
                self.yOffsetPupil,
                self.diamBeamAtMask,
                self.magAtMask,
                self.clockDegMask,
                data_path=2,
            )

        # Delete the temporary file
        os.unlink(f.name)


if __name__ == '__main__':
    unittest.main()
