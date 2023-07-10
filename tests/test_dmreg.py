"""Unit tests for the DMREG functions."""
import os
import unittest
import numpy as np
from astropy.io import fits

from coralign.dmreg.dmreg import (
    calc_software_mask,
    calc_translation_clocking,
    calc_scale,
)
from coralign.util.dmhtoph import dmhtoph, volts_to_dmh
from coralign.util.loadyaml import loadyaml
from coralign.util.pad_crop import pad_crop as inin
from coralign.util.shapes import simple_pupil

LOCALPATH = os.path.dirname(os.path.abspath(__file__))


class TestSoftwareMask(unittest.TestCase):
    """Test dm_software_mask."""

    def test_calc_software_mask_shape(self):
        """Test that output shape of calc_software_mask is correct."""
        nrow = 10
        ncol = 11
        amp0 = inin(np.ones((nrow-2, ncol-2)), (nrow, ncol))
        ph0 = np.zeros((nrow, ncol))
        amp1 = inin(np.ones((nrow-2, ncol-2)), (nrow, ncol))
        ph1 = np.ones((nrow, ncol))
        whichDM = 1
        nErode = 1
        deltaPhaseThreshRad = 0.5
        boolMask = calc_software_mask(amp0, ph0, amp1, ph1, whichDM, nErode,
                                      deltaPhaseThreshRad)
        self.assertEqual(boolMask.shape, amp0.shape)

    def test_calc_software_mask_output_values(self):
        """Test that output values of calc_software_mask are correct."""
        nrow = 21
        ncol = 30
        amp0 = inin(np.ones((nrow-2, ncol-2)), (nrow, ncol))
        ph0 = np.zeros((nrow, ncol))
        amp1 = inin(np.ones((nrow-2, ncol-2)), (nrow, ncol))
        ph1 = np.ones((nrow, ncol))
        whichDM = 1
        nErode = 1
        deltaPhaseThreshRad = 0.1
        boolMask = calc_software_mask(amp0, ph0, amp1, ph1, whichDM, nErode,
                                      deltaPhaseThreshRad)
        answer = inin(np.ones((nrow-4, ncol-4)), (nrow, ncol)).astype(int)
        self.assertTrue(np.sum(np.abs(boolMask.astype(int) - answer)) == 0)

        # Test again with a large phase deviation in one area.
        ph1[9:11, 8] = 10  # alter a 2x1 block, which eroded becomes 12 pixels
        boolMask = calc_software_mask(amp0, ph0, amp1, ph1, whichDM, nErode,
                                      deltaPhaseThreshRad)
        boolMask = boolMask.astype(int)
        self.assertTrue(np.sum(np.abs(boolMask - answer))-12 == 0)

    def test_inputs_calc_software_mask(self):
        """Test the inputs of calc_software_mask."""
        nrow = 10
        ncol = 11
        amp0 = inin(np.ones((nrow-2, ncol-2)), (nrow, ncol))
        ph0 = np.zeros((nrow, ncol))
        amp1 = inin(np.ones((nrow-2, ncol-2)), (nrow, ncol))
        ph1 = np.ones((nrow, ncol))
        whichDM = 1
        nErode = 1
        deltaPhaseThreshRad = 0.5

        # Check standard inputs do not raise anything first
        calc_software_mask(amp0, ph0, amp1, ph1, whichDM, nErode,
                           deltaPhaseThreshRad)

        for amp0Bad in (-1, 1, 1.1, 1j, np.ones((5, )), 'string'):
            with self.assertRaises(ValueError):
                calc_software_mask(amp0Bad, ph0, amp1, ph1, whichDM, nErode,
                                   deltaPhaseThreshRad)
        for ph0Bad in (-1, 1, 1.1, 1j, np.ones((5, )), 'string'):
            with self.assertRaises(ValueError):
                calc_software_mask(amp0, ph0Bad, amp1, ph1, whichDM, nErode,
                                   deltaPhaseThreshRad)
        for amp1Bad in (-1, 1, 1.1, 1j, np.ones((5, )), 'string'):
            with self.assertRaises(ValueError):
                calc_software_mask(amp0, ph0, amp1Bad, ph1, whichDM, nErode,
                                   deltaPhaseThreshRad)
        for ph1Bad in (-1, 1, 1.1, 1j, np.ones((5, )), 'string'):
            with self.assertRaises(ValueError):
                calc_software_mask(amp0, ph0, amp1, ph1Bad, whichDM, nErode,
                                   deltaPhaseThreshRad)
        for whichDMBad in (-1, 0, 1.1, 3, 1j, np.ones((1, )), 'string'):
            with self.assertRaises(ValueError):
                calc_software_mask(amp0, ph0, amp1, ph1, whichDMBad, nErode,
                                   deltaPhaseThreshRad)
        for nErodeBad in (-1, 1.1, 1j, np.ones(1), np.ones((4, 5)), 'string'):
            with self.assertRaises(ValueError):
                calc_software_mask(amp0, ph0, amp1, ph1, whichDM, nErodeBad,
                                   deltaPhaseThreshRad)
        for deltaPhaseThreshRadBad in (-1, 0, 1j, np.ones(1),
                                       np.ones((4, 5)), 'string'):
            with self.assertRaises(ValueError):
                calc_software_mask(amp0, ph0, amp1, ph1, whichDM, nErode,
                                   deltaPhaseThreshRadBad)

        # Shape equivalency tests for amp0, ph0, amp1, ph1
        wrong2D = np.zeros((8, 9))
        with self.assertRaises(ValueError):
            calc_software_mask(wrong2D, ph0, amp1, ph1, whichDM, nErode,
                               deltaPhaseThreshRad)
        with self.assertRaises(ValueError):
            calc_software_mask(amp0, wrong2D, amp1, ph1, whichDM, nErode,
                               deltaPhaseThreshRad)
        with self.assertRaises(ValueError):
            calc_software_mask(amp0, ph0, wrong2D, ph1, whichDM, nErode,
                               deltaPhaseThreshRad)
        with self.assertRaises(ValueError):
            calc_software_mask(amp0, ph0, amp1, wrong2D, whichDM, nErode,
                               deltaPhaseThreshRad)


class TestTranslationRotation(unittest.TestCase):
    """Test calc_translation_clocking."""

    def setUp(self):
        self.dm_fn = 'testdata/ut_dm_for_dmreg.yaml'
        dm = loadyaml(self.dm_fn)
        nact = dm['dms']['DM1']['registration']['nact']
        ppact_d = dm['dms']['DM1']['registration']['ppact_d']
        inffn = dm['dms']['DM1']['registration']['inffn']
        # ppact_cx = dm['dms']['DM1']['registration']['ppact_cx']
        # ppact_cy = dm['dms']['DM1']['registration']['ppact_cy']
        ppact_d = dm['dms']['DM1']['registration']['ppact_d']
        flipx = dm['dms']['DM1']['registration']['flipx']
        gainfn = dm['dms']['DM1']['voltages']['gainfn']
        inf_func = fits.getdata(os.path.join(LOCALPATH, inffn))
        gainmap = fits.getdata(os.path.join(LOCALPATH, gainfn))
        
        self.fn_fit_params = 'testdata/ut_trans_rot_settings.yaml'
        self.xOffsetPupil = 90
        self.yOffsetPupil = -33
        self.lam = 575e-9
        
        self.xOffsetTrue = 86.2
        self.yOffsetTrue = -30.9
        self.clockingTrue = 1.5
        ppact_cx_true = 6.83
        ppact_cy_true = 6.54
        dx = self.xOffsetTrue
        dy = self.yOffsetTrue
        thact = self.clockingTrue
        ppact_cx = ppact_cx_true
        ppact_cy = ppact_cy_true

        # Generate pupil mask
        diamPupilPix = 300
        nx = 500
        ny = 500
        xOffset = self.xOffsetPupil
        yOffset = self.yOffsetPupil
        diamInner = 0.30 * diamPupilPix
        diamOuter = 1.00 * diamPupilPix
        strutAngles = np.arange(45, 360, 90)
        strutWidth = 0.02 * diamPupilPix
        pupil = simple_pupil(nx, ny, xOffset, yOffset, diamInner, diamOuter,
                             strutAngles=strutAngles, strutWidth=strutWidth,
                             nSubpixels=100)

        # This is the same DM command used to make amp1 and ph1: a 2-striped +
        fn_deltaV = os.path.join(LOCALPATH, 'testdata',
                                'delta_DM_V_for_trans_rot_calib.fits')
        self.deltaV = fits.getdata(fn_deltaV)
        dmrad = volts_to_dmh(gainmap, self.deltaV, self.lam)

        self.amp0 = pupil
        self.amp1 = pupil
        self.usablePixelMap = np.floor(pupil)
        self.ph0 = np.zeros_like(pupil)
        self.ph1 = dmhtoph(ny, nx, dmrad, nact, inf_func, ppact_d, ppact_cx,
                           ppact_cy, dx, dy, thact, flipx)
        
        self.whichDM = 1
        

    def test_calc_translation_clocking(self):
        """Test clocking error and translation error of DM registration."""

        tol_offset = 0.1  # pixels
        tol_clocking = 0.25  # mrad

        xOffsetEst, yOffsetEst, clockEst = calc_translation_clocking(
            self.amp0, self.ph0, self.amp1, self.ph1, self.usablePixelMap,
            self.whichDM, self.deltaV,
            self.xOffsetPupil, self.yOffsetPupil, self.dm_fn, self.fn_fit_params, self.lam,
            data_path=LOCALPATH,
        )

        self.assertTrue(np.abs(self.xOffsetTrue - xOffsetEst) < tol_offset)
        self.assertTrue(np.abs(self.yOffsetTrue - yOffsetEst) < tol_offset)
        self.assertTrue((self.clockingTrue - clockEst)*np.pi/180*1000 < tol_clocking)

    def test_inputs_calc_translation_clocking(self):
        """Test the inputs of calc_translation_clocking."""
        whichDM = self.whichDM
        amp0 = self.amp0
        amp1 = self.amp1
        ph0 = self.ph0
        ph1 = self.ph1
        usablePixelMap = self.usablePixelMap
        dm_fn = self.dm_fn
        fn_fit_params = self.fn_fit_params
        xOffsetPupil = self.xOffsetPupil
        yOffsetPupil = self.yOffsetPupil
        lam = self.lam
        deltaV = self.deltaV

        # Bad input tests
        for amp0Bad in (-1, 1, 1.1, 1j, np.ones((5, )), 'string'):
            with self.assertRaises(ValueError):
                calc_translation_clocking(amp0Bad, ph0, amp1, ph1,
                          usablePixelMap, whichDM, deltaV, xOffsetPupil,
                          yOffsetPupil, dm_fn, fn_fit_params, lam)
        for ph0Bad in (-1, 1, 1.1, 1j, np.ones((5, )), 'string'):
            with self.assertRaises(ValueError):
                calc_translation_clocking(amp0, ph0Bad, amp1, ph1,
                          usablePixelMap, whichDM, deltaV, xOffsetPupil,
                          yOffsetPupil, dm_fn, fn_fit_params, lam)
        for amp1Bad in (-1, 1, 1.1, 1j, np.ones((5, )), 'string'):
            with self.assertRaises(ValueError):
                calc_translation_clocking(amp0, ph0, amp1Bad, ph1,
                          usablePixelMap, whichDM, deltaV, xOffsetPupil,
                          yOffsetPupil, dm_fn, fn_fit_params, lam)
        for ph1Bad in (-1, 1, 1.1, 1j, np.ones((5, )), 'string'):
            with self.assertRaises(ValueError):
                calc_translation_clocking(amp0, ph0, amp1, ph1Bad,
                          usablePixelMap, whichDM, deltaV, xOffsetPupil,
                          yOffsetPupil, dm_fn, fn_fit_params, lam)
        for usablePixelMapBad in (-1, 1, 1.1, 1j, np.ones((5, )), 'string'):
            with self.assertRaises(ValueError):
                calc_translation_clocking(amp0, ph0, amp1, ph1,
                          usablePixelMapBad, whichDM, deltaV, xOffsetPupil,
                          yOffsetPupil, dm_fn, fn_fit_params, lam)

        for whichDMBad in (-1, 0, 1.1, 3, 1j, np.ones((1, )), 'string'):
            with self.assertRaises(ValueError):
                calc_translation_clocking(amp0, ph0, amp1, ph1,
                          usablePixelMap, whichDMBad, deltaV, xOffsetPupil,
                          yOffsetPupil, dm_fn, fn_fit_params, lam)
        for deltaVBad in (-1, 1, 1.1, 1j, np.ones(5),
                          np.ones((47, 48)), 'string'):
            with self.assertRaises(ValueError):
                calc_translation_clocking(amp0, ph0, amp1, ph1,
                          usablePixelMap, whichDM, deltaVBad, xOffsetPupil,
                          yOffsetPupil, dm_fn, fn_fit_params, lam)
        for xOffsetBad in (1j, np.ones(1), np.ones((4, 5)), 'string'):
            with self.assertRaises(ValueError):
                calc_translation_clocking(amp0, ph0, amp1, ph1,
                          usablePixelMap, whichDM, deltaV, xOffsetBad,
                          yOffsetPupil, dm_fn, fn_fit_params, lam)
        for yOffsetBad in (1j, np.ones((4, 5)), 'string'):
            with self.assertRaises(ValueError):
                calc_translation_clocking(amp0, ph0, amp1, ph1,
                          usablePixelMap, whichDM, deltaV, xOffsetPupil,
                          yOffsetBad, dm_fn, fn_fit_params, lam)
        for lamBad in (-1, 0, 1j, np.ones(1), np.ones((4, 5)), 'string'):
            with self.assertRaises(ValueError):
                calc_translation_clocking(amp0, ph0, amp1, ph1,
                          usablePixelMap, whichDM, deltaV, xOffsetPupil,
                          yOffsetPupil, dm_fn, fn_fit_params, lamBad)

        # Shape equivalency tests for amp0, ph0, amp1, ph1, usablePixelMap
        wrong2D = np.zeros((8, 9))
        with self.assertRaises(ValueError):
            calc_translation_clocking(wrong2D, ph0, amp1, ph1,
                          usablePixelMap, whichDM, deltaV, xOffsetPupil,
                          yOffsetPupil, dm_fn, fn_fit_params, lam)
        with self.assertRaises(ValueError):
            calc_translation_clocking(amp0, wrong2D, amp1, ph1,
                          usablePixelMap, whichDM, deltaV, xOffsetPupil,
                          yOffsetPupil, dm_fn, fn_fit_params, lam)
        with self.assertRaises(ValueError):
            calc_translation_clocking(amp0, ph0, wrong2D, ph1,
                          usablePixelMap, whichDM, deltaV, xOffsetPupil,
                          yOffsetPupil, dm_fn, fn_fit_params, lam)
        with self.assertRaises(ValueError):
            calc_translation_clocking(amp0, ph0, amp1, wrong2D,
                          usablePixelMap, whichDM, deltaV, xOffsetPupil,
                          yOffsetPupil, dm_fn, fn_fit_params, lam)
        with self.assertRaises(ValueError):
            calc_translation_clocking(amp0, ph0, amp1, ph1,
                          wrong2D, whichDM, deltaV, xOffsetPupil,
                          yOffsetPupil, dm_fn, fn_fit_params, lam)


class TestScale(unittest.TestCase):
    """Test calc_scale."""

    def setUp(self):
        self.dm_fn = 'testdata/ut_dm_for_dmreg.yaml'
        dm = loadyaml(self.dm_fn)
        nact = dm['dms']['DM1']['registration']['nact']
        ppact_d = dm['dms']['DM1']['registration']['ppact_d']
        inffn = dm['dms']['DM1']['registration']['inffn']
        ppact_cx = dm['dms']['DM1']['registration']['ppact_cx']
        ppact_cy = dm['dms']['DM1']['registration']['ppact_cy']
        ppact_d = dm['dms']['DM1']['registration']['ppact_d']
        flipx = dm['dms']['DM1']['registration']['flipx']
        gainfn = dm['dms']['DM1']['voltages']['gainfn']
        inf_func = fits.getdata(os.path.join(LOCALPATH, inffn))
        gainmap = fits.getdata(os.path.join(LOCALPATH, gainfn))
        
        self.fn_fit_params = 'testdata/ut_scale_settings.yaml'
        xOffsetPupil = 90
        yOffsetPupil = -33
        self.lam = 575e-9
        
        self.xOffsetTrue = 86.2
        self.yOffsetTrue = -30.9
        self.clockingTrue = 1.5
        self.ppact_cx_true = ppact_cx #* 1.03  # 6.700
        self.ppact_cy_true = ppact_cy #* 1.02  # 6.541
        dx = self.xOffsetTrue
        dy = self.yOffsetTrue
        thact = self.clockingTrue

        # Generate pupil mask
        diamPupilPix = 300
        nx = 500
        ny = 500
        self.xOffset = xOffsetPupil
        self.yOffset = yOffsetPupil
        diamInner = 0.30 * diamPupilPix
        diamOuter = 1.00 * diamPupilPix
        strutAngles = np.arange(45, 360, 90)
        strutWidth = 0.02 * diamPupilPix
        pupil = simple_pupil(nx, ny, self.xOffset, self.yOffset, diamInner, diamOuter,
                             strutAngles=strutAngles, strutWidth=strutWidth,
                             nSubpixels=100)       

        # Same DM command used to make amp1 and ph1: a square outline
        fn_deltaV = os.path.join(LOCALPATH, 'testdata',
                                'delta_DM_V_for_scale_calib.fits')
        self.deltaV = fits.getdata(fn_deltaV)
        dmrad = volts_to_dmh(gainmap, self.deltaV, self.lam)

        self.amp0 = pupil
        self.amp1 = pupil
        self.usablePixelMap = np.floor(pupil)
        self.ph0 = np.zeros_like(pupil)
        self. ph1 = 2 * dmhtoph(
            ny, nx, dmrad, nact, inf_func, ppact_d, self.ppact_cx_true,
            self.ppact_cy_true, dx, dy, thact, flipx,
        )

        self.whichDM = 1

    def test_calc_scale(self):
        """Test the scale estimation of DM registration."""
        tol_scale = 0.2  # percent

        ppact_cx_est, ppact_cy_est = calc_scale(
            self.amp0, self.ph0, self.amp1, self.ph1, self.usablePixelMap,
            self.whichDM, self.deltaV, self.dm_fn, self.fn_fit_params,
            self.lam, self.xOffsetTrue, self.yOffsetTrue, self.clockingTrue,
            data_path=LOCALPATH,
        )

        print('ppact_cx_est = %.4f' % ppact_cx_est)
        print('ppact_cy_est = %.4f' % ppact_cy_est)
        self.assertTrue(np.abs(self.ppact_cx_true - ppact_cx_est) /
                        self.ppact_cx_true * 100 < tol_scale)
        self.assertTrue(np.abs(self.ppact_cy_true - ppact_cy_est) /
                        self.ppact_cy_true * 100 < tol_scale)

    def test_inputs_calc_scale(self):
        """Test the inputs of calc_scale."""
        whichDM = self.whichDM
        amp0 = self.amp0
        amp1 = self.amp1
        ph0 = self.ph0
        ph1 = self.ph1
        usablePixelMap = self.usablePixelMap
        dm_fn = self.dm_fn
        fn_fit_params = self.fn_fit_params
        xOffset = self.xOffset
        yOffset = self.yOffset
        clocking = self.clockingTrue
        lam = self.lam
        deltaV = self.deltaV

        # Bad input tests
        for amp0Bad in (-1, 1, 1.1, 1j, np.ones((5, )), 'string'):
            with self.assertRaises(ValueError):
                calc_scale(amp0Bad, ph0, amp1, ph1, usablePixelMap, whichDM,
                           deltaV, dm_fn, fn_fit_params, lam,
                           xOffset, yOffset, clocking)
        for ph0Bad in (-1, 1, 1.1, 1j, np.ones((5, )), 'string'):
            with self.assertRaises(ValueError):
                calc_scale(amp0, ph0Bad, amp1, ph1, usablePixelMap, whichDM,
                           deltaV, dm_fn, fn_fit_params, lam,
                        xOffset, yOffset, clocking)
        for amp1Bad in (-1, 1, 1.1, 1j, np.ones((5, )), 'string'):
            with self.assertRaises(ValueError):
                calc_scale(amp0, ph0, amp1Bad, ph1, usablePixelMap, whichDM,
                           deltaV, dm_fn, fn_fit_params, lam,
                           xOffset, yOffset, clocking)
        for ph1Bad in (-1, 1, 1.1, 1j, np.ones((5, )), 'string'):
            with self.assertRaises(ValueError):
                calc_scale(amp0, ph0, amp1, ph1Bad, usablePixelMap, whichDM,
                           deltaV, dm_fn, fn_fit_params, lam,
                           xOffset, yOffset, clocking)
        for usablePixelMapBad in (-1, 1, 1.1, 1j, np.ones((5, )), 'string'):
            with self.assertRaises(ValueError):
                calc_scale(amp0, ph0, amp1, ph1, usablePixelMapBad, whichDM,
                           deltaV, dm_fn, fn_fit_params, lam,
                           xOffset, yOffset, clocking)
        for whichDMBad in (-1, 0, 1.1, 3, 1j, np.ones((1, )), 'string'):
            with self.assertRaises(ValueError):
                calc_scale(amp0, ph0, amp1, ph1, usablePixelMap, whichDMBad,
                           deltaV, dm_fn, fn_fit_params, lam,
                           xOffset, yOffset, clocking)
        for deltaVBad in (-1, 1, 1.1, 1j, np.ones(5),
                          np.ones((47, 48)), 'string'):
            with self.assertRaises(ValueError):
                calc_scale(amp0, ph0, amp1, ph1, usablePixelMap, whichDM,
                           deltaVBad, dm_fn, fn_fit_params, lam,
                           xOffset, yOffset, clocking)
        for lamBad in (-1, 0, 1j, np.ones(1), np.ones((4, 5)), 'string'):
            with self.assertRaises(ValueError):
                calc_scale(amp0, ph0, amp1, ph1, usablePixelMap, whichDM,
                           deltaV, dm_fn, fn_fit_params, lamBad,
                           xOffset, yOffset, clocking)
        for xOffsetBad in (1j, np.ones(1), np.ones((4, 5)), 'string'):
            with self.assertRaises(ValueError):
                calc_scale(amp0, ph0, amp1, ph1, usablePixelMap, whichDM,
                           deltaV, dm_fn, fn_fit_params, lam,
                           xOffsetBad, yOffset, clocking)
        for yOffsetBad in (1j, np.ones((4, 5)), 'string'):
            with self.assertRaises(ValueError):
                calc_scale(amp0, ph0, amp1, ph1, usablePixelMap, whichDM,
                           deltaV, dm_fn, fn_fit_params, lam,
                           xOffset, yOffsetBad, clocking)
        for clockingBad in (1j, np.ones(1), np.ones((4, 5)), 'string'):
            with self.assertRaises(ValueError):
                calc_scale(amp0, ph0, amp1, ph1, usablePixelMap, whichDM,
                           deltaV, dm_fn, fn_fit_params, lam,
                           xOffset, yOffset, clockingBad)

        # Shape equivalency tests for amp0, ph0, amp1, ph1, usablePixelMap
        wrong2D = np.zeros((8, 9))
        with self.assertRaises(ValueError):
            calc_scale(wrong2D, ph0, amp1, ph1, usablePixelMap, whichDM,
                       deltaV, dm_fn, fn_fit_params, lam,
                       xOffset, yOffset, clocking)
        with self.assertRaises(ValueError):
            calc_scale(amp0, wrong2D, amp1, ph1, usablePixelMap, whichDM,
                       deltaV, dm_fn, fn_fit_params, lam,
                       xOffset, yOffset, clocking)
        with self.assertRaises(ValueError):
            calc_scale(amp0, ph0, wrong2D, ph1, usablePixelMap, whichDM,
                       deltaV, dm_fn, fn_fit_params, lam,
                       xOffset, yOffset, clocking)
        with self.assertRaises(ValueError):
            calc_scale(amp0, ph0, amp1, wrong2D, usablePixelMap, whichDM,
                       deltaV, dm_fn, fn_fit_params, lam,
                       xOffset, yOffset, clocking)
        with self.assertRaises(ValueError):
            calc_scale(amp0, ph0, amp1, ph1, wrong2D, whichDM,
                       deltaV, dm_fn, fn_fit_params, lam,
                       xOffset, yOffset, clocking)


if __name__ == '__main__':
    unittest.main()
