"""Unit tests for the CONJUGATE module."""
import unittest
import os
from math import isclose
import numpy as np
from astropy.io import fits

from coralign.util import shapes
from coralign.util.ampthresh import ampthresh
from coralign.util.nollzernikes import xyzern, gen_zernikes
from coralign.util.pad_crop import pad_crop as inin
from coralign.util.dmhtoph import dmhtoph, volts_to_dmh, dmh_to_volts
from coralign.util.loadyaml import loadyaml
from coralign.util.math import ceil_even, rms
from coralign.conjugate.conjugate import (
    flatten, gen_split_wfe, smooth_surface, conv_surf_to_dm_cmd,
    phase_cost_function
)

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))


class TestFlatten(unittest.TestCase):
    """Test functionality and inputs of flatten()."""

    def setUp(self):
        """Initialize variables used in all the functional tests."""
        self.localpath = os.path.dirname(os.path.abspath(__file__))
        self.dm_fn = os.path.join(self.localpath, 'testdata', 'dm.yaml')
        self.fcm_fn = os.path.join(self.localpath, 'testdata', 'fcm.yaml')
        self.flatten_params_fn = os.path.join(
            self.localpath, 'testdata', 'fixed_inputs_for_conjugate.yaml')

        # pupil calibration data
        self.diamPupil = 386.
        self.xOffset = 0.
        self.yOffset = 0.

        # Load FCM Calibration Data
        fcmConfig = loadyaml(self.fcm_fn)
        self.focusPerStroke = fcmConfig["focusPerStroke"]
        self.strokePerStep = fcmConfig["strokePerStep"]

        # Load fixed settings for CONJUGATE
        flattenDict = loadyaml(self.flatten_params_fn)
        arrayExtFac = flattenDict['arrayExtFac']
        self.nArray = ceil_even(arrayExtFac*self.diamPupil)

        # Load DM Calibration Data to get "nact" and make total voltage maps
        self.dm_fn = os.path.join(self.localpath, 'testdata', 'dm.yaml')
        dmConfig = loadyaml(self.dm_fn)
        dm = dmConfig["dms"]
        dm1r = dm['DM1']['registration']
        dm2r = dm['DM2']['registration']
        self.dm1r = dm1r
        self.dm2r = dm2r
        dm1gainmap_fn = os.path.join(self.localpath,
                                     dm['DM1']['voltages']['gainfn'])
        self.dm1gainmap = fits.getdata(dm1gainmap_fn)
        dm2gainmap_fn = os.path.join(self.localpath,
                                     dm['DM2']['voltages']['gainfn'])
        self.dm2gainmap = fits.getdata(dm2gainmap_fn)

        self.Vdm1In = 50.*np.ones((dm1r['nact'], dm1r['nact']))
        self.Vdm2In = 50.*np.ones((dm2r['nact'], dm2r['nact']))
        self.cmdFCMIn = 1000

        self.lam = 575e-9  # used only for height-to-phase or vice-versa
        self.dm1inffn = os.path.join(self.localpath, dm1r['inffn'])
        self.dm2inffn = os.path.join(self.localpath, dm2r['inffn'])

        # ph_fn = os.path.join(
        #     self.localpath, 'testdata',ph_spm_smallZ4_N386
        #     'ph_cgi_entrance_phase_b_N386_radians_575nm.fits')
        # amp_fn = os.path.join(
        #     self.localpath, 'testdata',
        #     'amp_cgi_entrance_phase_b_N386_radians_575nm.fits')
        # self.ph = fits.getdata(ph_fn)
        # self.amp = fits.getdata(amp_fn)

        nBeam = 386
        nx = self.nArray
        ny = nx
        xOffset = 0.
        yOffset = 0.
        diamInner = 0.20*nBeam
        diamOuter = 1.00*nBeam
        strutAngles = np.array([0, 90, 180, 270]) + 15
        strutWidth = 0.032*nBeam
        self.amp = shapes.simple_pupil(
            nx, ny, xOffset, yOffset, diamInner, diamOuter,
            strutAngles=strutAngles, strutWidth=strutWidth)

        zernIndVec = np.arange(4, 30)
        rng = np.random.default_rng(55555)
        zernCoefVec = rng.standard_normal(zernIndVec.shape) *\
            (40e-9 / self.lam)

        self.ph = (gen_zernikes(zernIndVec, zernCoefVec, self.xOffset,
                                self.yOffset, nBeam, nArray=nx) *
                   ampthresh(self.amp))

        self.flagUseFCM = True
        self.flagUseDMs = True
        self.flagHighZern = True

    def test_first_flattening_on_orbit(self):
        """Flatten the wavefront for the first time on orbit."""
        flagUseFCM = True
        flagUseDMs = True
        flagHighZern = True

        Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
            self.Vdm1In, self.Vdm2In, self.cmdFCMIn, flagUseDMs, flagUseFCM,
            flagHighZern, self.lam, self.amp, self.ph,
            self.diamPupil, self.xOffset, self.yOffset,
            self.dm_fn, self.fcm_fn, self.flatten_params_fn,
            data_path=LOCAL_PATH)

        # Make WFE maps for Z1,2,3 and Z4
        nArray = self.nArray
        wfeZ1to3 = gen_zernikes(np.array([1, 2, 3]), zernCoefVec[0:3],
                                self.xOffset, self.yOffset,
                                self.diamPupil, nArray=nArray)
        wfeZ4 = gen_zernikes(np.array([4]), np.array([zernCoefVec[3]],),
                             self.xOffset, self.yOffset,
                             self.diamPupil, nArray=nArray)

        # Verify that the output conjugates the input
        usablePixelMap = ampthresh(self.amp)
        usablePixelMapPad = inin(usablePixelMap, (nArray, nArray))
        wfeMeasPad = inin(self.ph, (nArray, nArray))

        deltaV1 = Vdm1Out-self.Vdm1In
        deltaV2 = Vdm2Out-self.Vdm2In
        dm1phase = volts_to_dmh(self.dm1gainmap, deltaV1, self.lam)
        dm2phase = volts_to_dmh(self.dm2gainmap, deltaV2, self.lam)

        dm1r = self.dm1r
        DM1SurfActual = dmhtoph(
            nArray, nArray, dm1phase, dm1r['nact'],
            fits.getdata(self.dm1inffn), dm1r['ppact_d'],
            dm1r['ppact_cx'], dm1r['ppact_cy'], dm1r['dx'],
            dm1r['dy'], dm1r['thact'], dm1r['flipx'])

        dm2r = self.dm2r
        DM2SurfActual = dmhtoph(
            nArray, nArray, dm2phase, dm2r['nact'],
            fits.getdata(self.dm2inffn), dm2r['ppact_d'],
            dm2r['ppact_cx'], dm2r['ppact_cy'], dm2r['dx'],
            dm2r['dy'], dm2r['thact'], dm2r['flipx'])

        wfeResidual = usablePixelMapPad*(wfeMeasPad - (wfeZ1to3 + wfeZ4) +
                                         2*(DM1SurfActual + DM2SurfActual))
        rmsError = np.sqrt(np.mean(wfeResidual[usablePixelMapPad == 1]**2))
        # print('CONJUGATE: RMS WFE residual = %.3g rad' % rmsError)
        self.assertTrue(rmsError < 20e-9*(2*np.pi/self.lam))

    def test_first_flattening_on_orbit_flipx(self):
        """Flatten the wavefront for the first time on orbit."""
        flagUseFCM = True
        flagUseDMs = True
        flagHighZern = True

        dm_fn = os.path.join(self.localpath, 'testdata', 'dm_flipx.yaml')
        dmConfig = loadyaml(dm_fn)
        dm = dmConfig["dms"]
        dm1r = dm['DM1']['registration']
        dm2r = dm['DM2']['registration']
        dm1inffn = os.path.join(self.localpath, dm1r['inffn'])
        dm2inffn = os.path.join(self.localpath, dm2r['inffn'])
        dm1gainmap_fn = os.path.join(self.localpath,
                                     dm['DM1']['voltages']['gainfn'])
        dm1gainmap = fits.getdata(dm1gainmap_fn)
        dm2gainmap_fn = os.path.join(self.localpath,
                                     dm['DM2']['voltages']['gainfn'])
        dm2gainmap = fits.getdata(dm2gainmap_fn)

        Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
            self.Vdm1In, self.Vdm2In, self.cmdFCMIn, flagUseDMs, flagUseFCM,
            flagHighZern, self.lam, self.amp, self.ph,
            self.diamPupil, self.xOffset, self.yOffset,
            dm_fn, self.fcm_fn, self.flatten_params_fn, data_path=LOCAL_PATH)

        # Make WFE maps for Z1,2,3 and Z4
        nArray = self.nArray
        wfeZ1to3 = gen_zernikes(np.array([1, 2, 3]), zernCoefVec[0:3],
                                self.xOffset, self.yOffset,
                                self.diamPupil, nArray=nArray)
        wfeZ4 = gen_zernikes(np.array([4]), np.array([zernCoefVec[3]],),
                             self.xOffset, self.yOffset,
                             self.diamPupil, nArray=nArray)

        # Verify that the output conjugates the input
        usablePixelMap = ampthresh(self.amp)
        usablePixelMapPad = inin(usablePixelMap, (nArray, nArray))
        wfeMeasPad = inin(self.ph, (nArray, nArray))

        deltaV1 = Vdm1Out-self.Vdm1In
        deltaV2 = Vdm2Out-self.Vdm2In
        dm1phase = volts_to_dmh(dm1gainmap, deltaV1, self.lam)
        dm2phase = volts_to_dmh(dm2gainmap, deltaV2, self.lam)

        # dm1r = self.dm1r
        DM1SurfActual = dmhtoph(
            nArray, nArray, dm1phase, dm1r['nact'],
            fits.getdata(dm1inffn), dm1r['ppact_d'],
            dm1r['ppact_cx'], dm1r['ppact_cy'], dm1r['dx'],
            dm1r['dy'], dm1r['thact'], dm1r['flipx'])

        # dm2r = self.dm2r
        DM2SurfActual = dmhtoph(
            nArray, nArray, dm2phase, dm2r['nact'],
            fits.getdata(dm2inffn), dm2r['ppact_d'],
            dm2r['ppact_cx'], dm2r['ppact_cy'], dm2r['dx'],
            dm2r['dy'], dm2r['thact'], dm2r['flipx'])

        wfeResidual = usablePixelMapPad*(wfeMeasPad - (wfeZ1to3 + wfeZ4) +
                                         2*(DM1SurfActual + DM2SurfActual))
        rmsError = np.sqrt(np.mean(wfeResidual[usablePixelMapPad == 1]**2))
        # print('CONJUGATE: RMS WFE residual = %.3g rad' % rmsError)
        self.assertTrue(rmsError < 20e-9*(2*np.pi/self.lam))

    def test_equal_dm1_dm2_allocations(self):
        """With >Z11 turned off, make sure DM1 and DM2 split the WFE evenly."""
        flagUseFCM = True
        flagUseDMs = True
        flagHighZern = False

        Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
            self.Vdm1In, self.Vdm2In, self.cmdFCMIn, flagUseDMs, flagUseFCM,
            flagHighZern, self.lam, self.amp, self.ph,
            self.diamPupil, self.xOffset, self.yOffset,
            self.dm_fn, self.fcm_fn, self.flatten_params_fn,
            data_path=LOCAL_PATH)

        deltaV1 = Vdm1Out-self.Vdm1In
        deltaV2 = Vdm2Out-self.Vdm2In
        maxAbsSum = np.max(np.abs(deltaV1 - deltaV2))
        # print('CONJUGATE: Max difference in DM1 and DM2 commands = %.3g'
        #       % maxAbsSum)
        abs_tol = 1000*np.finfo(float).eps
        self.assertTrue(maxAbsSum < abs_tol)

    def test_equal_dm1_dm2_allocations_flipx(self):
        """With >Z11 turned off, make sure DM1 and DM2 split the WFE evenly."""
        flagUseFCM = True
        flagUseDMs = True
        flagHighZern = False

        dm_fn = os.path.join(self.localpath, 'testdata', 'dm_flipx.yaml')

        Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
            self.Vdm1In, self.Vdm2In, self.cmdFCMIn, flagUseDMs, flagUseFCM,
            flagHighZern, self.lam, self.amp, self.ph,
            self.diamPupil, self.xOffset, self.yOffset,
            dm_fn, self.fcm_fn, self.flatten_params_fn, data_path=LOCAL_PATH)

        deltaV1 = Vdm1Out-self.Vdm1In
        deltaV2 = Vdm2Out-self.Vdm2In
        maxAbsSum = np.max(np.abs(np.fliplr(deltaV1) - deltaV2))
        abs_tol = 1000*np.finfo(float).eps
        self.assertTrue(maxAbsSum < abs_tol)

    def test_smallZ4_SPMs(self):
        """Flatten the WFE after moving in an SPM. Small focus change case."""
        flagUseFCM = False
        flagUseDMs = True
        flagHighZern = False

        rng = np.random.default_rng(21)
        zernIndVec = np.arange(4, 12)
        zernCoefVec = rng.standard_normal(zernIndVec.shape) *\
            (3e-9 / self.lam)
        zernCoefVec[0] = 3e-9 / self.lam

        nBeam = 386
        wfeMeas = gen_zernikes(zernIndVec, zernCoefVec, self.xOffset,
                               self.yOffset, nBeam, nArray=self.nArray)

        # wfeMeas = fits.getdata(os.path.join(
        #     self.localpath, 'testdata', 'ph_spm_smallZ4_N386.fits'))
        # amp = fits.getdata(os.path.join(
        #     self.localpath, 'testdata', 'SPM_SPC-20200617_386.fits'))

        Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
            self.Vdm1In, self.Vdm2In, self.cmdFCMIn, flagUseDMs, flagUseFCM,
            flagHighZern, self.lam, self.amp, wfeMeas,
            self.diamPupil, self.xOffset, self.yOffset,
            self.dm_fn, self.fcm_fn, self.flatten_params_fn,
            data_path=LOCAL_PATH)

        # Make WFE maps for Z1,2,3
        nArray = self.nArray
        wfeZ1to3 = gen_zernikes(np.array([1, 2, 3]), zernCoefVec[0:3],
                                self.xOffset, self.yOffset,
                                self.diamPupil, nArray=nArray)

        # Verify that the output conjugates the input
        usablePixelMap = ampthresh(self.amp)
        usablePixelMapPad = inin(usablePixelMap, (nArray, nArray))
        wfeMeasPad = inin(wfeMeas, (nArray, nArray))

        deltaV1 = Vdm1Out-self.Vdm1In
        deltaV2 = Vdm2Out-self.Vdm2In
        dm1phase = volts_to_dmh(self.dm1gainmap, deltaV1, self.lam)
        dm2phase = volts_to_dmh(self.dm2gainmap, deltaV2, self.lam)

        dm1r = self.dm1r
        DM1SurfActual = dmhtoph(
            nArray, nArray, dm1phase, dm1r['nact'],
            fits.getdata(self.dm1inffn), dm1r['ppact_d'],
            dm1r['ppact_cx'], dm1r['ppact_cy'], dm1r['dx'],
            dm1r['dy'], dm1r['thact'], dm1r['flipx'])

        dm2r = self.dm2r
        DM2SurfActual = dmhtoph(
            nArray, nArray, dm2phase, dm2r['nact'],
            fits.getdata(self.dm2inffn), dm2r['ppact_d'],
            dm2r['ppact_cx'], dm2r['ppact_cy'], dm2r['dx'],
            dm2r['dy'], dm2r['thact'], dm2r['flipx'])

        wfeResidual = usablePixelMapPad*(wfeMeasPad - wfeZ1to3 +
                                         2*(DM1SurfActual + DM2SurfActual))
        rmsError = np.sqrt(np.mean(wfeResidual[usablePixelMapPad]**2))
        # print('CONJUGATE: RMS WFE residual for small Z4 SPM case = %.3g rad'
        #       % rmsError)
        self.assertTrue(rmsError < 1e-9*(2*np.pi/self.lam))

    def test_largeZ4_SPMs(self):
        """Flatten the WFE after moving in an SPM. Large focus change case."""
        flagUseFCM = True
        flagUseDMs = True
        flagHighZern = False

        rng = np.random.default_rng(21)
        zernIndVec = np.arange(4, 12)
        zernCoefVec = rng.standard_normal(zernIndVec.shape) *\
            (10e-9 / self.lam)
        zernCoefVec[0] = 100e-9 / self.lam

        nBeam = 386
        wfeMeas = gen_zernikes(zernIndVec, zernCoefVec, self.xOffset,
                               self.yOffset, nBeam, nArray=self.nArray)

        # wfeMeas = fits.getdata(os.path.join(
        #     self.localpath, 'testdata', 'ph_spm_largeZ4_N386.fits'))
        # amp = fits.getdata(os.path.join(
        #     self.localpath, 'testdata', 'SPM_SPC-20200610_386.fits'))

        Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
            self.Vdm1In, self.Vdm2In, self.cmdFCMIn, flagUseDMs, flagUseFCM,
            flagHighZern, self.lam, self.amp, wfeMeas,
            self.diamPupil, self.xOffset, self.yOffset,
            self.dm_fn, self.fcm_fn, self.flatten_params_fn,
            data_path=LOCAL_PATH)

        # Make WFE maps for Z1,2,3 and Z4
        nArray = self.nArray
        wfeZ1to3 = gen_zernikes(np.array([1, 2, 3]), zernCoefVec[0:3],
                                self.xOffset, self.yOffset,
                                self.diamPupil, nArray=nArray)
        wfeZ4 = gen_zernikes(np.array([4]), np.array([zernCoefVec[3]],),
                             self.xOffset, self.yOffset,
                             self.diamPupil, nArray=nArray)

        # Verify that the output conjugates the input
        usablePixelMap = ampthresh(self.amp)
        usablePixelMapPad = inin(usablePixelMap, (nArray, nArray))
        wfeMeasPad = inin(wfeMeas, (nArray, nArray))

        deltaV1 = Vdm1Out-self.Vdm1In
        deltaV2 = Vdm2Out-self.Vdm2In
        dm1phase = volts_to_dmh(self.dm1gainmap, deltaV1, self.lam)
        dm2phase = volts_to_dmh(self.dm2gainmap, deltaV2, self.lam)

        flipx = False
        dm1r = self.dm1r
        DM1SurfActual = dmhtoph(
            nArray, nArray, dm1phase, dm1r['nact'],
            fits.getdata(self.dm1inffn), dm1r['ppact_d'],
            dm1r['ppact_cx'], dm1r['ppact_cy'], dm1r['dx'],
            dm1r['dy'], dm1r['thact'], flipx)

        dm2r = self.dm2r
        DM2SurfActual = dmhtoph(
            nArray, nArray, dm2phase, dm2r['nact'],
            fits.getdata(self.dm2inffn), dm2r['ppact_d'],
            dm2r['ppact_cx'], dm2r['ppact_cy'], dm2r['dx'],
            dm2r['dy'], dm2r['thact'], flipx)

        wfeResidual = usablePixelMapPad*(wfeMeasPad - wfeZ1to3 - wfeZ4 +
                                         2*(DM1SurfActual + DM2SurfActual))
        rmsError = np.sqrt(np.mean(wfeResidual[usablePixelMapPad]**2))
        # print('CONJUGATE: RMS WFE residual for large Z4 SPM case = %.3g rad'
        #       % rmsError)
        self.assertTrue(rmsError < 1e-9*(2*np.pi/self.lam))

    def test_nonfunctional_actuators(self):
        """Test phase flattening with a different dead actuator on each DM."""
        peakSurfPoke = 1e-8  # meters
        maxAllowedError = 1e-10  # meters

        flagUseFCM = False
        flagUseDMs = True
        flagHighZern = True

        nArray = self.amp.shape[0]

        dm_fn = os.path.join(self.localpath, 'testdata',
                             'dm_with_bad_act.yaml')
        dmConfig = loadyaml(dm_fn)
        dm = dmConfig["dms"]
        dm1r = dm['DM1']['registration']
        dm2r = dm['DM2']['registration']
        dm1r = dm1r
        dm2r = dm2r

        dm1r.pop('inffn')
        dm1r['inf_func'] = fits.getdata(self.dm1inffn)
        peakInf1 = np.max(dm1r['inf_func'])

        dm2r.pop('inffn')
        dm2r['inf_func'] = fits.getdata(self.dm2inffn)
        peakInf2 = np.max(dm2r['inf_func'])

        nact1 = dm1r['nact']
        dmin1 = np.zeros((nact1, nact1))  # all acts
        dmin1[36, 24] = peakSurfPoke/peakInf1
        nact2 = dm2r['nact']
        dmin2 = np.zeros((nact2, nact2))  # all acts
        dmin2[11, 22] = peakSurfPoke/peakInf2

        d1 = dmhtoph(nrow=nArray, ncol=nArray, dmin=dmin1, **dm1r)
        d2 = dmhtoph(nrow=nArray, ncol=nArray, dmin=dmin2, **dm2r)
        dm1gain = 4e-9
        dm2gain = 4e-9
        ph = (d1*dm1gain + d2*dm2gain)/self.lam*(2.0*np.pi)  # radians

        Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
            self.Vdm1In, self.Vdm2In, self.cmdFCMIn, flagUseDMs, flagUseFCM,
            flagHighZern, self.lam, self.amp, ph,
            self.diamPupil, self.xOffset, self.yOffset,
            dm_fn, self.fcm_fn, self.flatten_params_fn,
            data_path=LOCAL_PATH)

        dV1 = Vdm1Out - self.Vdm1In
        dV2 = Vdm2Out - self.Vdm2In

        d1out = 2*dmhtoph(nrow=nArray, ncol=nArray, dmin=dV1, **dm1r)
        d2out = 2*dmhtoph(nrow=nArray, ncol=nArray, dmin=dV2, **dm2r)
        d12out = d1out + d2out

        # Verify that total WFE residual is small
        maxSurfResidual = np.max(d12out + (d1 + d2))
        # print('CONJUGATE: max residual = %.3g meters' % maxSurfResidual)
        self.assertTrue(maxSurfResidual < maxAllowedError)

        # Verify that each DM had equal contribution
        maxPeakDifference = np.abs(np.max(np.abs(d1out)) -
                                   np.max(np.abs(d2out)))
        # print('CONJUGATE: max DM1-DM2 WFE difference = %.3g meters' %
        #       maxPeakDifference)
        self.assertTrue(maxPeakDifference < maxAllowedError)

    def test_minimize_focus_residual_when_setting_coarse_fcm(self):
        """Make sure coarse FCM calibration is off by <= half a step."""
        flagUseFCM = True
        flagUseDMs = False
        flagHighZern = False
        # Sorry, only reason it's not a one-shot is the coarse step size.
        # Probably don't need to do iteratively; could just show residual is
        # less than 1/2 FCM coarse step.

        nBeam = 386
        zernIndVec = np.array([4, ])
        zernCoefVec = np.array([1, ])
        wfe = gen_zernikes(zernIndVec, zernCoefVec, self.xOffset,
                           self.yOffset, nBeam, nArray=self.nArray)

        # wfe = fits.getdata(os.path.join(
        #     self.localpath, 'testdata', 'ph_Z4_1radianRMS_N386.fits'))
        # amp = fits.getdata(os.path.join(
        #     self.localpath, 'testdata',
        #     'amp_cgi_entrance_phase_b_N386_radians_575nm.fits'))

        Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
            self.Vdm1In, self.Vdm2In, self.cmdFCMIn, flagUseDMs, flagUseFCM,
            flagHighZern, self.lam, self.amp, wfe,
            self.diamPupil, self.xOffset, self.yOffset,
            self.dm_fn, self.fcm_fn, self.flatten_params_fn,
            data_path=LOCAL_PATH)

        deltaCmdFCM = cmdFCMOut - self.cmdFCMIn
        stepSizeRadiansRMS = (2*np.pi / self.lam * self.focusPerStroke *
                              self.strokePerStep)
        focusRadiansRMSEst = deltaCmdFCM*stepSizeRadiansRMS
        focusRadiansRMSTrue = 1.0
        absDiff = np.abs(focusRadiansRMSEst - focusRadiansRMSTrue)
        self.assertTrue(absDiff < 0.5)

    # Input Tests
    #
    # The last three inputs are YAML file names and are not tested because
    # they are directly passed to loadyaml(), which already has all its own
    # unit tests.
    def test_input_0(self):
        """Verify that an exception is raised for bad inputs."""
        for Vdm1InBad in (-1, 0.5, [2, ], 1j*np.ones((5, 10)), 'str'):
            with self.assertRaises(ValueError):
                Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
                    Vdm1InBad, self.Vdm2In, self.cmdFCMIn,
                    self.flagUseDMs, self.flagUseFCM, self.flagHighZern,
                    self.lam, self.amp, self.ph,
                    self.diamPupil, self.xOffset, self.yOffset,
                    self.dm_fn, self.fcm_fn, self.flatten_params_fn,
                    data_path=LOCAL_PATH)

    def test_input_1(self):
        """Verify that an exception is raised for bad inputs."""
        for Vdm2InBad in (-1, 0.5, [2, ], 1j*np.ones((5, 10)), 'str'):
            with self.assertRaises(ValueError):
                Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
                    self.Vdm1In, Vdm2InBad, self.cmdFCMIn,
                    self.flagUseDMs, self.flagUseFCM, self.flagHighZern,
                    self.lam, self.amp, self.ph,
                    self.diamPupil, self.xOffset, self.yOffset,
                    self.dm_fn, self.fcm_fn, self.flatten_params_fn,
                    data_path=LOCAL_PATH)

    def test_input_2(self):
        """Verify that an exception is raised for bad inputs."""
        for cmdFCMInBad in (-1.5, 2.5, 1j, [3, ], np.ones((5, 10)), 'str'):
            with self.assertRaises(ValueError):
                Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
                    self.Vdm1In, self.Vdm2In, cmdFCMInBad,
                    self.flagUseDMs, self.flagUseFCM, self.flagHighZern,
                    self.lam, self.amp, self.ph,
                    self.diamPupil, self.xOffset, self.yOffset,
                    self.dm_fn, self.fcm_fn, self.flatten_params_fn,
                    data_path=LOCAL_PATH)

    def test_input_3(self):
        """Verify that an exception is raised for bad inputs."""
        for flagUseDMsBad in (-1, 0, 1, [True, ], np.ones((5, 10)), 'str'):
            with self.assertRaises(TypeError):
                Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
                    self.Vdm1In, self.Vdm2In, self.cmdFCMIn,
                    flagUseDMsBad, self.flagUseFCM, self.flagHighZern,
                    self.lam, self.amp, self.ph,
                    self.diamPupil, self.xOffset, self.yOffset,
                    self.dm_fn, self.fcm_fn, self.flatten_params_fn,
                    data_path=LOCAL_PATH)

    def test_input_4(self):
        """Verify that an exception is raised for bad inputs."""
        for flagUseFCMBad in (-1, 0, 1, [True, ], np.ones((5, 10)), 'str'):
            with self.assertRaises(TypeError):
                Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
                    self.Vdm1In, self.Vdm2In, self.cmdFCMIn,
                    self.flagUseDMs, flagUseFCMBad, self.flagHighZern,
                    self.lam, self.amp, self.ph,
                    self.diamPupil, self.xOffset, self.yOffset,
                    self.dm_fn, self.fcm_fn, self.flatten_params_fn,
                    data_path=LOCAL_PATH)

    def test_input_5(self):
        """Verify that an exception is raised for bad inputs."""
        for cmdFCMInBad in (1.5, 1j, [True, ], np.ones((5, 10)), 'str'):
            with self.assertRaises(ValueError):
                Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
                    self.Vdm1In, self.Vdm2In, cmdFCMInBad,
                    self.flagUseDMs, self.flagUseFCM, self.flagHighZern,
                    self.lam, self.amp, self.ph,
                    self.diamPupil, self.xOffset, self.yOffset,
                    self.dm_fn, self.fcm_fn, self.flatten_params_fn,
                    data_path=LOCAL_PATH)

    def test_input_6(self):
        """Verify that an exception is raised for bad inputs."""
        for lamBad in (-1, 0, 1j, [0.5, ], np.ones((5, 10)), 'str'):
            with self.assertRaises(ValueError):
                Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
                    self.Vdm1In, self.Vdm2In, self.cmdFCMIn,
                    self.flagUseDMs, self.flagUseFCM, self.flagHighZern,
                    lamBad, self.amp, self.ph,
                    self.diamPupil, self.xOffset, self.yOffset,
                    self.dm_fn, self.fcm_fn, self.flatten_params_fn,
                    data_path=LOCAL_PATH)

    def test_input_7(self):
        """Verify that an exception is raised for bad inputs."""
        for ampBad in (-1, 0.5, [2, ], 1j*np.ones((5, 10)), 'str'):
            with self.assertRaises(ValueError):
                Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
                    self.Vdm1In, self.Vdm2In, self.cmdFCMIn,
                    self.flagUseDMs, self.flagUseFCM, self.flagHighZern,
                    self.lam, ampBad, self.ph,
                    self.diamPupil, self.xOffset, self.yOffset,
                    self.dm_fn, self.fcm_fn, self.flatten_params_fn,
                    data_path=LOCAL_PATH)

    def test_input_8(self):
        """Verify that an exception is raised for bad inputs."""
        for phBad in (-1, 0.5, [2, ], 1j*np.ones((5, 10)), 'str'):
            with self.assertRaises(ValueError):
                Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
                    self.Vdm1In, self.Vdm2In, self.cmdFCMIn,
                    self.flagUseDMs, self.flagUseFCM, self.flagHighZern,
                    self.lam, self.amp, phBad,
                    self.diamPupil, self.xOffset, self.yOffset,
                    self.dm_fn, self.fcm_fn, self.flatten_params_fn,
                    data_path=LOCAL_PATH)

    def test_input_9(self):
        """Verify that an exception is raised for bad inputs."""
        for diamPupilBad in (-1, 0, 1j, [0.5, ], np.ones((5, 10)), 'str'):
            with self.assertRaises(ValueError):
                Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
                    self.Vdm1In, self.Vdm2In, self.cmdFCMIn,
                    self.flagUseDMs, self.flagUseFCM, self.flagHighZern,
                    self.lam, self.amp, self.ph,
                    diamPupilBad, self.xOffset, self.yOffset,
                    self.dm_fn, self.fcm_fn, self.flatten_params_fn,
                    data_path=LOCAL_PATH)

    def test_input_10(self):
        """Verify that an exception is raised for bad inputs."""
        for xOffsetBad in (1j, [2, ], np.ones((5, 10)), 'str'):
            with self.assertRaises(ValueError):
                Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
                    self.Vdm1In, self.Vdm2In, self.cmdFCMIn,
                    self.flagUseDMs, self.flagUseFCM, self.flagHighZern,
                    self.lam, self.amp, self.ph,
                    self.diamPupil, xOffsetBad, self.yOffset,
                    self.dm_fn, self.fcm_fn, self.flatten_params_fn,
                    data_path=LOCAL_PATH)

    def test_input_11(self):
        """Verify that an exception is raised for bad inputs."""
        for yOffsetBad in (1j, [2, ], np.ones((5, 10)), 'str'):
            with self.assertRaises(ValueError):
                Vdm1Out, Vdm2Out, cmdFCMOut, zernCoefVec = flatten(
                    self.Vdm1In, self.Vdm2In, self.cmdFCMIn,
                    self.flagUseDMs, self.flagUseFCM, self.flagHighZern,
                    self.lam, self.amp, self.ph,
                    self.diamPupil, self.xOffset, yOffsetBad,
                    self.dm_fn, self.fcm_fn, self.flatten_params_fn,
                    data_path=LOCAL_PATH)


class TestConjugateSupportFunctions(unittest.TestCase):
    """Test all the supporting methods of conjugate."""

    def test_gen_split_wfe(self):
        """Test that gen_split_wfe correctly generates WFE maps."""
        # Cases: flagUseFCM == True, flagUseFCM == False
        diamPupil = 100
        wfeShape = (120, 130)
        xOffset = 0.0
        yOffset = 0.0
        maxNollZern = 11
        zernCoefVec = np.arange(1, 13)

        x = (np.arange(wfeShape[1], dtype=np.float64) - wfeShape[1]//2)
        y = (np.arange(wfeShape[0], dtype=np.float64) - wfeShape[0]//2)
        xx, yy = np.meshgrid(x, y)

        # Create the true WFE allocations
        zernIndVec = np.arange(1, 13)
        zernCube = xyzern(xx, yy, diamPupil/2., zernIndVec)
        wfeZ1to3True = np.zeros(wfeShape, dtype=np.float64)
        wfeZ4True = np.zeros(wfeShape, dtype=np.float64)
        wfeZ4or5to11True = np.zeros(wfeShape, dtype=np.float64)
        for iz in (0, 1, 2):
            wfeZ1to3True += zernCoefVec[iz]*np.squeeze(zernCube[iz, :, :])
        for iz in (3,):
            wfeZ4True += zernCoefVec[iz]*np.squeeze(zernCube[iz, :, :])
        for iz in (4, 5, 6, 7, 8, 9, 10):
            wfeZ4or5to11True += zernCoefVec[iz]*np.squeeze(zernCube[iz, :, :])

        flagUseFCM = True
        wfeZ1to3, wfeZ4, wfeZ4or5to11 = gen_split_wfe(
            flagUseFCM, maxNollZern, zernCoefVec, diamPupil, wfeShape,
            xOffset, yOffset)

        self.assertTrue(isclose(np.sum(np.abs(wfeZ1to3 - wfeZ1to3True)), 0.0,
                                abs_tol=1e-4))
        self.assertTrue(isclose(np.sum(np.abs(wfeZ4 - wfeZ4True)), 0.0,
                                abs_tol=1e-4))
        self.assertTrue(isclose(np.sum(np.abs(wfeZ4or5to11 -
                                              wfeZ4or5to11True)), 0.0,
                                abs_tol=1e-4))

        flagUseFCM = False
        wfeZ1to3, wfeZ4, wfeZ4or5to11 = gen_split_wfe(
            flagUseFCM, maxNollZern, zernCoefVec, diamPupil, wfeShape,
            xOffset, yOffset)

        # wfeZ4True = np.zeros(wfeShape, dtype=np.float64)
        wfeZ4or5to11True = np.zeros(wfeShape, dtype=np.float64)
        for iz in (3, 4, 5, 6, 7, 8, 9, 10):
            wfeZ4or5to11True += zernCoefVec[iz]*np.squeeze(zernCube[iz, :, :])

        self.assertTrue(isclose(np.sum(np.abs(wfeZ4)), 0.0, abs_tol=1e-4))
        self.assertTrue(isclose(np.sum(np.abs(wfeZ4or5to11 -
                                              wfeZ4or5to11True)), 0.0,
                                abs_tol=1e-4))
        pass

    def test_smooth_surface(self):
        """Test that smooth_surface correctly smooths an array."""
        windowWidth = 3
        arrayIn = np.zeros((12, 12))
        arrayIn[1::3, 1::3] = 1
        arrayOut = smooth_surface(arrayIn, windowWidth)
        arrayExpected = np.ones((12, 12)) / 9.0
        diffSum = np.sum(np.abs(arrayOut - arrayExpected))
        self.assertTrue(isclose(diffSum, 0.0, abs_tol=1e-8))
        self.assertTrue(isclose(np.sum(arrayOut), np.sum(arrayExpected),
                                abs_tol=1e-8))
        pass

    def test_conv_surf_to_dm_cmd(self):
        """Test conv_surf_to_dm_cmd."""
        # The fit isn't as good at the last actuator, so don't include
        dVROI = inin(np.ones((46, 46)), (48, 48))
        lam = 575e-9

        dm_fn = os.path.join(LOCAL_PATH, 'testdata', 'dm_offset.yaml')
        flipx = False

        # Load DM Calibration Data
        dmConfig = loadyaml(dm_fn)
        dm = dmConfig["dms"]
        dm1r = dm['DM1']['registration']
        dm1gainmap_fn = os.path.join(LOCAL_PATH,
                                     dm['DM1']['voltages']['gainfn'])
        dm1gainmap = fits.getdata(dm1gainmap_fn)
        dm1phase = volts_to_dmh(dm1gainmap, dVROI, lam)
        dm1inffn = os.path.join(LOCAL_PATH, dm1r['inffn'])

        # Generate a map of the region the DM can control
        nrow = 500
        ncol = 500
        dm1TestSurf = dmhtoph(
            nrow, ncol, dm1phase, dm1r['nact'], fits.getdata(dm1inffn),
            dm1r['ppact_d'], dm1r['ppact_cx'], dm1r['ppact_cy'], dm1r['dx'],
            dm1r['dy'], dm1r['thact'], flipx=flipx)
        # dm1TestSurf = dmhtoph(dm1phase, dm1r['nact'],
        #                       fits.getdata(dm1inffn),
        #                       dm1r['ppact_d'], dm1r['ppact_cx'],
        #                       dm1r['ppact_cy'], dm1r['dx'], dm1r['dy'],
        #                       dm1r['thact'], flipx=flipx)
        usablePixelMap = dm1TestSurf.copy()
        usablePixelMap[dm1TestSurf >= 0.7*np.max(dm1TestSurf)] = 1
        usablePixelMap[dm1TestSurf < 0.7*np.max(dm1TestSurf)] = 0

        # Create coordinates for the the error map to fit
        # diamPupil = np.min(dm1TestSurf.shape)
        x = (np.arange(dm1TestSurf.shape[1], dtype=np.float64)
             - dm1TestSurf.shape[1]//2)
        y = (np.arange(dm1TestSurf.shape[0], dtype=np.float64)
             - dm1TestSurf.shape[0]//2)
        xx, yy = np.meshgrid(x, y)

        # Create the error map to fit
        diamPupil = 386
        zernIndVec = np.arange(4, 31)

        rng = np.random.default_rng(55555)
        # zernCoefVec = rng.standard_normal(zernIndVec.shape) *\
        #     (40e-9 / self.lam)

        zernCoefVec = rng.standard_normal(zernIndVec.size) *\
            (30e-9 / 575e-9)
        zernCube = xyzern(xx, yy, diamPupil/2., zernIndVec)
        zernAbMap = np.zeros(dm1TestSurf.shape, dtype=np.float64)
        for ii in range(zernIndVec.size):
            zernAbMap += zernCoefVec[ii]*np.squeeze(zernCube[ii, :, :])

        # zernAbMap = fits.getdata(os.path.join(
        #     localpath, 'testdata', 'ph_for_conv_surf_to_dm_cmd.fits'))
        # usablePixelMap = fits.getdata(os.path.join(
        #     localpath, 'testdata', 'usable_for_conv_surf_to_dm_cmd.fits'))

        mapShape = zernAbMap.shape

        # # Load DM Calibration Data
        # dm_fn = os.path.join(LOCAL_PATH, 'testdata', 'dm_offset.yaml')
        # dmConfig = loadyaml(dm_fn)
        # dm = dmConfig["dms"]
        # dm1r = dm['DM1']['registration']
        # dm1gainmap_fn = os.path.join(LOCAL_PATH,
        #                              dm['DM1']['voltages']['gainfn'])
        # dm1gainmap = fits.getdata(dm1gainmap_fn)
        # dm1phase = volts_to_dmh(dm1gainmap, dVROI, lam)
        # dm1inffn = os.path.join(LOCAL_PATH, dm1r['inffn'])
        # dm1r['inf_func'] = fits.getdata(dm1inffn)

        # Compute delta DM settings
        hdm1 = conv_surf_to_dm_cmd(zernAbMap, dm['DM1'], data_path=LOCAL_PATH)
        dV = dmh_to_volts(dm1gainmap, hdm1, lam)
        dm1phase = volts_to_dmh(dm1gainmap, dV, lam)
        # dm1r.pop('inffn')
        # dm1Surf = dmhtoph(mapShape[0], mapShape[1], dm1phase, **dm1r)
        dm1Surf = dmhtoph(
            nrow, ncol, dm1phase, dm1r['nact'], fits.getdata(dm1inffn),
            dm1r['ppact_d'], dm1r['ppact_cx'], dm1r['ppact_cy'], dm1r['dx'],
            dm1r['dy'], dm1r['thact'], flipx=flipx)
        # RMS error is dominated by the edges of the WFE map
        diff = (zernAbMap - dm1Surf) * usablePixelMap  # [radians]
        diffRMS = rms(diff[usablePixelMap.astype(bool)])
        # print('CONJUGATE: RMS WFE residual = %.3g rad' % diffRMS)

        # self.assertTrue(diffRMS < 0.04)

    def test_phase_cost_function(self):
        """Test phase_cost_function."""
        # If wfeZ4or5to11 and wfeAboveZ11 have same RMS,
        # then alpha=0 should give cost==0
        usablePixelMap = np.ones((2, 2), dtype=bool)

        alpha = 0.0
        wfeZ4or5to11 = np.array([[-1, 1], [1, -1]])
        wfeAboveZ11 = np.array([[1, 1], [-1, -1]])
        cost = phase_cost_function(alpha, wfeZ4or5to11, wfeAboveZ11,
                                   usablePixelMap)
        self.assertTrue(isclose(cost, 0.0, abs_tol=1e-8))

        # If rms(wfeZ4or5to11) < rms(wfeAboveZ11),
        # then alpha=0 should give cost > 0
        alpha = 0.0
        wfeZ4or5to11 = np.array([[-1, 1], [1, -1]])
        wfeAboveZ11 = 4*np.array([[-1, 1], [1, -1]])
        cost = phase_cost_function(alpha, wfeZ4or5to11, wfeAboveZ11,
                                   usablePixelMap)
        self.assertTrue(isclose(cost, (4.-1.)**2, abs_tol=1e-8))

        # If rms(wfeZ4or5to11) > rms(wfeAboveZ11)
        alpha = 1/3.0
        wfeZ4or5to11 = 3*np.array([[-1, 1], [1, -1]])
        wfeAboveZ11 = np.array([[-1, 1], [1, -1]])
        cost = phase_cost_function(alpha, wfeZ4or5to11, wfeAboveZ11,
                                   usablePixelMap)
        self.assertTrue(isclose(cost, 0.0, abs_tol=1e-8))
        pass


class TestSupportFunctionInputs(unittest.TestCase):
    """Test suite for valid function inputs."""

    def test_gen_split_wfe_inputs(self):
        """Test incorrect inputs of gen_split_wfe."""
        for flagUseFCMBad in (0, 1, 1.5, np.ones((5, 10)), 'string'):
            flagUseFCM = True
            maxNollZern = 9
            zernCoef = np.random.rand(11)
            diamPupil = 386.
            surfShape = (501, 485)
            xOffset = -0.1
            yOffset = 0.1

            for flagUseDM1Bad in (0, 1, 1.5, np.ones((5, 10)), 'string'):
                with self.assertRaises(TypeError):
                    gen_split_wfe(flagUseFCMBad, maxNollZern,
                                  zernCoef, diamPupil, surfShape,
                                  xOffset, yOffset)
            for maxNollZernBad in (-1, 0, 1.5, np.ones((5,)), 'string'):
                with self.assertRaises(ValueError):
                    gen_split_wfe(flagUseFCM, maxNollZernBad,
                                  zernCoef, diamPupil, surfShape,
                                  xOffset, yOffset)
            for zernCoefBad in (1, np.ones((5, 10)), 'string'):
                with self.assertRaises(ValueError):
                    gen_split_wfe(flagUseFCM, maxNollZern,
                                  zernCoefBad, diamPupil, surfShape,
                                  xOffset, yOffset)
            for diamPupilBad in (-1.5, 0, np.ones((5,)), 'string'):
                with self.assertRaises(ValueError):
                    gen_split_wfe(flagUseFCM, maxNollZern, zernCoef,
                                  diamPupilBad, surfShape,
                                  xOffset, yOffset)
            for surfShapeBad in (0, np.ones((5,)), (10, 10, 10), 'string'):
                with self.assertRaises(ValueError):
                    gen_split_wfe(flagUseFCM, maxNollZern, zernCoef,
                                  diamPupil, surfShapeBad,
                                  xOffset, yOffset)
            for xOffsetBad in (1.5+1j, np.ones((5,)), np.ones((10, 10)),
                               'string'):
                with self.assertRaises(ValueError):
                    gen_split_wfe(flagUseFCM, maxNollZern, zernCoef,
                                  diamPupil, surfShape,
                                  xOffsetBad, yOffset)
            for yOffsetBad in (1.5+1j, np.ones((5,)), np.ones((10, 10)),
                               'string'):
                with self.assertRaises(ValueError):
                    gen_split_wfe(flagUseFCM, maxNollZern, zernCoef,
                                  diamPupil, surfShape,
                                  xOffset, yOffsetBad)

    def test_smooth_surface_inputs(self):
        """Test incorrect inputs of smooth_surface."""
        with self.assertRaises(ValueError):
            smooth_surface(np.ones((3, 3, 4)), 3.2)
        with self.assertRaises(ValueError):
            smooth_surface(np.ones((10, 10)), np.array([2, 4.]))
        with self.assertRaises(ValueError):
            smooth_surface(np.ones((10, 10)), -4.2)

    def test_conv_surf_to_dm_cmd_inputs(self):
        """Test incorrect inputs of conv_surf_to_dm_cmd."""
        localpath = os.path.dirname(os.path.abspath(__file__))

        # Load DM Calibration Data
        dm_fn = os.path.join(localpath, 'testdata', 'dm_offset.yaml')
        dmConfig = loadyaml(dm_fn)
        dm = dmConfig["dms"]
        dm1 = dm['DM1']
        surfaceToFit = np.ones((500, 500))

        for surfaceToFitBad in (1, np.ones((5, 10, 20)), 'string'):
            with self.assertRaises(ValueError):
                conv_surf_to_dm_cmd(surfaceToFitBad, dm1)

        for dm1Bad in (1, np.ones((5, 10)), 'string'):
            with self.assertRaises(TypeError):
                conv_surf_to_dm_cmd(surfaceToFit, dm1Bad)

    def test_phase_cost_function_inputs(self):
        """Test incorrect inputs of phase_cost_function."""
        shape = (100, 102)
        alpha = 0.25
        wfeZ4or5to11 = 4*np.random.rand(shape[0], shape[1])
        wfeAboveZ11 = np.random.rand(shape[0], shape[1])
        usablePixelMap = np.round(np.random.rand(shape[0],
                                                 shape[1])).astype(bool)

        for alphaBad in (-0.1, 1.1, 1j, np.ones((5, 10)), 'string'):
            with self.assertRaises(ValueError):
                phase_cost_function(alphaBad, wfeZ4or5to11,
                                    wfeAboveZ11, usablePixelMap)

        for wfeZ4or5to11Bad in (-0.1, 1.1, 1j, [2, ], np.ones((5, 10, 3)),
                                np.ones((5, 10)), 1j*np.ones(shape), 'string'):
            with self.assertRaises(ValueError):
                phase_cost_function(alpha, wfeZ4or5to11Bad,
                                    wfeAboveZ11, usablePixelMap)

        for wfeAboveZ11Bad in (-0.1, 1.1, 1j, [2, ], np.ones((5, 10, 3)),
                               np.ones((5, 10)), 1j*np.ones(shape), 'string'):
            with self.assertRaises(ValueError):
                phase_cost_function(alpha, wfeZ4or5to11,
                                    wfeAboveZ11Bad, usablePixelMap)

        for usablePixelMapBad in (-0.1, 1.1, 1j, [2, ], np.ones((5, 10, 3)),
                                  np.ones((5, 10)), 1.5*np.ones(shape),
                                  'string'):
            with self.assertRaises(ValueError):
                phase_cost_function(alpha, wfeZ4or5to11,
                                    wfeAboveZ11Bad, usablePixelMap)


if __name__ == '__main__':
    unittest.main()
