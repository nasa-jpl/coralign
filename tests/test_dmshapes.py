"""
Test class for functions to make shapes for a DM
"""

import unittest

import numpy as np

from coralign.util.dmshapes import probe, sine, grid, zernike


class TestProbe(unittest.TestCase):
    """
    Unit tests for function to generate a DM probe
    """
    def setUp(self):
        """
        Predefine OK parameters so we don't have to repeat them
        """
        self.nact = 48
        self.dact = 46.3
        self.xcenter = 0
        self.ycenter = 0
        self.clock = 0
        self.height = 100e-9
        self.ximin = 2
        self.ximax = 11
        self.etamin = -11
        self.etamax = 11
        self.phase = 0
        pass

    # inputs
    def test_success(self):
        """
        Run with valid inputs should not throw anything
        """
        probe(nact=self.nact, dact=self.dact, xcenter=self.xcenter,
              ycenter=self.ycenter, clock=self.clock, height=self.height,
              ximin=self.ximin, ximax=self.ximax, etamin=self.etamin,
              etamax=self.etamax, phase=self.phase)

        pass

    def test_odd_success(self):
        """
        Odd-numbered actuator counts should not break anything, even though
        they center on an actuator rather than on a gap
        """
        probe(nact=self.nact+1, dact=self.dact, xcenter=self.xcenter,
              ycenter=self.ycenter, clock=self.clock, height=self.height,
              ximin=self.ximin, ximax=self.ximax, etamin=self.etamin,
              etamax=self.etamax, phase=self.phase)
        pass


    def test_nact_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, [5], -1, -1.5, 1.5, 0]:
            with self.assertRaises(TypeError):
                probe(nact=perr, dact=self.dact, xcenter=self.xcenter,
                      ycenter=self.ycenter, clock=self.clock,
                      height=self.height,
                      ximin=self.ximin, ximax=self.ximax, etamin=self.etamin,
                      etamax=self.etamax, phase=self.phase)
                pass
            pass
        pass


    def test_dact_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, [5], -1, -1.5, 0]:
            with self.assertRaises(TypeError):
                probe(nact=self.nact, dact=perr, xcenter=self.xcenter,
                      ycenter=self.ycenter, clock=self.clock,
                      height=self.height,
                      ximin=self.ximin, ximax=self.ximax, etamin=self.etamin,
                      etamax=self.etamax, phase=self.phase)
                pass
            pass
        pass

    def test_height_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, [5], -1, -1.5, 0]:
            with self.assertRaises(TypeError):
                probe(nact=self.nact, dact=self.dact, xcenter=self.xcenter,
                      ycenter=self.ycenter, clock=self.clock,
                      height=perr,
                      ximin=self.ximin, ximax=self.ximax, etamin=self.etamin,
                      etamax=self.etamax, phase=self.phase)
                pass
            pass
        pass

    def test_xcenter_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, [5]]:
            with self.assertRaises(TypeError):
                probe(nact=self.nact, dact=self.dact, xcenter=perr,
                      ycenter=self.ycenter, clock=self.clock,
                      height=self.height,
                      ximin=self.ximin, ximax=self.ximax, etamin=self.etamin,
                      etamax=self.etamax, phase=self.phase)
                pass
            pass
        pass

    def test_ycenter_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, [5]]:
            with self.assertRaises(TypeError):
                probe(nact=self.nact, dact=self.dact, xcenter=self.xcenter,
                      ycenter=perr, clock=self.clock,
                      height=self.height,
                      ximin=self.ximin, ximax=self.ximax, etamin=self.etamin,
                      etamax=self.etamax, phase=self.phase)
                pass
            pass
        pass

    def test_clock_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, [5]]:
            with self.assertRaises(TypeError):
                probe(nact=self.nact, dact=self.dact, xcenter=self.xcenter,
                      ycenter=self.ycenter, clock=perr,
                      height=self.height,
                      ximin=self.ximin, ximax=self.ximax, etamin=self.etamin,
                      etamax=self.etamax, phase=self.phase)
                pass
            pass
        pass

    def test_ximin_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, [5]]:
            with self.assertRaises(TypeError):
                probe(nact=self.nact, dact=self.dact, xcenter=self.xcenter,
                      ycenter=self.ycenter, clock=self.clock,
                      height=self.height,
                      ximin=perr, ximax=self.ximax, etamin=self.etamin,
                      etamax=self.etamax, phase=self.phase)
                pass
            pass
        pass

    def test_ximax_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, [5]]:
            with self.assertRaises(TypeError):
                probe(nact=self.nact, dact=self.dact, xcenter=self.xcenter,
                      ycenter=self.ycenter, clock=self.clock,
                      height=self.height,
                      ximin=self.ximin, ximax=perr, etamin=self.etamin,
                      etamax=self.etamax, phase=self.phase)
                pass
            pass
        pass

    def test_etamin_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, [5]]:
            with self.assertRaises(TypeError):
                probe(nact=self.nact, dact=self.dact, xcenter=self.xcenter,
                      ycenter=self.ycenter, clock=self.clock,
                      height=self.height,
                      ximin=self.ximin, ximax=self.ximax, etamin=perr,
                      etamax=self.etamax, phase=self.phase)
                pass
            pass
        pass

    def test_etamax_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, [5]]:
            with self.assertRaises(TypeError):
                probe(nact=self.nact, dact=self.dact, xcenter=self.xcenter,
                      ycenter=self.ycenter, clock=self.clock,
                      height=self.height,
                      ximin=self.ximin, ximax=self.ximax, etamin=self.etamin,
                      etamax=perr, phase=self.phase)
                pass
            pass
        pass

    def test_phase_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, [5]]:
            with self.assertRaises(TypeError):
                probe(nact=self.nact, dact=self.dact, xcenter=self.xcenter,
                      ycenter=self.ycenter, clock=self.clock,
                      height=self.height,
                      ximin=self.ximin, ximax=self.ximax, etamin=self.etamin,
                      etamax=self.etamax, phase=perr)
                pass
            pass
        pass


    # functionality
    def test_ximin_less_than_ximax(self):
        """
        Should always fail if min >= max
        """
        for perr in [self.ximax, self.ximax+0.1]:
            with self.assertRaises(ValueError):
                probe(nact=self.nact, dact=self.dact, xcenter=self.xcenter,
                      ycenter=self.ycenter, clock=self.clock,
                      height=self.height,
                      ximin=perr, ximax=self.ximax, etamin=self.etamin,
                      etamax=self.etamax, phase=self.phase)
                pass
            pass
        pass

    def test_etamin_less_than_etamax(self):
        """
        Should always fail if min >= max
        """
        for perr in [self.etamax, self.etamax+0.1]:
            with self.assertRaises(ValueError):
                probe(nact=self.nact, dact=self.dact, xcenter=self.xcenter,
                      ycenter=self.ycenter, clock=self.clock,
                      height=self.height,
                      ximin=self.ximin, ximax=self.ximax, etamin=perr,
                      etamax=self.etamax, phase=self.phase)
                pass
            pass
        pass

    def test_rotation(self):
        """
        Test rotation is clockwise in camera (same default direction as rot90)
        """
        tol = 1e-13

        ph0 = 0 # explicitly a sine for asymmetry
        dmp0 = probe(nact=self.nact, dact=self.dact, xcenter=self.xcenter,
              ycenter=self.ycenter, clock=self.clock, height=self.height,
              ximin=self.ximin, ximax=self.ximax, etamin=self.etamin,
              etamax=self.etamax, phase=ph0)

        dmp90 = probe(nact=self.nact, dact=self.dact, xcenter=self.xcenter,
              ycenter=self.ycenter, clock=self.clock+90, height=self.height,
              ximin=self.ximin, ximax=self.ximax, etamin=self.etamin,
              etamax=self.etamax, phase=ph0)

        self.assertTrue(np.max(np.abs(np.rot90(dmp0, 1) - dmp90)) < tol)
        pass

    def test_spacing(self):
        """
        This test is to make sure we understand size of sincs and the offset of
        the sine by checking an explicit analytic case.

        If dact divides nact evenly *and* the xcenter and ycenter are offset by
        0.5 (so they fall on integers) *and* (ximax-ximin)/dact and
        (etamax-etamin)/dact are integers as well *and* there is no non-90deg
        clocking, then we're doing sinc(n) for integer n, and this is zero
        everywhere except at (0,0). If phase == 0, then we post-multiply by
        sin(n), which *is* zero at (0,0), and the array should be identically
        zero modulo numerical error.

        This will fail if we break any of these conditions.
        """
        tol = 1e-13

        dmpn = probe(nact=48, dact=12, xcenter=0.5,
              ycenter=0.5, clock=0, height=self.height,
              ximin=0, ximax=12, etamin=-12,
              etamax=12, phase=0)

        z = np.zeros_like(dmpn)

        self.assertTrue(np.max(np.abs(dmpn-z)) < tol)

        # not x/y centered
        f0 = probe(nact=48, dact=12, xcenter=0.0,
              ycenter=0.5, clock=0, height=self.height,
              ximin=0, ximax=12, etamin=-12,
              etamax=12, phase=0)
        # nact does not divide dact
        f1 = probe(nact=48, dact=46.3, xcenter=0.5,
              ycenter=0.5, clock=0, height=self.height,
              ximin=0, ximax=12, etamin=-12,
              etamax=12, phase=0)
        # clocked
        f2 = probe(nact=48, dact=12, xcenter=0.5,
              ycenter=0.5, clock=15, height=self.height,
              ximin=0, ximax=12, etamin=-12,
              etamax=12, phase=0)
        # non-integer (ximax-ximin)/dact
        f3 = probe(nact=48, dact=12, xcenter=0.5,
              ycenter=0.5, clock=0, height=self.height,
              ximin=0, ximax=13, etamin=-12,
              etamax=12, phase=0)

        for f in [f0, f1, f2, f3]:
            self.assertFalse(np.max(np.abs(f - z)) < tol)
            pass
        pass

    def test_xshift(self):
        """
        Verify xcenters go in the correct direction (camx is cols)
        """
        tol = 1e-13

        x0 = probe(nact=self.nact, dact=self.dact, xcenter=self.xcenter,
              ycenter=self.ycenter, clock=self.clock, height=self.height,
              ximin=self.ximin, ximax=self.ximax, etamin=self.etamin,
              etamax=self.etamax, phase=self.phase)

        xs = probe(nact=self.nact, dact=self.dact, xcenter=self.xcenter+1,
              ycenter=self.ycenter, clock=self.clock, height=self.height,
              ximin=self.ximin, ximax=self.ximax, etamin=self.etamin,
              etamax=self.etamax, phase=self.phase)

        xr = np.roll(x0, 1, 1)
        self.assertTrue(np.max(np.abs(xr[:, 1:] - xs[:, 1:])) < tol)
        pass

    def test_yshift(self):
        """
        Verify ycenters go in the correct direction (camy is rows)
        """
        tol = 1e-13

        y0 = probe(nact=self.nact, dact=self.dact, xcenter=self.xcenter,
              ycenter=self.ycenter, clock=self.clock, height=self.height,
              ximin=self.ximin, ximax=self.ximax, etamin=self.etamin,
              etamax=self.etamax, phase=self.phase)

        ys = probe(nact=self.nact, dact=self.dact, xcenter=self.xcenter,
              ycenter=self.ycenter+1, clock=self.clock, height=self.height,
              ximin=self.ximin, ximax=self.ximax, etamin=self.etamin,
              etamax=self.etamax, phase=self.phase)

        yr = np.roll(y0, 1, 0)
        self.assertTrue(np.max(np.abs(yr[1:, :] - ys[1:, :])) < tol)
        pass

    def test_phase_shift_equals_rot(self):
        """
        Verify that a 180 phase shift matches a 180 deg clock (for a sine)
        """
        tol = 1e-13

        p0 = probe(nact=self.nact, dact=self.dact, xcenter=self.xcenter,
              ycenter=self.ycenter, clock=self.clock, height=self.height,
              ximin=self.ximin, ximax=self.ximax, etamin=self.etamin,
              etamax=self.etamax, phase=180)

        ps = probe(nact=self.nact, dact=self.dact, xcenter=self.xcenter,
              ycenter=self.ycenter, clock=self.clock+180, height=self.height,
              ximin=self.ximin, ximax=self.ximax, etamin=self.etamin,
              etamax=self.etamax, phase=0)

        self.assertTrue(np.max(np.abs(p0 - ps)) < tol)
        pass

    def test_90_rot_equals_xi_eta_swap(self):
        """
        Verify that a 90 degree rotation is equivalent to swapping xi and eta
        """
        tol = 1e-13

        p0 = probe(nact=self.nact, dact=self.dact, xcenter=self.xcenter,
              ycenter=self.ycenter, clock=self.clock, height=self.height,
              ximin=self.ximin, ximax=self.ximax, etamin=self.etamin,
              etamax=self.etamax, phase=self.phase)

        ps = probe(nact=self.nact, dact=self.dact, xcenter=self.xcenter,
              ycenter=self.ycenter, clock=self.clock+90, height=self.height,
              ximin=self.etamin, ximax=self.etamax, etamin=self.ximin,
              etamax=self.ximax, phase=self.phase)

        self.assertTrue(np.max(np.abs(p0 - ps)) < tol)
        pass

    def test_dact_and_xi_eta_scale(self):
        """
        Verify xi/eta scale linearly with dact (i.e. as diameter drops, the
        lambda/D required to hit a certain pixel in the focal plane also drops)
        """
        tol = 1e-13

        p0 = probe(nact=self.nact, dact=self.dact, xcenter=self.xcenter,
              ycenter=self.ycenter, clock=self.clock, height=self.height,
              ximin=self.ximin, ximax=self.ximax, etamin=self.etamin,
              etamax=self.etamax, phase=self.phase)

        ps = probe(nact=self.nact, dact=self.dact/2., xcenter=self.xcenter,
              ycenter=self.ycenter, clock=self.clock, height=self.height,
              ximin=self.ximin/2., ximax=self.ximax/2., etamin=self.etamin/2.,
              etamax=self.etamax/2., phase=self.phase)

        self.assertTrue(np.max(np.abs(p0 - ps)) < tol)
        pass

    def test_clockwise_rot(self):
        """
        Verify that, as seen in the camera, the probe rotates clockwise.  A
        small clockwise rotation should mean the left half of the first row
        has more probe motions than the right (since the sinc tail moved
        into place)
        """

        nact = 48
        p0 = probe(nact=nact, dact=self.dact, xcenter=self.xcenter,
              ycenter=self.ycenter, clock=15, height=self.height,
              ximin=self.ximin, ximax=self.ximax, etamin=self.etamin,
              etamax=self.etamax, phase=self.phase)

        lh = np.sum(np.abs(p0[0, :24])**2)
        rh = np.sum(np.abs(p0[0, 24:])**2)

        self.assertTrue(lh > rh)
        pass

    def test_clock_before_center(self):
        """
        Verify that the clocking is done prior to recentering.  We'll compare
        a phase=90 cosine (even function) with decenter, a 90 degree clock,
        and matched xi/eta choices so it is symmetric under 90 deg rotations
        since it's even.  If they'd been decentered before clocking, the two
        will not line up.
        """
        tol = 1e-13

        p0 = probe(nact=self.nact, dact=self.dact, xcenter=8,
              ycenter=13, clock=0, height=self.height,
              ximin=-12, ximax=12, etamin=0,
              etamax=12, phase=90)

        p90 = probe(nact=self.nact, dact=self.dact, xcenter=8,
              ycenter=13, clock=90, height=self.height,
              ximin=-12, ximax=12, etamin=0,
              etamax=12, phase=90)

        self.assertTrue(np.max(np.abs(p0-p90)) < tol)
        pass


    def test_output_size(self):
        """Check outputs match docs"""
        p0 = probe(nact=self.nact, dact=self.dact, xcenter=8,
              ycenter=13, clock=0, height=self.height,
              ximin=-12, ximax=12, etamin=0,
              etamax=12, phase=90)
        self.assertTrue(p0.shape == (self.nact, self.nact))
        pass


class TestSine(unittest.TestCase):
    """
    Unit tests for function to generate a DM sine
    """
    def setUp(self):
        """
        Predefine OK parameters so we don't have to repeat them
        """
        self.nact = 48
        self.dact = 46.3
        self.amplitude = 100e-9
        self.phase = 0
        self.freqx = 8
        self.freqy = 0
        pass


    # inputs
    def test_success(self):
        """
        Run with valid inputs should not throw anything
        """
        sine(nact=self.nact, dact=self.dact, amplitude=self.amplitude,
             phase=self.phase, freqx=self.freqx, freqy=self.freqy)
        pass

    def test_output_size(self):
        """Check outputs match docs"""
        s = sine(nact=self.nact, dact=self.dact, amplitude=self.amplitude,
             phase=self.phase, freqx=self.freqx, freqy=self.freqy)
        self.assertTrue(s.shape == (self.nact, self.nact))
        pass

    def test_nact_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, [5], -1, -1.5, 1.5, 0]:
            with self.assertRaises(TypeError):
                sine(nact=perr, dact=self.dact, amplitude=self.amplitude,
                     phase=self.phase, freqx=self.freqx, freqy=self.freqy)
                pass
            pass
        pass


    def test_dact_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, [5], -1, -1.5, 0]:
            with self.assertRaises(TypeError):
                sine(nact=self.nact, dact=perr, amplitude=self.amplitude,
                     phase=self.phase, freqx=self.freqx, freqy=self.freqy)
                pass
            pass
        pass


    def test_amplitude_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, [5], -1, -1.5, 0]:
            with self.assertRaises(TypeError):
                sine(nact=self.nact, dact=self.dact, amplitude=perr,
                     phase=self.phase, freqx=self.freqx, freqy=self.freqy)
                pass
            pass
        pass


    def test_phase_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, [5]]:
            with self.assertRaises(TypeError):
                sine(nact=self.nact, dact=self.dact, amplitude=self.amplitude,
                     phase=perr, freqx=self.freqx, freqy=self.freqy)
                pass
            pass
        pass


    def test_freqx_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, [5]]:
            with self.assertRaises(TypeError):
                sine(nact=self.nact, dact=self.dact, amplitude=self.amplitude,
                     phase=self.phase, freqx=perr, freqy=self.freqy)
                pass
            pass
        pass


    def test_freqy_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, [5]]:
            with self.assertRaises(TypeError):
                sine(nact=self.nact, dact=self.dact, amplitude=self.amplitude,
                     phase=self.phase, freqx=self.freqx, freqy=perr)
                pass
            pass
        pass

    def test_outputs_sinusoids(self):
        """
        Verify sin^2(x) + cos^2(x) = 1 everywhere on DM
        """
        tol = 1e-13

        SX = sine(nact=self.nact, dact=self.dact, amplitude=self.amplitude,
             phase=0, freqx=self.freqx, freqy=self.freqy)
        CX = sine(nact=self.nact, dact=self.dact, amplitude=self.amplitude,
             phase=90, freqx=self.freqx, freqy=self.freqy)

        self.assertTrue(np.max(np.abs(SX**2 + CX**2 - self.amplitude**2))
                         < tol)
        pass

    def test_180_phase(self):
        """
        Verify 180 degrees out of phase is the negative of the original
        """
        tol = 1e-13

        for phase in [0, 23.8942, 45, 90, 180]:
            s = sine(nact=self.nact, dact=self.dact, amplitude=self.amplitude,
                 phase=phase, freqx=self.freqx, freqy=self.freqy)
            s180 = sine(nact=self.nact, dact=self.dact,
                        amplitude=self.amplitude,
                        phase=phase+180, freqx=self.freqx, freqy=self.freqy)
            self.assertTrue(np.max(np.abs(s + s180)) < tol)
        pass

    def test_dact_scales_as_expected(self):
        """
        dact sets the diameter, and the spatial frequencies are referenced to
        the diameter (cycles per dact-sized aperture).  Halving dact and
        halving the spatial frequencies should produce the same DM setting
        (fewer cycles across smaller aperture).  This test verifies this.
        """
        tol = 1e-13

        for fx, fy in [(0, 8), (8, 0), (6, 6), (-6, -6)]:
            s = sine(nact=self.nact, dact=self.dact,
                     amplitude=self.amplitude, phase=self.phase,
                     freqx=fx, freqy=fy)
            shalf = sine(nact=self.nact, dact=self.dact/2.,
                     amplitude=self.amplitude, phase=self.phase,
                     freqx=fx/2., freqy=fy/2.)
            self.assertTrue(np.max(np.abs(s-shalf)) < tol)
            pass

        pass

    def test_cos_even(self):
        """
        Verify cosine (phase=90) is even (i.e. f(x) = f(-x))
        """
        tol = 1e-13

        for fx in [8, 4, 2]:
            c = sine(nact=self.nact, dact=self.dact,
                     amplitude=self.amplitude, phase=90,
                     freqx=fx, freqy=0)
            cn = sine(nact=self.nact, dact=self.dact,
                     amplitude=self.amplitude, phase=90,
                     freqx=-fx, freqy=0)
            self.assertTrue(np.max(np.abs(c-cn)) < tol)
            pass

        for fy in [8, 4, 2]:
            c = sine(nact=self.nact, dact=self.dact,
                     amplitude=self.amplitude, phase=90,
                     freqx=0, freqy=fy)
            cn = sine(nact=self.nact, dact=self.dact,
                     amplitude=self.amplitude, phase=90,
                     freqx=0, freqy=fy)
            self.assertTrue(np.max(np.abs(c-cn)) < tol)
            pass
        pass


    def test_sin_odd(self):
        """
        Verify sine (phase=0) is odd (i.e. f(x) = -f(-x))
        """
        tol = 1e-13

        for fx in [8, 4, 2]:
            s = sine(nact=self.nact, dact=self.dact,
                     amplitude=self.amplitude, phase=0,
                     freqx=fx, freqy=0)
            sn = sine(nact=self.nact, dact=self.dact,
                     amplitude=self.amplitude, phase=0,
                     freqx=-fx, freqy=0)
            self.assertTrue(np.max(np.abs(s+sn)) < tol)
            pass

        for fy in [8, 4, 2]:
            s = sine(nact=self.nact, dact=self.dact,
                     amplitude=self.amplitude, phase=0,
                     freqx=0, freqy=fy)
            sn = sine(nact=self.nact, dact=self.dact,
                     amplitude=self.amplitude, phase=0,
                     freqx=0, freqy=-fy)
            self.assertTrue(np.max(np.abs(s+sn)) < tol)
            pass
        pass


    def test_x_direction(self):
        """
        Verify a sinusoid with only spatial frequency components in x is
        constant-valued along y (i.e. got direction right)
        """
        tol = 1e-13

        s = sine(nact=self.nact, dact=self.dact,
                 amplitude=self.amplitude, phase=self.phase,
                 freqx=6, freqy=0)
        for col in range(s.shape[1]):
            self.assertTrue((np.max(s[:, col])-np.min(s[:, col])) < tol)
            pass
        pass

    def test_y_direction(self):
        """
        Verify a sinusoid with only spatial frequency components in y is
        constant-valued along x (i.e. got direction right)
        """
        tol = 1e-13

        s = sine(nact=self.nact, dact=self.dact,
                 amplitude=self.amplitude, phase=self.phase,
                 freqx=0, freqy=6)
        for row in range(s.shape[0]):
            self.assertTrue((np.max(s[row, :])-np.min(s[row, :])) < tol)
            pass
        pass

    def test_diag_direction(self):
        """
        Verify a sinusoid with equal spatial frequency components in x and y
        is constant along the opposite diagonals (i.e. got x/y orientation
        right)
        """
        tol = 1e-13

        s = sine(nact=self.nact, dact=self.dact,
                 amplitude=self.amplitude, phase=self.phase,
                 freqx=6, freqy=6)
        for col in range(1, s.shape[1]): # skip scalar corner
            vec = np.zeros((col,))
            for row in range(col):
                vec[row] = s[row, col-row]
                pass
            self.assertTrue((np.max(vec)-np.min(vec)) < tol)
            pass
        pass


    def test_spatial_freq_xy(self):
        """
        Verify that a spatial frequency placed on the DM is actually at that
        frequency
        """
        tol = 1e-13

        dact = 48
        dx = 8

        # Along X
        sx = sine(nact=48, dact=dact, amplitude=1, phase=0,
                 freqx=dact/dx, freqy=0)
        # Same sign at one period
        self.assertTrue(np.abs(sx[0, 0] - sx[0, dx]) < tol)
        # Opposite sign at half period
        self.assertTrue(np.abs(sx[0, 0] + sx[0, dx//2]) < tol)

        # Along y
        sy = sine(nact=48, dact=dact, amplitude=1, phase=0,
                 freqx=0, freqy=dact/dx)
        # Same sign at one period
        self.assertTrue(np.abs(sy[0, 0] - sy[dx, 0]) < tol)
        # Opposite sign at half period
        self.assertTrue(np.abs(sy[0, 0] + sy[dx//2, 0]) < tol)

        pass




class TestGrid(unittest.TestCase):
    """
    Unit tests for function to generate a DM grid
    """
    def setUp(self):
        """
        Predefine OK parameters so we don't have to repeat them
        """
        self.nact = 48
        self.xspace = 4
        self.yspace = 4
        self.xoffset = 0
        self.yoffset = 0
        self.pokesize = 20e-9
        pass


    def test_success(self):
        """
        Run with valid inputs should not throw anything
        """
        grid(nact=self.nact, xspace=self.xspace, yspace=self.yspace,
             xoffset=self.xoffset, yoffset=self.yoffset,
             pokesize=self.pokesize)
        pass


    def test_output_size(self):
        """Check outputs match docs"""
        g = grid(nact=self.nact, xspace=self.xspace, yspace=self.yspace,
             xoffset=self.xoffset, yoffset=self.yoffset,
             pokesize=self.pokesize)
        self.assertTrue(g.shape == (self.nact, self.nact))
        pass


    def test_nact_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, [5], -1, -1.5, 1.5, 0]:
            with self.assertRaises(TypeError):
                grid(nact=perr, xspace=self.xspace, yspace=self.yspace,
                     xoffset=self.xoffset, yoffset=self.yoffset,
                     pokesize=self.pokesize)
                pass
            pass
        pass

    def test_xspace_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, [5], -1, -1.5, 1.5, 0]:
            with self.assertRaises(TypeError):
                grid(nact=self.nact, xspace=perr, yspace=self.yspace,
                     xoffset=self.xoffset, yoffset=self.yoffset,
                     pokesize=self.pokesize)
                pass
            pass
        pass

    def test_yspace_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, [5], -1, -1.5, 1.5, 0]:
            with self.assertRaises(TypeError):
                grid(nact=self.nact, xspace=self.xspace, yspace=perr,
                     xoffset=self.xoffset, yoffset=self.yoffset,
                     pokesize=self.pokesize)
                pass
            pass
        pass

    def test_xoffset_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, [5], -1, -1.5, 1.5]:
            with self.assertRaises(TypeError):
                grid(nact=self.nact, xspace=self.xspace, yspace=self.yspace,
                     xoffset=perr, yoffset=self.yoffset,
                     pokesize=self.pokesize)
                pass
            pass
        pass

    def test_yoffset_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, [5], -1, -1.5, 1.5]:
            with self.assertRaises(TypeError):
                grid(nact=self.nact, xspace=self.xspace, yspace=self.yspace,
                     xoffset=self.xoffset, yoffset=perr,
                     pokesize=self.pokesize)
                pass
            pass
        pass

    def test_pokesize_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, [5]]:
            with self.assertRaises(TypeError):
                grid(nact=self.nact, xspace=self.xspace, yspace=self.yspace,
                     xoffset=self.xoffset, yoffset=self.yoffset,
                     pokesize=perr)
                pass
            pass
        pass

    def test_xyspace_beyond_nact(self):
        """
        Verify spacings larger than the DM are caught
        """
        with self.assertRaises(ValueError):
            grid(nact=self.nact, xspace=self.nact+1, yspace=self.yspace,
                 xoffset=self.xoffset, yoffset=self.yoffset,
                 pokesize=self.pokesize)
            pass

        with self.assertRaises(ValueError):
            grid(nact=self.nact, xspace=self.xspace, yspace=self.nact+1,
                 xoffset=self.xoffset, yoffset=self.yoffset,
                 pokesize=self.pokesize)
            pass
        pass

    def test_xyoffset_beyond_xyspace(self):
        """
        Verify offsets larger than the spacing are caught
        """
        with self.assertRaises(ValueError):
            grid(nact=self.nact, xspace=self.xspace, yspace=self.yspace,
                 xoffset=self.xspace, yoffset=self.yoffset,
                 pokesize=self.pokesize)
            pass

        with self.assertRaises(ValueError):
            grid(nact=self.nact, xspace=self.xspace, yspace=self.yspace,
                 xoffset=self.xoffset, yoffset=self.yspace,
                 pokesize=self.pokesize)
            pass
        pass

    def test_xyspace_edge_cases_valid(self):
        """
        Verify xspace/yspace edge cases are not caught (full range available)
        """
        grid(nact=self.nact, xspace=1, yspace=self.yspace,
             xoffset=self.xoffset, yoffset=self.yoffset,
             pokesize=self.pokesize)
        grid(nact=self.nact, xspace=self.nact, yspace=self.yspace,
             xoffset=self.xoffset, yoffset=self.yoffset,
             pokesize=self.pokesize)
        grid(nact=self.nact, xspace=self.xspace, yspace=1,
             xoffset=self.xoffset, yoffset=self.yoffset,
             pokesize=self.pokesize)
        grid(nact=self.nact, xspace=self.xspace, yspace=self.nact,
             xoffset=self.xoffset, yoffset=self.yoffset,
             pokesize=self.pokesize)
        pass

    def test_xyoffset_edge_cases_valid(self):
        """
        Verify xoffset/yoffset edge cases are not caught (full range available)
        """
        grid(nact=self.nact, xspace=self.xspace, yspace=self.yspace,
             xoffset=0, yoffset=self.yoffset,
             pokesize=self.pokesize)
        grid(nact=self.nact, xspace=self.xspace, yspace=self.yspace,
             xoffset=self.xspace-1, yoffset=self.yoffset,
             pokesize=self.pokesize)
        grid(nact=self.nact, xspace=self.xspace, yspace=self.yspace,
             xoffset=self.xoffset, yoffset=0,
             pokesize=self.pokesize)
        grid(nact=self.nact, xspace=self.xspace, yspace=self.yspace,
             xoffset=self.xoffset, yoffset=self.yspace-1,
             pokesize=self.pokesize)
        pass


    def test_neg_pokevalid(self):
        """
        Verify pokesize works with both positive and negative poke amplitudes
        as expected
        """
        grid(nact=self.nact, xspace=self.xspace, yspace=self.yspace,
             xoffset=self.xoffset, yoffset=self.yoffset,
             pokesize=np.abs(self.pokesize))
        grid(nact=self.nact, xspace=self.xspace, yspace=self.yspace,
             xoffset=self.xoffset, yoffset=self.yoffset,
             pokesize=-np.abs(self.pokesize))
        pass

    def test_poke_all_or_none(self):
        """
        Verify that a poked array is either 0 or pokesize everywhere
        """
        g = grid(nact=self.nact, xspace=self.xspace, yspace=self.yspace,
             xoffset=self.xoffset, yoffset=self.yoffset,
             pokesize=self.pokesize)
        self.assertTrue(np.logical_or((g == 0), (g == self.pokesize)).all())
        pass

    def test_specific_cases_vs_expectation(self):
        """
        Verify specific cases show up as planned
        Pick these to have x/y asymmetries, so they break if we get the
        convention wrong
        """

        # In writing arrays line-by-line, r[0,0] is upper left, r[x,0] moves
        # down rows top-to-bottom, and r[0,x] moves along columns
        r6 = np.array([[0, 0, 0, 0, 0, 0],
                       [1, 0, 1, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [1, 0, 1, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0]])

        r7 = np.array([[0, 0, 0, 0, 0, 0, 0],
                       [1, 0, 1, 0, 1, 0, 1],
                       [0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0],
                       [1, 0, 1, 0, 1, 0, 1],
                       [0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0]])

        piston = np.ones((8, 8))

        g6 = grid(6, 2, 3, 0, 1, 1)
        g7 = grid(7, 2, 3, 0, 1, 1)
        g8 = grid(8, 1, 1, 0, 0, 1)

        self.assertTrue((r6 == g6).all())
        self.assertTrue((r7 == g7).all())
        self.assertTrue((piston == g8).all())
        pass


class TestZernike(unittest.TestCase):
    """
    Unit tests for function to generate a Zernike polynomial on a grid
    """

    def setUp(self):
        """
        Predefine OK parameters so we don't have to repeat them
        """
        self.nact = 48
        self.dact = 46.3
        self.zernsize = 1e-9
        pass


    def test_success(self):
        """
        Run with valid inputs should not throw anything
        """
        order = 4
        zernike(nact=self.nact, dact=self.dact, zernsize=self.zernsize,
                order=order)
        pass

    def test_output_size(self):
        """
        Check output sizes match docs
        """
        order = 4
        z = zernike(nact=self.nact, dact=self.dact, zernsize=self.zernsize,
                order=order)
        self.assertTrue(z.shape == (self.nact, self.nact))
        pass


    def test_nact_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        order = 4

        for perr in [None, 'txt', 1j, [5], -1, -1.5, 1.5, 0]:
            with self.assertRaises(TypeError):
                zernike(nact=perr, dact=self.dact,
                        zernsize=self.zernsize, order=order)
                pass
            pass
        pass

    def test_dact_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        order = 4

        for perr in [None, 'txt', 1j, [5], -1, -1.5, 0]:
            with self.assertRaises(TypeError):
                zernike(nact=self.nact, dact=perr,
                        zernsize=self.zernsize, order=order)
                pass
            pass
        pass

    def test_zernsize_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        order = 4

        for perr in [None, 'txt', 1j, [5]]:
            with self.assertRaises(TypeError):
                zernike(nact=self.nact, dact=self.dact,
                        zernsize=perr, order=order)
                pass
            pass
        pass

    def test_order_invalid(self):
        """
        Test bad parameter values fail as expected
        """
        for perr in [None, 'txt', 1j, [5], -1, -1.5, 1.5, 0]:
            with self.assertRaises(TypeError):
                zernike(nact=self.nact, dact=self.dact,
                        zernsize=self.zernsize, order=perr)
                pass
            pass
        pass

    def test_piston(self):
        """
        Test that a piston (order==1) produces a uniform setting with the
        correct rms.

        Implementation note: the combination of the correct rms in this test
        with the cross-Zernike normalization check in test_normalization() is
        is intended to verify that all normalizations are absolutely correct.
        Verifying the rms directly on a non-piston is not very effective as the
        x-y grids are generally coarse and don't add up exactly; the difference
        can be reduced by larger grid sizes but will generally never be exact
        enough to test in an automated unit test fashion.  It may still be a
        useful thing to check by eye by using a large nact and doing
         np.sqrt(np.mean(np.square()))
        """
        rms = 1.0
        tol = 1e-13

        piston = zernike(nact=self.nact, dact=self.dact,
                         zernsize=rms, order=1)
        self.assertTrue(np.max(np.abs(piston - rms)) < tol)
        pass

    def test_normalization(self):
        """
        Spot-check that the zernikes are being normalized relative to each
        other using the analytic form of the lowest Noll terms:

         Z1 = 1
         Z2 = 2r cos(theta)
         Z3 = 2r sin(theta)
         Z4 = sqrt(3)*(2r**2 - 1)
         Z5 = sqrt(6)*r**2 sin(2*theta)
         Z6 = sqrt(6)*r**2 cos(2*theta)

        which can be combined nonlinearly as, e.g.
         Z2**2 + Z3**2 - 2*Z1 == 2/sqrt(3)*Z4
         sqrt(Z5**2 + Z6**2)/sqrt(6) == (Z2**2 + Z3**2)/4
        """
        rms = 1.0
        tol = 1e-13

        Z1 = zernike(nact=self.nact, dact=self.dact,
                         zernsize=rms, order=1)
        Z2 = zernike(nact=self.nact, dact=self.dact,
                         zernsize=rms, order=2)
        Z3 = zernike(nact=self.nact, dact=self.dact,
                         zernsize=rms, order=3)
        Z4 = zernike(nact=self.nact, dact=self.dact,
                         zernsize=rms, order=4)
        Z5 = zernike(nact=self.nact, dact=self.dact,
                         zernsize=rms, order=5)
        Z6 = zernike(nact=self.nact, dact=self.dact,
                         zernsize=rms, order=6)

        self.assertTrue(np.max(np.abs(Z2**2 + Z3**2 - 2*Z1 - 2/np.sqrt(3)*Z4))
                        < tol)
        self.assertTrue(np.max(np.abs(np.sqrt((Z5**2 + Z6**2)/6)
                                      - (Z2**2 + Z3**2)/4)) < tol)
        pass

    def test_odd_even(self):
        """
        Spot-check that Zernike polynomials with two orthogonal components
         (e.g. tilt, astigmatism, coma, trefoil...) have the correct phasing
         assigned to each.  Noll convention is that two will be adjacently
         numbered (Z2/Z3, Z5/Z6, etc.) and the odd-valued one will be the sine
         (which is an odd function) and the even-valued one will be the cosine
         (which is an even function).

        We test odd/even-ness by flipping the array about x.  Odd functions
         will go f --> -f, even functions will go f --> f

        We'll skip the radially-symmetric ones (Z1, Z4, Z11, ...) as these
         don't have sin/cos terms to mix up (and are always even anyway)
        """
        rms = 1.0
        tol = 1e-13

        # Z2/Z3, Z5/Z6, Z7/Z8, Z9/Z10, Z12/Z13, Z14/Z15
        for odd in [3, 5, 7, 9, 13, 15]:
            ddm = zernike(nact=self.nact, dact=self.dact,
                          zernsize=rms, order=odd)
            self.assertTrue(np.max(np.abs(ddm - (-np.flipud(ddm)))) < tol)
            pass

        for even in [2, 6, 8, 10, 12, 14]:
            ddm = zernike(nact=self.nact, dact=self.dact,
                          zernsize=rms, order=even)
            self.assertTrue(np.max(np.abs(ddm - (np.flipud(ddm)))) < tol)
            pass

        pass




if __name__ == '__main__':
    unittest.main()
