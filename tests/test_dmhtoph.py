# pylint: disable=unsubscriptable-object
"""Unit tests for dmhtoph.py."""

import unittest
import os

import numpy as np
from astropy.io import fits

from coralign.util.dmhtoph import (
    dmhtoph, dmh_to_volts, volts_to_dmh
)
from coralign.util.pad_crop import pad_crop

# set up some defaults
localpath = os.path.dirname(os.path.abspath(__file__))
fninf_pist = os.path.join(localpath, 'testdata',
                          'ut_influence_dm5v2_inffix.fits')

p = fits.getdata(fninf_pist)
ppa = 10  # correct value for this shape

# Test shapes
nact = 48
dmin0 = np.ones((nact, nact))  # all acts
dmin1 = np.ones((nact, nact))
dmin1[1:-1, 1:-1] = 0  # outer acts
dmin2 = np.zeros((nact, nact))
dmin2[0, 0] = 1  # one act


class TestDmhtoph(unittest.TestCase):
    """Unit test suite for dmhtoph()."""

    def test_outputs(self):
        """Check outputs are nrow x ncol."""
        for nrow, ncol in [(600, 600),
                           (600, 300),
                           (300, 600),
                           (601, 601)]:
            d0 = dmhtoph(nrow=nrow, ncol=ncol,
                         dmin=dmin0, nact=nact, inf_func=p, ppact_d=ppa,
                         ppact_cx=ppa-1, ppact_cy=ppa-1,
                         dx=0, dy=0, thact=0, flipx=False)
            self.assertTrue(d0.shape == (nrow, ncol))
            pass

        pass


    def test_array_symmetry(self):
        """
        Verify that with no translation/rotation/scale and a symmetric input
        we get a symmetric output.  Use odd to be pixel-centered.
        """
        tol = 1e-13
        d0 = dmhtoph(nrow=601, ncol=601,
                     dmin=dmin0, nact=nact, inf_func=p, ppact_d=ppa,
                     ppact_cx=ppa-1, ppact_cy=ppa-1,
                     dx=0, dy=0, thact=0, flipx=False)
        self.assertTrue((np.abs(d0-np.fliplr(np.flipud(d0))) < tol).all())
        pass

    # Success tests
    def test_convolution_works(self):
        """Convolve one actuator and check it matches influence function."""
        tol = 1e-13
        N = p.shape[0]
        nrow = (nact - 1)*ppa + N
        ncol = (nact - 1)*ppa + N
        d0 = dmhtoph(nrow=nrow, ncol=ncol,
                     dmin=dmin2, nact=nact, inf_func=p, ppact_d=ppa,
                     ppact_cx=ppa, ppact_cy=ppa,
                     dx=0, dy=0, thact=0, flipx=False)
        self.assertTrue((np.abs(d0[:N, :N] - p) < tol).all())
        pass

    def test_output_real(self):
        """Verify output is real-valued."""
        for pcxy in [(10, 10), (4, 4), (4.5, 4), (4, 4.5), (4.5, 4.5)]:
            d0 = dmhtoph(nrow=601, ncol=601,
                         dmin=dmin1, nact=nact, inf_func=p, ppact_d=ppa,
                         ppact_cx=pcxy[0], ppact_cy=pcxy[1],
                         dx=0, dy=0, thact=0, flipx=False)
            self.assertTrue(np.isreal(d0).all())
            pass
        pass

    def test_flipx_flips_as_expected(self):
        """
        Verify DM flip flag matches theory expectation; odd sizing means
        arrays are symmetric about center.
        """
        dmin2a = np.zeros((nact-1, nact-1))
        dmin2a[0, -1] = 1
        tol = 1e-13

        d2 = dmhtoph(nrow=601, ncol=601,
                     dmin=dmin2[:-1, :-1], nact=nact-1, inf_func=p,
                     ppact_d=ppa,
                     ppact_cx=ppa-1, ppact_cy=ppa-1,
                     dx=0, dy=0, thact=0, flipx=False)
        d2a = dmhtoph(nrow=601, ncol=601,
                      dmin=dmin2a, nact=nact-1, inf_func=p, ppact_d=ppa,
                      ppact_cx=ppa-1, ppact_cy=ppa-1,
                      dx=0, dy=0, thact=0, flipx=True)
        self.assertTrue((np.abs(d2a-d2) < tol).all())
        pass

    def test_thact_rotates_as_expected(self):
        """
        Verify DM rotate matches theory expectations; odd sizing means
        arrays are symmetric about center.
        """
        tol = 1e-13

        # asymmetric pattern
        dminF1 = np.zeros((nact-1, nact-1))
        dminF1[:, 0] = 1
        dminF1[nact//2, :nact//2] = 1
        dminF1[-1, :] = 1

        # 90 deg counterclockwise
        dminF2 = np.rot90(dminF1, 3)

        dF1 = dmhtoph(nrow=601, ncol=601,
                      dmin=dminF1, nact=nact-1, inf_func=p, ppact_d=ppa,
                      ppact_cx=ppa-1, ppact_cy=ppa-1,
                      dx=0, dy=0, flipx=False, thact=90)

        dF2 = dmhtoph(nrow=601, ncol=601,
                      dmin=dminF2, nact=nact-1, inf_func=p, ppact_d=ppa,
                      ppact_cx=ppa-1, ppact_cy=ppa-1,
                      dx=0, dy=0, flipx=False, thact=0)
        self.assertTrue((np.abs(dF1-dF2) < tol).all())
        pass

    def test_dxdy_shift_as_expected(self):
        """Verify DM x/y shift matches theory expectations"""
        # use roll to shift by integer amounts
        tol = 1e-13

        d0 = dmhtoph(nrow=601, ncol=601,
                     dmin=dmin0, nact=nact, inf_func=p, ppact_d=ppa,
                     ppact_cx=4, ppact_cy=4,
                     dx=0, dy=0, thact=0, flipx=False)

        xylist = [(5, 0), (0, 4), (-6, 0), (0, -5), (2, 3), (-4, 9)]
        biggest = np.max(np.abs(xylist)).astype('int')

        for xy in xylist:
            ddx = dmhtoph(nrow=601, ncol=601,
                          dmin=dmin0, nact=nact, inf_func=p, ppact_d=ppa,
                          ppact_cx=4, ppact_cy=4,
                          dx=xy[0], dy=xy[1], thact=0, flipx=False)
            rolld0 = np.roll(np.roll(d0, xy[0], axis=1), xy[1], axis=0)
            dim = (ddx.shape[0] - 2*biggest).astype('int')

            # roll moves data from one edge over to the other.  This data is
            # not necessarily correct for that location, so we'll trim the
            # edges that were rolled and check the center.
            trim_ddx = pad_crop(ddx, (dim, dim))
            trim_rd0 = pad_crop(rolld0, (dim, dim))

            self.assertTrue((np.abs(trim_rd0 - trim_ddx) < tol).all())
            pass
        pass

    def test_piston_normalization(self):
        """
        "inffix" influence function is a special case; it's been adjusted
        so DM piston of 1 gives a surface piston of 1.  Check DM code
        preserves this behavior.  Saved file has this property
        """
        tol = 1e-6
        pist_height = 3.0
        dpist = pist_height*np.ones((nact, nact))

        N = p.shape[0]
        nrow = (nact - 1)*ppa + N
        ncol = (nact - 1)*ppa + N

        for pcxy in [(10, 10), (4, 4), (4.5, 5), (6, 5.5)]:
            ppact_cx = pcxy[0]
            ppact_cy = pcxy[1]

            d0 = dmhtoph(nrow=nrow, ncol=ncol,
                         dmin=dpist, nact=nact, inf_func=p, ppact_d=ppa,
                         ppact_cx=ppact_cx, ppact_cy=ppact_cy,
                         dx=0, dy=0, thact=0, flipx=False)
            scx = slice(int(ncol//2 - ppact_cx*(nact/8) + 0.5),
                        int(ncol//2 + ppact_cx*(nact/8) + 0.5))
            scy = slice(int(nrow//2 - ppact_cy*(nact/8) + 0.5),
                        int(nrow//2 + ppact_cy*(nact/8) + 0.5))
            self.assertTrue(np.abs(pist_height-d0[scy, scx].mean()) < tol)
            pass

        pass

    def test_asymmetric_convolution_works(self):
        """
        convolve one asymmetric actuator, shifted to be pixel-aligned with
        original corner, and check it matches influence function
        should ensure we don't flip/rotate inf function.
        """
        tol = 1e-13

        F = (p + np.roll(p, 4, axis=1)
             + np.roll(p, -4, axis=1)
             + np.roll(p, 2, axis=0)
             + np.roll(np.roll(p, 4, axis=1), 4, axis=0))/5.

        N = F.shape[0]
        nrow = (nact - 1)*ppa + N
        ncol = (nact - 1)*ppa + N

        d0 = dmhtoph(nrow=nrow, ncol=ncol,
                     dmin=dmin2, nact=nact, inf_func=F, ppact_d=ppa,
                     ppact_cx=ppa, ppact_cy=ppa,
                     dx=0, dy=0, thact=0, flipx=False)
        self.assertTrue((np.abs(d0[:N, :N] - F) < tol).all())
        pass

    # Failure tests
    def test_nrow_positive_scalar_integer(self):
        """Check number-of-actuator type valid."""
        for perr in [[], -48, 0, (48,), 48.0, (6, 6), 1j, 'nact']:
            with self.assertRaises(TypeError):
                dmhtoph(nrow=perr, ncol=601,
                        dmin=dmin0, nact=nact, inf_func=p, ppact_d=ppa,
                        ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                        flipx=False)
                pass
            pass
        pass

    def test_ncol_positive_scalar_integer(self):
        """Check number-of-actuator type valid."""
        for perr in [[], -48, 0, (48,), 48.0, (6, 6), 1j, 'nact']:
            with self.assertRaises(TypeError):
                dmhtoph(nrow=601, ncol=perr,
                        dmin=dmin0, nact=nact, inf_func=p, ppact_d=ppa,
                        ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                        flipx=False)
                pass
            pass
        pass

    def test_dmin_real(self):
        """Check input DM real."""
        dminc = 1j*dmin0
        with self.assertRaises(TypeError):
            dmhtoph(nrow=601, ncol=601,
                    dmin=dminc, nact=nact, inf_func=p, ppact_d=ppa,
                    ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                    flipx=False)
            pass
        pass

    def test_dmin_2Darray(self):
        """Check input DM type valid."""
        for dmin in [np.ones((48,)), np.ones((48, 48, 48)), 'dmin', []]:
            with self.assertRaises(TypeError):
                dmhtoph(nrow=601, ncol=601,
                        dmin=dmin, nact=nact, inf_func=p, ppact_d=ppa,
                        ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                        flipx=False)
                pass
            pass
        pass

    def test_nact_positive_scalar_integer(self):
        """Check number-of-actuator type valid."""
        for nacte in [[], -48, 0, (48,), 48.0, (6, 6), 1j, 'nact']:
            with self.assertRaises(TypeError):
                dmhtoph(nrow=601, ncol=601,
                        dmin=dmin0, nact=nacte, inf_func=p, ppact_d=ppa,
                        ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                        flipx=False)
                pass
            pass
        pass

    def test_dmin_with_nonnact_sides(self):
        """Check DM size and nact are self-consistent."""
        for dmin in [np.ones((nact+1, nact+1)),
                     np.ones((nact, nact+1)),
                     np.ones((nact+1, nact))]:
            with self.assertRaises(TypeError):
                dmhtoph(nrow=601, ncol=601,
                        dmin=dmin, nact=nact, inf_func=p, ppact_d=ppa,
                        ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                        flipx=False)
                pass
            pass
        pass

    def test_inf_func_2Darray(self):
        """Check influence function array type valid."""
        for perr in [np.ones((20,)), np.ones((10, 10, 10)), [], 'inf']:
            with self.assertRaises(TypeError):
                dmhtoph(nrow=601, ncol=601,
                        dmin=dmin0, nact=nact, inf_func=perr, ppact_d=ppa,
                        ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                        flipx=False)
                pass
            pass
        pass

    def test_inf_func_square(self):
        """Check influence function array is square."""
        for perr in [np.ones((91, 90)), np.ones((90, 91))]:
            with self.assertRaises(TypeError):
                dmhtoph(nrow=601, ncol=601,
                        dmin=dmin0, nact=nact, inf_func=perr, ppact_d=ppa,
                        ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                        flipx=False)
                pass
            pass
        pass

    def test_inf_func_odd(self):
        """Check influence function array is odd-sized."""
        for perr in [np.ones((90, 90))]:
            with self.assertRaises(TypeError):
                dmhtoph(nrow=601, ncol=601,
                        dmin=dmin0, nact=nact, inf_func=perr, ppact_d=ppa,
                        ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                        flipx=False)
                pass
            pass
        pass

    def test_ppactd_realpositivescalar(self):
        """Check influence function scaling type valid."""
        for perr in [-1.5, -1, 0, 1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                dmhtoph(nrow=601, ncol=601,
                        dmin=dmin0, nact=nact, inf_func=p, ppact_d=perr,
                        ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                        flipx=False)
                pass
            pass
        pass

    def test_ppactcx_realpositivescalar(self):
        """Check output X scaling type valid."""
        for perr in [-1.5, -1, 0, 1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                dmhtoph(nrow=601, ncol=601,
                        dmin=dmin0, nact=nact, inf_func=p, ppact_d=ppa,
                        ppact_cx=perr, ppact_cy=4, dx=0, dy=0, thact=0,
                        flipx=False)
                pass
            pass
        pass

    def test_ppactcy_realpositivescalar(self):
        """Check output Y scaling type valid."""
        for perr in [-1.5, -1, 0, 1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                dmhtoph(nrow=601, ncol=601,
                        dmin=dmin0, nact=nact, inf_func=p, ppact_d=ppa,
                        ppact_cx=4, ppact_cy=perr, dx=0, dy=0, thact=0,
                        flipx=False)
                pass
            pass
        pass

    def test_dx_realscalar(self):
        """Check offset X scaling type valid."""
        for perr in [1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                dmhtoph(nrow=601, ncol=601,
                        dmin=dmin0, nact=nact, inf_func=p, ppact_d=ppa,
                        ppact_cx=4, ppact_cy=4, dx=perr, dy=0, thact=0,
                        flipx=False)
                pass
            pass
        pass

    def test_dy_realscalar(self):
        """Check offset Y scaling type valid."""
        for perr in [1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                dmhtoph(nrow=601, ncol=601,
                        dmin=dmin0, nact=nact, inf_func=p, ppact_d=ppa,
                        ppact_cx=4, ppact_cy=4, dx=0, dy=perr, thact=0,
                        flipx=False)
                pass
            pass
        pass

    def test_thact_realscalar(self):
        """Check rotation angle type valid."""
        for perr in [1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                dmhtoph(nrow=601, ncol=601,
                        dmin=dmin0, nact=nact, inf_func=p, ppact_d=ppa,
                        ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=perr,
                        flipx=False)
                pass
            pass
        pass

    def test_design_sampled_more_than_camera(self):
        """
        Should fail if your camera sampling is finer than your theoretical
        influence function (because you need a better sampling to downsample).
        """
        for ppactlist in [(10, 11, 11),
                          (10, 4, 11),
                          (10, 11, 4)]:
            pd, pcx, pcy = ppactlist
            with self.assertRaises(TypeError):
                dmhtoph(nrow=601, ncol=601,
                        dmin=dmin0, nact=nact, inf_func=p, ppact_d=pd,
                        ppact_cx=pcx, ppact_cy=pcy, dx=0, dy=0, thact=0,
                        flipx=False)
                pass
            pass
        pass

    def test_inf_func_smaller_than_DM_grid(self):
        """Check that the influence function fails if bigger than DMxo."""
        with self.assertRaises(TypeError):
            pwide = pad_crop(p, (ppa*nact + 1, ppa*nact + 1))
            dmhtoph(nrow=601, ncol=601,
                    dmin=dmin0, nact=nact, inf_func=pwide, ppact_d=ppa,
                    ppact_cx=4, ppact_cy=4, dx=0, dy=0, thact=0,
                    flipx=False)
            pass
        pass


class TestVoltsToDmh(unittest.TestCase):
    """Unit test suite for volts_to_dmh()."""

    def setUp(self):
        """Set some good defaults we can call and save space."""
        self.lam = 500e-9
        self.volts = np.ones((48, 48))
        self.gainmap = 1e-9*np.ones((48, 48))

    def test_correct_conversion(self):
        """Check volts converted to actuator height as expected."""
        volts = 50*np.ones((48, 48))
        dmh = volts_to_dmh(self.gainmap, volts, self.lam)
        self.assertTrue((dmh == 0.2*np.pi*np.ones((48, 48))).all())

    def test_invalid_gainmap(self):
        """Check gainmap array type valid."""
        for gainmap in [np.ones((48,)), np.ones((48, 48, 48)), [], 1, 'txt']:
            with self.assertRaises(TypeError):
                volts_to_dmh(gainmap, self.volts, self.lam)

    def test_invalid_volts(self):
        """Check voltage array type valid."""
        for volts in [np.ones((48,)), np.ones((48, 48, 48)), [], 1, 'txt']:
            with self.assertRaises(TypeError):
                volts_to_dmh(self.gainmap, volts, self.lam)

    def test_invalid_lam(self):
        """Check wavelength type valid."""
        volts = 50*np.ones_like(self.gainmap)
        for lam in [(5,), -8, 0, 1j, [], None, 'text', np.ones_like(volts)]:
            with self.assertRaises(TypeError):
                volts_to_dmh(self.gainmap, self.volts, lam)

    def test_volts_not_same_size_as_gainmap(self):
        """Check behavior when input DM size does not match gainmap size."""
        # 2D array, but not the right one
        for gainmap in [np.ones((47, 48)),
                        np.ones((48, 47)),
                        np.ones((47, 47))]:
            with self.assertRaises(TypeError):
                volts_to_dmh(gainmap, self.volts, self.lam)

        for volts in [np.ones((47, 48)),
                        np.ones((48, 47)),
                        np.ones((47, 47))]:
            with self.assertRaises(TypeError):
                volts_to_dmh(self.gainmap, volts, self.lam)


class TestDmhToVolts(unittest.TestCase):
    """Unit test suite for dmh_to_volts()."""

    def setUp(self):
        """Set some good defaults we can call and save space."""
        self.lam = 500e-9
        self.gainmap = 1e-9*np.ones((48, 48))
        self.dmh = 50*np.ones((48, 48))

    def test_correct_conversion(self):
        """Check actuator height converted to volts as expected."""
        dmh = 0.2*np.pi*np.ones((48, 48))
        volts = dmh_to_volts(self.gainmap, dmh, self.lam)
        self.assertTrue((volts == 50*np.ones((48, 48))).all())

    def test_invalid_gainmap(self):
        """Check actuator height array type valid."""
        for gainmap in [np.ones((48,)), np.ones((48, 48, 48)), [], 1, 'txt']:
            with self.assertRaises(TypeError):
                dmh_to_volts(gainmap, self.dmh, self.lam)

    def test_invalid_dmh(self):
        """Check actuator height array type valid."""
        for dmh in [np.ones((48,)), np.ones((48, 48, 48)), [], 1, 'txt']:
            with self.assertRaises(TypeError):
                dmh_to_volts(self.gainmap, dmh, self.lam)

    def test_invalid_lam(self):
        """Check wavelength type valid."""
        dmh = 0.2*np.pi*np.ones_like(self.gainmap)
        for lam in [(5,), -8, 0, 1j, [], None, 'text', np.ones_like(dmh)]:
            with self.assertRaises(TypeError):
                dmh_to_volts(self.gainmap, self.dmh, lam)

    def test_dmh_not_same_size_as_gainmap(self):
        """Check behavior when input arrays have mismatched shapes."""
        # 2D array, but not the right one
        for gainmap in [np.ones((47, 48)),
                        np.ones((48, 47)),
                        np.ones((47, 47))]:
            with self.assertRaises(TypeError):
                dmh_to_volts(gainmap, self.dmh, self.lam)

        for dmh in [np.ones((47, 48)),
                    np.ones((48, 47)),
                    np.ones((47, 47))]:
            with self.assertRaises(TypeError):
                dmh_to_volts(self.gainmap, dmh, self.lam)



if __name__ == '__main__':
    unittest.main()
