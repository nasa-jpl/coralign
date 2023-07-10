"""
Unit tests for fresnelprop.py
"""

import unittest

import numpy as np

from coralign.util.fresnelprop import fresnelprop
from coralign.util.fresnelprop import do_fft, do_ifft
from coralign.util.fresnelprop import fresnelprop_fp, get_fp


class TestFresnelprop(unittest.TestCase):
    """
    Unit test suite for fresnelprop()
    """

    def test_fresnel_propagation_talbot(self):
        """
        Talbot imaging is one of the few cases where Fresnel propagation
        produces analytic solutions.  In particular, we'll test the three
        special cases of Fresnel propagation from an amplitude grating:
        self-image, contrast reversal, and the frequency-doubled subimage.
        (See e.g. section 4.5.2 in Goodman's Fourier Optics.)
        """
        # pick some parameters
        tol = 1e-13
        m = 0.25
        period = 32 # pixels, needs to be int for roll()
        lam = 633.e-9
        nxfresnel = 2048
        pixpermeter = 4000.0 # needs to be float for division with period

        # set up amplitude grating
        x = np.arange(nxfresnel) - nxfresnel//2
        XX = np.outer(x, np.ones_like(x))
        e0 = 0.5*(1+m*np.cos(2*np.pi*XX/period))
        ec = np.roll(e0, period//2, axis=0)

        # Case 1: self image
        for n in [1, ]:
            z0 = (2*n)*((period/pixpermeter)**2/lam)
            e1 = fresnelprop(e0, lam, z0, nxfresnel, pixpermeter)
            self.assertTrue(np.max(np.abs(e1-e0)) <= tol)
            pass

        # Case 2: contrast reversal
        for n in [1, ]:
            z0 = (2*n - 1)*((period/pixpermeter)**2/lam)
            e1 = fresnelprop(e0, lam, z0, nxfresnel, pixpermeter)
            self.assertTrue(np.max(np.abs(e1-ec)) <= tol)
            pass

        # Case 3: subimage (use intensities as fields will be rotated
        for n in [1, 2]:
            z0 = (n - 0.5)*((period/pixpermeter)**2/lam)
            e1 = fresnelprop(e0, lam, z0, nxfresnel, pixpermeter)
            int1 = np.abs(e1)**2
            int0c = (np.abs(e0)**2 + np.abs(ec)**2)/2.
            self.assertTrue(np.max(np.abs(int1-int0c)) <= tol)
            pass

        pass

    def test_input_output_same_size(self):
        """Verify input and output sizes match."""
        for e in [np.ones((128, 128)), np.ones((127, 127)),
                  np.ones((128, 127)), np.ones((127, 128))]:
            e1 = fresnelprop(e, 633e-9, -1, 1024, 4000)
            self.assertTrue(e1.shape == e.shape)
        pass

    def test_z_negative_ok(self):
        """Check neg z does not throw a TypeError"""
        try:
            fresnelprop(np.ones((128, 128)), 633e-9, -1, 1024, 4000)
        except TypeError:
            self.fail('Input checking incorrectly flagged negative z')
        pass

    def test_z_zero_ok(self):
        """Verify z=0 produces original."""
        e0 = np.ones((128, 128))
        e1 = fresnelprop(e0, 633e-9, 0, 1024, 4000)
        self.assertTrue((e1 == e0).all())
        pass


    # Failure tests
    def test_lam_realpositivescalar(self):
        """Check wavelength type valid."""
        for lam in [(633e-9, 633e-9), [], (633e-9,), 'lam',
                    633e-9*1j, -633e-9, 0]:
            with self.assertRaises(TypeError):
                fresnelprop(np.ones((128, 128)), lam, 1, 1024, 4000)
                pass
            pass
        pass

    def test_z_realscalar(self):
        """Check z type valid."""
        for z in [(1, 1), [], (1,), 'z', 1j]:
            with self.assertRaises(TypeError):
                fresnelprop(np.ones((128, 128)), 633e-9, z, 1024, 4000)
                pass
            pass
        pass

    def test_pixpermeter_realpositivescalar(self):
        """Check scaling type valid."""
        for pixpermeter in [(4000, 4000), [], (4000,), 'pixpermeter',
                            4000*1j, -4000, 0]:
            with self.assertRaises(TypeError):
                fresnelprop(np.ones((128, 128)), 633e-9, 1, 1024, pixpermeter)
                pass
            pass
        pass

    def test_nxfresnel_singlevaluedinteger(self):
        """Check Fresnel array size type valid."""
        # used as arange() input, should be integer or may misbehave
        for nxfresnel in [(1024, 1024), [], (1024,), 'nxfresnel',
                          1024.0, -1024]:
            with self.assertRaises(TypeError):
                fresnelprop(np.ones((128, 128)), 633e-9, 1, nxfresnel, 4000)
                pass
            pass
        pass

    def test_e_is_2D_array(self):
        """Check electric field type valid."""
        for e in [np.ones((10, 10, 10)), [], np.ones((10,)), 'e', 10]:
            with self.assertRaises(TypeError):
                fresnelprop(e, 633e-9, 1, 1024, 4000)
                pass
            pass
        pass

    def test_violated_sampling_criterion(self):
        """Check array size valid for angular spectrum propagation."""
        lam = 575.0e-9
        z = 1.0
        pixpermeter = 8000.
        nxfresnel = int(np.ceil(lam * np.abs(z) * (pixpermeter**2))) - 2

        with self.assertRaises(ValueError):
            fresnelprop(np.eye(nxfresnel), lam, z, nxfresnel, pixpermeter)


class TestFresnelpropFP(unittest.TestCase):
    """Unit test suite for fresnelprop_fp() and get_fp()."""

    def test_fresnel_propagation_talbot(self):
        """
        Talbot imaging is one of the few cases where Fresnel propagation
        produces analytic solutions.  In particular, we'll test the three
        special cases of Fresnel propagation from an amplitude grating:
        self-image, contrast reversal, and the frequency-doubled subimage.
        (See e.g. section 4.5.2 in Goodman's Fourier Optics.)
        """
        # pick some parameters
        tol = 1e-13
        m = 0.25
        period = 32  # pixels, needs to be int for roll()
        lam = 633.e-9
        nxfresnel = 2048
        pixpermeter = 4000.0  # needs to be float for division with period

        # set up amplitude grating
        x = np.arange(nxfresnel) - nxfresnel//2
        XX = np.outer(x, np.ones_like(x))
        e0 = 0.5*(1+m*np.cos(2*np.pi*XX/period))
        ec = np.roll(e0, period//2, axis=0)

        # Case 1: self image
        for n in [1, ]:
            z0 = (2*n)*((period/pixpermeter)**2/lam)
            fp = get_fp(lam, z0, nxfresnel, pixpermeter)
            e1 = fresnelprop_fp(e0, z0, nxfresnel, fp)
            self.assertTrue(np.max(np.abs(e1-e0)) <= tol)
            pass

        # Case 2: contrast reversal
        for n in [1, ]:
            z0 = (2*n - 1)*((period/pixpermeter)**2/lam)
            fp = get_fp(lam, z0, nxfresnel, pixpermeter)
            e1 = fresnelprop_fp(e0, z0, nxfresnel, fp)
            self.assertTrue(np.max(np.abs(e1-ec)) <= tol)
            pass

        # Case 3: subimage (use intensities as fields will be rotated
        for n in [1, 2]:
            z0 = (n - 0.5)*((period/pixpermeter)**2/lam)
            fp = get_fp(lam, z0, nxfresnel, pixpermeter)
            e1 = fresnelprop_fp(e0, z0, nxfresnel, fp)
            int1 = np.abs(e1)**2
            int0c = (np.abs(e0)**2 + np.abs(ec)**2)/2.
            self.assertTrue(np.max(np.abs(int1-int0c)) <= tol)
            pass
        pass

    def test_input_output_same_size(self):
        """Verify input and output sizes match."""
        for e in [np.ones((128, 128)), np.ones((127, 127)),
                  np.ones((128, 127)), np.ones((127, 128))]:
            fp = get_fp(633e-9, -1, 1024, 4000)
            e1 = fresnelprop_fp(e, -1, 1024, fp)
            self.assertTrue(e1.shape == e.shape)
        pass

    def test_z_negative_ok(self):
        """Check neg z does not throw a TypeError."""
        try:
            fp = get_fp(633e-9, -1, 1024, 4000)
            fresnelprop_fp(np.ones((128, 128)), -1, 1024, fp)
        except TypeError:
            self.fail('Input checking incorrectly flagged negative z')
        pass

    def test_z_zero_ok(self):
        """Verify z=0 produces original."""
        e0 = np.ones((128, 128))
        fp = get_fp(633e-9, 0, 1024, 4000)
        e1 = fresnelprop_fp(e0, 0, 1024, fp)
        self.assertTrue((e1 == e0).all())
        pass

    def test_fresnelprop_and_fresnelpropfp_match(self):
        """Check fresnelprop() and fresnelprop_fp() match outputs."""
        e = np.ones((128, 128))
        lam = 633e-9
        z = 1.4
        nxfresnel = 1024
        pixpermeter = 4000

        e0 = fresnelprop(e, lam, z, nxfresnel, pixpermeter)
        fp = get_fp(lam, z, nxfresnel, pixpermeter)
        e1 = fresnelprop_fp(e, z, nxfresnel, fp)
        self.assertTrue((e1 == e0).all())
        pass


    # Failure on fresnelprop_fp tests
    def test_z_realscalar_f(self):
        """Check z type valid."""
        e = np.ones((128, 128)).astype('complex128')
        fp = np.ones((128, 128)).astype('complex128')
        for z in [(1, 1), [], (1,), 'z', 1j]:
            with self.assertRaises(TypeError):
                fresnelprop_fp(e, z, 1024, fp)
                pass
            pass
        pass

    def test_nxfresnel_singlevaluedinteger_f(self):
        """Check Fresnel array size type valid."""
        e = np.ones((128, 128)).astype('complex128')
        fp = np.ones((128, 128)).astype('complex128')
        z = 1

        # used as arange() input, should be integer or may misbehave
        for nxfresnel in [(1024, 1024), [], (1024,), 'nxfresnel',
                          1024.0, -1024]:
            with self.assertRaises(TypeError):
                fresnelprop_fp(e, z, nxfresnel, fp)
                pass
            pass
        pass

    def test_e_is_2D_array_f(self):
        """Check electric field type valid."""
        fp = np.ones((128, 128)).astype('complex128')
        z = 1
        nxfresnel = 1024

        for e in [np.ones((10, 10, 10)), [], np.ones((10,)), 'e', 10]:
            with self.assertRaises(TypeError):
                fresnelprop_fp(e, z, nxfresnel, fp)
                pass
            pass
        pass

    def test_fp_is_2D_array_f(self):
        """Check quadratic phase type valid."""
        e = np.ones((128, 128)).astype('complex128')
        z = 1
        nxfresnel = 1024

        for fp in [np.ones((10, 10, 10)), [], np.ones((10,)), 'fp', 10]:
            with self.assertRaises(TypeError):
                fresnelprop_fp(e, z, nxfresnel, fp)
                pass
            pass
        pass


    # Failure on get_fp tests
    def test_lam_realpositivescalar_g(self):
        """Check wavelength type valid."""
        for lam in [(633e-9, 633e-9), [], (633e-9,), 'lam',
                    633e-9*1j, -633e-9, 0]:
            with self.assertRaises(TypeError):
                get_fp(lam, 1, 1024, 4000)
                pass
            pass
        pass

    def test_z_realscalar_g(self):
        """Check z type valid."""
        for z in [(1, 1), [], (1,), 'z', 1j]:
            with self.assertRaises(TypeError):
                get_fp(633e-9, z, 1024, 4000)
                pass
            pass
        pass

    def test_pixpermeter_realpositivescalar_g(self):
        """Check scaling type valid."""
        for pixpermeter in [(4000, 4000), [], (4000,), 'pixpermeter',
                            4000*1j, -4000, 0]:
            with self.assertRaises(TypeError):
                get_fp(633e-9, 1, 1024, pixpermeter)
                pass
            pass
        pass

    def test_nxfresnel_singlevaluedinteger_g(self):
        """Check Fresnel array size type valid."""
        # used as arange() input, should be integer or may misbehave
        for nxfresnel in [(1024, 1024), [], (1024,), 'nxfresnel',
                          1024.0, -1024]:
            with self.assertRaises(TypeError):
                get_fp(633e-9, 1, nxfresnel, 4000)
                pass
            pass
        pass

    def test_violated_sampling_criterion(self):
        """Check array size valid for angular spectrum propagation."""
        lam = 575.0e-9
        z = 1.0
        pixpermeter = 8000.
        nxfresnel = int(np.ceil(lam * np.abs(z) * (pixpermeter**2))) - 2

        with self.assertRaises(ValueError):
            get_fp(lam, z, nxfresnel, pixpermeter)


class TestDoFFT_IFFT(unittest.TestCase):
    """Unit test suite for do_fft() and do_ifft()."""

    def test_fft_samesize(self):
        """Verify do_fft() output size matches input."""
        for e in [np.ones((48, 48)),
                  np.ones((48, 47)),
                  np.ones((47, 48)),
                  np.ones((47, 47))]:
            e1 = do_fft(e)
            self.assertTrue(e1.shape == e.shape)
            pass
        pass

    def test_ifft_samesize(self):
        """Verify do_ifft() output size matches input."""
        for e in [np.ones((48, 48)),
                  np.ones((48, 47)),
                  np.ones((47, 48)),
                  np.ones((47, 47))]:
            e1 = do_ifft(e)
            self.assertTrue(e1.shape == e.shape)
            pass
        pass

    def test_outandback_even(self):
        """
        Verify that doing an FFT and an IFFT on the same data produces
        the original field for even-sized arrays.
        """
        tol = 1e-13
        nx = 48
        x = np.arange(nx) - nx//2
        xx, yy = np.meshgrid(x, x)

        # fft first
        for e in [np.ones((nx, nx)),
                  np.exp(1j*2*np.pi*np.sin(4*xx + 6*yy)/nx)]:
            e1 = do_ifft(do_fft(e))
            self.assertTrue(np.max(np.abs(e1 - e)) <= tol)
            pass

        # ifft first
        for e in [np.ones((nx, nx)),
                  np.exp(1j*2*np.pi*np.sin(4*xx + 6*yy)/nx)]:
            e1 = do_fft(do_ifft(e))
            self.assertTrue(np.max(np.abs(e1 - e)) <= tol)
            pass

        pass

    def test_outandback_odd(self):
        """
        Verify that doing an FFT and an IFFT on the same data produces
        the original field for odd-sized arrays.
        """
        tol = 1e-13
        nx = 49
        x = np.arange(nx) - nx//2
        xx, yy = np.meshgrid(x, x)

        # fft first
        for e in [np.ones((nx, nx)),
                  np.exp(1j*2*np.pi*np.sin(4*xx + 6*yy)/nx)]:
            e1 = do_ifft(do_fft(e))
            self.assertTrue(np.max(np.abs(e1 - e)) <= tol)
            pass

        # ifft first
        for e in [np.ones((nx, nx)),
                  np.exp(1j*2*np.pi*np.sin(4*xx + 6*yy)/nx)]:
            e1 = do_fft(do_ifft(e))
            self.assertTrue(np.max(np.abs(e1 - e)) <= tol)
            pass

        pass

    def test_FFT(self):
        """Compare numerical FFT to analytic FFT for a special known case."""
        # tilt to delta function, uniform ones == zero tilt
        nx = 48
        x = np.arange(nx) - nx/2
        e = np.ones((nx, nx))

        e1_expect = np.zeros_like(e)
        e1_expect[np.where(x == 0), np.where(x == 0)] = nx**2

        e1 = do_fft(e)
        self.assertTrue((e1 == e1_expect).all())

        pass

    def test_fftinputs(self):
        """Check do_fft() input type valid."""
        for e in [np.ones((6,)), np.ones((6, 6, 6)), [], 'fft']:
            with self.assertRaises(TypeError):
                do_fft(e)
                pass
            pass
        pass

    def test_ifftinputs(self):
        """Check do_ifft() input type valid."""
        for e in [np.ones((6,)), np.ones((6, 6, 6)), [], 'fft']:
            with self.assertRaises(TypeError):
                do_ifft(e)
                pass
            pass
        pass


if __name__ == '__main__':
    unittest.main()
