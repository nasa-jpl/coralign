"""Unit tests for mft.py."""
import unittest
import numpy as np

from coralign.util.mft import do_mft, do_offcenter_mft, do_imft, _interval
from coralign.util.pad_crop import pad_crop
from coralign.util.fresnelprop import do_fft


class TestDoMFT_IMFT(unittest.TestCase):
    """Unit test suite for do_mft() and do_imft()."""

    def setUp(self):
        """Define reused variables."""
        self.e = np.ones((128, 128))
        self.outshape = (512, 512)
        self.pixperlod = 4
        self.pixperpupil = 128
        pass

    def test_success_mft(self):
        """do_mft completes without incident on valid inputs."""
        do_mft(self.e, self.outshape, self.pixperlod, self.pixperpupil)
        pass

    def test_success_imft(self):
        """do_imft completes without incident on valid inputs."""
        do_imft(self.e, self.outshape, self.pixperlod, self.pixperpupil)
        pass

    def test_success_mft_nonsquare(self):
        """do_mft completes without incident on valid nonsquare inputs."""
        do_mft(np.ones((128, 127)), (511, 512),
               self.pixperlod, self.pixperpupil)
        pass

    def test_success_imft_nonsquare(self):
        """do_imft completes without incident on valid nonsquare inputs."""
        do_imft(np.ones((128, 127)), (511, 512),
                self.pixperlod, self.pixperpupil)
        pass

    def test_mft_correct_size(self):
        """Verify output shape matches commanded output shape in do_mft()."""
        for outshape in [(48, 48),
                         (48, 47),
                         (47, 48),
                         (47, 47)]:
            eout = do_mft(self.e, outshape, self.pixperlod, self.pixperpupil)
            self.assertTrue(eout.shape == outshape)
            pass
        pass

    def test_imft_correct_size(self):
        """Verify output shape matches commanded output shape in do_imft()."""
        for outshape in [(48, 48),
                         (48, 47),
                         (47, 48),
                         (47, 47)]:
            eout = do_imft(self.e, outshape, self.pixperlod, self.pixperpupil)
            self.assertTrue(eout.shape == outshape)
            pass
        pass

    def test_e_2Darray_mft(self):
        """Check e-field type valid in do_mft()."""
        for e in [np.ones((6,)), np.ones((6, 6, 6)), [], 'fft']:
            with self.assertRaises(TypeError):
                do_mft(e, self.outshape, self.pixperlod, self.pixperpupil)
                pass
            pass
        pass

    def test_e_2Darray_imft(self):
        """Check e-field type valid in do_imft()."""
        for e in [np.ones((6,)), np.ones((6, 6, 6)), [], 'fft']:
            with self.assertRaises(TypeError):
                do_mft(e, self.outshape, self.pixperlod, self.pixperpupil)
                pass
            pass
        pass

    def test_outshape_mft(self):
        """Check output size format valid in do_mft()."""
        for outshape in [(6,), (6, 6, 6), [], None, 6, 'fft']:
            with self.assertRaises(TypeError):
                do_mft(self.e, outshape, self.pixperlod, self.pixperpupil)
                pass
            pass
        pass

    def test_outshape_imft(self):
        """Check output size format valid in do_imft()."""
        for outshape in [(6,), (6, 6, 6), [], None, 6, 'fft']:
            with self.assertRaises(TypeError):
                do_imft(self.e, outshape, self.pixperlod, self.pixperpupil)
                pass
            pass
        pass

    def test_pixperlod_mft(self):
        """Check scaling type valid in do_mft()."""
        for pixperlod in [-1.5, -1, 0, 1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                do_mft(self.e, self.outshape, pixperlod, self.pixperpupil)
                pass
            pass
        pass

    def test_pixperlod_imft(self):
        """Check scaling type valid in do_imft()."""
        for pixperlod in [-1.5, -1, 0, 1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                do_imft(self.e, self.outshape, pixperlod, self.pixperpupil)
                pass
            pass
        pass

    def test_pixperpupil_mft(self):
        """Check scaling type valid in do_mft()."""
        for pixperpupil in [-1.5, -1, 0, 1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                do_mft(self.e, self.outshape, self.pixperlod, pixperpupil)
                pass
            pass
        pass

    def test_pixperpupil_imft(self):
        """Check scaling type valid in do_imft()."""
        for pixperpupil in [-1.5, -1, 0, 1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                do_imft(self.e, self.outshape, self.pixperlod, pixperpupil)
                pass
            pass
        pass

    def test_mft(self):
        """Test MFT for a known analytic Fourier transform case."""
        # Gaussians go to Gaussians, see table 2.1 in Goodman for convention
        tol = 1e-10

        d = 128
        outshape = (d, d)
        pixperlod = 4
        pixperpupil = d

        x = _interval(d, d)[0]
        XX, YY = np.meshgrid(x, x)
        RR = np.hypot(XX, YY)
        nRR = RR/float(d)  # norm pupil diam of 1
        pRR = RR/float(pixperlod)  # can reuse for focal plane

        gauss_exp = 5
        e = np.exp(-np.pi*(nRR*gauss_exp)**2)

        e_mft = do_mft(e, outshape, pixperlod, pixperpupil)
        e_analytic = np.exp(-np.pi*(pRR/gauss_exp)**2)/(gauss_exp)**2

        self.assertTrue(np.max(np.abs(e_mft - e_analytic)) < tol)
        pass

    def test_mft_nonsquare(self):
        """Test MFT for a known analytic Fourier transform case."""
        # Gaussians go to Gaussians, see table 2.1 in Goodman for convention
        tol = 1e-10

        d = 128
        outshape = (d, d*2)
        pixperlod = 4
        pixperpupil = d

        x = _interval(d, d)[0]
        XX, YY = np.meshgrid(x, x)
        RR = np.hypot(XX, YY)
        nRR = RR/float(d)  # norm pupil diam of 1

        col = _interval(d*2, d*2)[0]
        X2, Y2 = np.meshgrid(col, x)
        R2 = np.hypot(X2, Y2)
        pRR = R2/float(pixperlod)  # can reuse for focal plane

        gauss_exp = 5
        e = np.exp(-np.pi*(nRR*gauss_exp)**2)

        e_mft = do_mft(e, outshape, pixperlod, pixperpupil)
        e_analytic = np.exp(-np.pi*(pRR/gauss_exp)**2)/(gauss_exp)**2

        self.assertTrue(np.max(np.abs(e_mft - e_analytic)) < tol)
        pass

    def test_imft(self):
        """Test IMFT for a known analytic Fourier transform case."""
        # Gaussians go to Gaussians, see table 2.1 in Goodman for convention
        tol = 1e-10

        d = 128
        outshape = (d, d)
        pixperlod = 4
        pixperpupil = d

        x = _interval(d, d)[0]
        XX, YY = np.meshgrid(x, x)
        RR = np.hypot(XX, YY)
        nRR = RR/float(d)  # norm pupil diam of 1
        pRR = RR/float(pixperlod)  # can reuse for focal plane

        gauss_exp = 5
        e = np.exp(-np.pi*(pRR/gauss_exp)**2)/(gauss_exp)**2

        e_imft = do_imft(e, outshape, pixperlod, pixperpupil)
        e_analytic = np.exp(-np.pi*(nRR*gauss_exp)**2)

        self.assertTrue(np.max(np.abs(e_imft - e_analytic)) < tol)
        pass

    def test_imft_nonsquare(self):
        """Test IMFT for a known analytic Fourier transform case."""
        # Gaussians go to Gaussians, see table 2.1 in Goodman for convention
        tol = 1e-10

        d = 128
        outshape = (d, d*2)
        pixperlod = 4
        pixperpupil = d

        x = _interval(d, d)[0]
        XX, YY = np.meshgrid(x, x)
        RR = np.hypot(XX, YY)
        pRR = RR/float(pixperlod)  # can reuse for focal plane

        col = _interval(d*2, d*2)[0]
        X2, Y2 = np.meshgrid(col, x)
        R2 = np.hypot(X2, Y2)
        nRR = R2/float(d)  # norm pupil diam of 1

        gauss_exp = 5
        e = np.exp(-np.pi*(pRR/gauss_exp)**2)/(gauss_exp)**2

        e_imft = do_imft(e, outshape, pixperlod, pixperpupil)
        e_analytic = np.exp(-np.pi*(nRR*gauss_exp)**2)

        self.assertTrue(np.max(np.abs(e_imft - e_analytic)) < tol)
        pass

    def test_mft_vs_fft(self):
        """Test that MFT and FFT both give same results for a known case."""
        # square region goes to sinc in each direction
        tol = 1e-13

        d = 32
        outshape = (d, d)
        pixperlod = 4
        nptse = d*pixperlod
        pixperpupil = nptse

        e = np.zeros((nptse, nptse))
        e[np.ceil(nptse/4).astype('int'):-np.ceil(nptse/4).astype('int'),
          np.ceil(nptse/4).astype('int'):-np.ceil(nptse/4).astype('int')] = 1

        e_mft = do_mft(e, outshape, pixperlod, pixperpupil)
        e_fft = pad_crop(do_fft(pad_crop(e,
                        (nptse*pixperlod, nptse*pixperlod))), (d, d))/nptse**2

        self.assertTrue(np.max(np.abs(e_mft - e_fft)) < tol)
        pass

    def test_zero_pad_mft_input(self):
        """
        Test padded input.

        Verify that an e with dimension larger than pixperpupil doesn't change
        the output, as long as the underlying data has a pupil of size
        pixperpupil
        """
        tol = 1e-13

        # use integer ratios so there's no confusion from hitting slightly
        # different sampling points

        scale = 2
        e2 = pad_crop(self.e, (scale*self.pixperpupil,
                                 scale*self.pixperpupil))

        out = do_mft(self.e, self.outshape, self.pixperlod, self.pixperpupil)
        out2 = do_mft(e2, self.outshape, self.pixperlod, self.pixperpupil)
        self.assertTrue(np.max(np.abs(out - out2)) < tol)
        pass

    def test_zero_pad_imft_output(self):
        """
        Test oversized output.

        Verify that an outshape value larger than pixperpupil doesn't change
        the center of the output.
        """
        tol = 1e-12

        # use integer ratios so there's no confusion from hitting slightly
        # different sampling points

        scale = 2
        outshape2 = (scale*self.outshape[0], scale*self.outshape[1])

        out = do_imft(self.e, self.outshape, self.pixperlod, self.pixperpupil)
        out2 = do_imft(self.e, outshape2, self.pixperlod, self.pixperpupil)
        out2crop = pad_crop(out2, self.outshape)
        self.assertTrue(np.max(np.abs(out - out2crop)) < tol)
        pass


class TestDoOffcenterMFT(unittest.TestCase):
    """Unit test suite for do_offcenter_mft()."""

    def setUp(self):
        """Define reused variables."""
        self.e = np.ones((128, 109))
        self.pupilShape = (255, 256)
        self.yxLowerLeft = (22, 9)
        self.efull = np.roll(
            pad_crop(self.e, self.pupilShape),
            [self.yxLowerLeft[0]-(self.pupilShape[0]//2-self.e.shape[0]//2),
             self.yxLowerLeft[1]-(self.pupilShape[1]//2-self.e.shape[1]//2)],
            axis=(0, 1))
        self.outshape = (512, 511)
        self.pixperlod = 4
        self.pixperpupil = 128
        pass

    def test_success_offcenter_mft(self):
        """do_offcenter_mft completes without incident on valid inputs."""
        do_offcenter_mft(self.e, self.outshape, self.pixperlod,
                         self.pixperpupil, self.pupilShape, self.yxLowerLeft)

    def test_success_mft_nonsquare(self):
        """do_offcenter_mft completes w/o issue on valid nonsquare inputs."""
        do_offcenter_mft(np.ones((128, 127)), (511, 512), self.pixperlod,
                         self.pixperpupil, self.pupilShape, self.yxLowerLeft)

    def test_offcenter_mft_correct_size(self):
        """Verify output shape matches commanded output shape."""
        for outshape in [(48, 48),
                         (48, 47),
                         (47, 48),
                         (47, 47)]:
            eout = do_offcenter_mft(self.e, outshape, self.pixperlod,
                                    self.pixperpupil, self.pupilShape,
                                    self.yxLowerLeft)
            self.assertTrue(eout.shape == outshape)

    def test_e_2Darray_offcenter_mft(self):
        """Check e-field type valid in do_offcenter_mft()."""
        for e in [np.ones((6,)), np.ones((6, 6, 6)), [], 'fft']:
            with self.assertRaises(TypeError):
                do_offcenter_mft(e, self.outshape, self.pixperlod,
                                 self.pixperpupil, self.pupilShape,
                                 self.yxLowerLeft)

    def test_outshape_offcenter_mft(self):
        """Check output size format valid in do_offcenter_mft()."""
        for outshape in [(6,), (6, 6, 6), [], None, 6, 'fft']:
            with self.assertRaises(TypeError):
                do_offcenter_mft(self.e, outshape, self.pixperlod,
                                 self.pixperpupil, self.pupilShape,
                                 self.yxLowerLeft)

    def test_pixperlod_offcenter_mft(self):
        """Check scaling type valid in do_offcenter_mft()."""
        for pixperlod in [-1.5, -1, 0, 1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                do_offcenter_mft(self.e, self.outshape, pixperlod,
                                 self.pixperpupil, self.pupilShape,
                                 self.yxLowerLeft)

    def test_pixperpupil_offcenter_mft(self):
        """Check scaling type valid in do_offcenter_mft()."""
        for pixperpupil in [-1.5, -1, 0, 1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                do_offcenter_mft(self.e, self.outshape, self.pixperlod,
                                 pixperpupil, self.pupilShape,
                                 self.yxLowerLeft)

    def test_pupilShape_offcenter_mft(self):
        """Check E-field size format valid in do_offcenter_mft()."""
        for pupilShape in [(6,), (6, 6, 6), [], None, 6, 'fft']:
            with self.assertRaises(TypeError):
                do_offcenter_mft(self.e, self.outshape, self.pixperlod,
                                 self.pixperpupil, pupilShape,
                                 self.yxLowerLeft)

    def test_yxLowerLeft_offcenter_mft(self):
        """Check coordinate vector format valid in do_offcenter_mft()."""
        for yxLowerLeft in [(6,), (6, 6, 6), [], None, 6, 'fft']:
            with self.assertRaises(TypeError):
                do_offcenter_mft(self.e, self.outshape, self.pixperlod,
                                 self.pixperpupil, self.pupilShape,
                                 yxLowerLeft)

    def test_offcenter_vs_centered_mft(self):
        """Check that offcenter MFT gives same result as a centered MFT."""
        e_on = do_mft(self.efull, self.outshape, self.pixperlod,
                      self.pixperpupil)
        e_off = do_offcenter_mft(self.e, self.outshape, self.pixperlod,
                                 self.pixperpupil, self.pupilShape,
                                 self.yxLowerLeft)

        abs_tol = 100*np.finfo(float).eps
        self.assertTrue(np.max(np.abs(e_on - e_off)) < abs_tol)


class TestInterval(unittest.TestCase):
    """Unit test suite for _interval()."""

    def test_interval_odd(self):
        """Verify interval is correct for odd-sized array."""
        expected = np.asarray([-2., -1., 0., 1., 2.])
        actual = _interval(5, 5)[0]
        self.assertTrue((expected == actual).all())
        pass

    def test_interval_even(self):
        """Verify interval is correct for even-sized array."""
        expected = np.asarray([-3., -2., -1., 0., 1., 2.])
        actual = _interval(6, 6)[0]
        self.assertTrue((expected == actual).all())
        pass

    def test_invalid_width(self):
        """Verify bad inputs caught."""
        npts = 10

        for width in [1j, -2, 0, (5.,), 'txt', None]:
            with self.assertRaises(TypeError):
                _interval(width=width, npts=npts)
                pass
            pass
        pass

    def test_invalid_npts(self):
        """Verify bad inputs caught."""
        width = 10

        for npts in [1j, -2, 0, (5.,), 'txt', None, 1.5]:
            with self.assertRaises(TypeError):
                _interval(width=width, npts=npts)
                pass
            pass
        pass


if __name__ == '__main__':
    unittest.main()
