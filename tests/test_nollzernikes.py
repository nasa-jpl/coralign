"""
Unit tests for nollzernikes.py
"""

import unittest
from math import isclose
import numpy as np

from coralign.util.pad_crop import pad_crop
from coralign.util.nollzernikes import xyzern, gen_zernikes, fit_zernikes


class TestXYZern(unittest.TestCase):
    """
    Unit test suite for xyzern()
    """

    def test_zernikes_as_expected(self):
        """
        Verify that known Zernike orders match shapes from this function
        """
        x = np.linspace(-5., 5., 100)
        y = np.linspace(-4., 6., 120)
        xm, ym = np.meshgrid(x, y)
        prad = 4.
        orders = [1, 2, 4, 8, 12, 13, 22]

        outarray = xyzern(xm, ym, prad, orders)

        # comparison data using Table 1 in Noll 1976
        rm = np.hypot(xm, ym)/float(prad)
        tm = np.arctan2(ym, xm)
        comparray = np.zeros((len(orders), xm.shape[0], xm.shape[1]))
        comparray[0] = np.ones_like(rm)
        comparray[1] = 2.*rm*np.cos(tm)
        comparray[2] = np.sqrt(3)*(2.*rm**2 - 1)
        comparray[3] = np.sqrt(8)*(3.*rm**2 - 2.)*rm*np.cos(tm)
        comparray[4] = np.sqrt(10)*(4*rm**2 - 3)*rm**2*np.cos(2*tm)
        comparray[5] = np.sqrt(10)*(4*rm**2 - 3)*rm**2*np.sin(2*tm)
        comparray[6] = np.sqrt(7)*(((20*rm**2 - 30)*rm**2 + 12)*rm**2 - 1)

        tol = 1e-12
        for j in range(len(orders)):
            self.assertTrue(np.max(np.abs(comparray[j] - outarray[j])) < tol)
            pass
        pass

    def test_unit_RMS(self):
        """Output shape should have rms of 1, up to discretization limits"""
        x = np.linspace(-5., 5., 100)
        y = np.linspace(-4., 6., 120)
        xm, ym = np.meshgrid(x, y)
        prad = 4.
        orders = [1, 2, 4, 8, 12, 13, 22]

        outarray = xyzern(xm, ym, prad, orders)

        mask = np.hypot(xm, ym) <= prad

        # could maybe tighten this, but really just want to check bulk
        # normalization is correct in an absolute sense. Getting this wrong
        # will make these numbers be 3.0, 7.0, 10.0, etc. instead of 1.
        tol = 0.01
        for j in range(len(orders)):
            rmsj = np.sqrt(np.mean(outarray[j, mask]**2))
            self.assertTrue(np.abs(rmsj - 1.0) < tol)
            pass
        pass

    def test_Nx1_2Darray(self):
        """
        Verify that Nx1 arrays go through without errors
        """
        x = np.linspace(-5., 5., 100)
        y = np.linspace(-4., 6., 120)
        xm, ym = np.meshgrid(x, y)
        prad = 4.
        orders = [1, 2, 4, 8, 12, 13, 22]

        xyzern(np.reshape(xm, (np.size(xm), 1)),
               np.reshape(ym, (np.size(ym), 1)), prad, orders)
        pass

    # Failure tests
    def test_x_2Darray(self):
        """Check correct failure on bad input array"""
        x = np.linspace(-5., 5., 100)
        y = np.linspace(-4., 6., 120)
        xm, ym = np.meshgrid(x, y)
        prad = 4.
        orders = [1, 2, 4, 8, 12, 13, 22]

        for badx in [xm[:, :-2], xm[:-2, :], np.ones((100,)),
                     np.ones((8, 8, 8)), 'text', 100, None]:
            with self.assertRaises(TypeError):
                xyzern(x=badx, y=ym, prad=prad, orders=orders)
                pass
            pass
        pass

    def test_y_2Darray(self):
        """Check correct failure on bad input array"""
        x = np.linspace(-5., 5., 100)
        y = np.linspace(-4., 6., 120)
        xm, ym = np.meshgrid(x, y)
        prad = 4.
        orders = [1, 2, 4, 8, 12, 13, 22]

        for bady in [ym[:, :-2], ym[:-2, :], np.ones((100,)),
                     np.ones((8, 8, 8)), 'text', 100, None]:
            with self.assertRaises(TypeError):
                xyzern(x=xm, y=bady, prad=prad, orders=orders)
                pass
            pass
        pass

    def test_prad_realpositivescalar(self):
        """Check correct failure on prad"""
        x = np.linspace(-5., 5., 100)
        y = np.linspace(-4., 6., 120)
        xm, ym = np.meshgrid(x, y)
        orders = [1, 2, 4, 8, 12, 13, 22]

        for badprad in [(4., 4.), [], (4.,), 'text',
                        4.*1j, -4., 0, None]:
            with self.assertRaises(TypeError):
                xyzern(x=xm, y=ym, prad=badprad, orders=orders)
                pass
            pass
        pass

    def test_orders_iterable(self):
        """Check correct failure if orders not iterable"""
        x = np.linspace(-5., 5., 100)
        y = np.linspace(-4., 6., 120)
        xm, ym = np.meshgrid(x, y)
        prad = 4.

        for badorders in [1, None, 'text']:
            with self.assertRaises(TypeError):
                xyzern(x=xm, y=ym, prad=prad, orders=badorders)
                pass
            pass
        pass

    def test_orders_elements_are_postive_scalar_integers(self):
        """Check correct failure if orders not iterable"""
        x = np.linspace(-5., 5., 100)
        y = np.linspace(-4., 6., 120)
        xm, ym = np.meshgrid(x, y)
        prad = 4.

        for badorders in [[5, 7, 8, -3], [1, 2, 4, 4.5], [3, 1j],
                          [1, 5, 'text'], [None, None, None], [0, 1, 2]]:
            with self.assertRaises(TypeError):
                xyzern(x=xm, y=ym, prad=prad, orders=badorders)
                pass
            pass
        pass


class TestPropcustom(unittest.TestCase):
    """Test all the methods of conjugate."""

    def test_gen_zernikes(self):
        """Test gen_zernikes."""
        nBeam = 100
        nArray = 110
        xOffset = 0
        yOffset = 0
        zernIndVec = np.array([1, 3, 5])
        zernCoefVec = np.array([0.5, -2.0, 3.1])

        x = (np.arange(nArray, dtype=np.float64) - nArray//2 - xOffset)
        y = (np.arange(nArray, dtype=np.float64) - nArray//2 - yOffset)
        xx, yy = np.meshgrid(x, y)

        # Create expected map
        zernCube = xyzern(xx, yy, nBeam/2., zernIndVec)
        zernAbMap0 = np.zeros([nArray, nArray], dtype=np.float64)
        for ii in range(zernIndVec.size):
            zernAbMap0 += zernCoefVec[ii]*np.squeeze(zernCube[ii, :, :])

        zernAbMap1 = gen_zernikes(zernIndVec, zernCoefVec, xOffset,
                                  yOffset, nBeam, nArray=nArray)

        diffSum = np.sum(np.abs(zernAbMap0-zernAbMap1))
        self.assertTrue(isclose(diffSum, 0, abs_tol=1e-7))
        pass

    def test_offset_zernikes(self):
        """Test that Zernikes translate as expected by comparing to np.roll."""
        nBeam = 100
        nArray = 200
        xOffset = -10
        yOffset = 30

        # Compute shifted Zernike and unshifted one for reference
        wfeZ4 = gen_zernikes(np.array([4]), np.array([1]), 0, 0, nBeam,
                             nArray=nArray)
        wfeZ4shift = gen_zernikes(np.array([4]), np.array([1]), xOffset,
                                  yOffset, nBeam, nArray=nArray)

        # Roll the shifted one, crop both, and see if the difference is zero
        wfeZ4shiftRoll = np.roll(wfeZ4shift, (-yOffset, -xOffset), axis=(0, 1))
        nMax = int(nArray - 2*np.max(np.abs((yOffset, xOffset))))
        wfeZ4crop = pad_crop(wfeZ4, (nMax, nMax))
        wfeZ4shiftRollCrop = pad_crop(wfeZ4shiftRoll, (nMax, nMax))
        diffSum = np.sum(np.abs(wfeZ4crop-wfeZ4shiftRollCrop))

        self.assertTrue(isclose(diffSum, 0, abs_tol=1e-7))

    def test_fit_zernikes(self):
        """Test that known Zernike coefficients are returned from the fit."""
        # Generate a shifted aberration map of only Zernike modes
        nBeam = 100
        nArray = 120

        xOffset = -15.55
        yOffset = 30.1

        maxNollZern = 11
        zernCoefIn = 20*np.random.rand(maxNollZern)

        # Generate a ROI mask
        x = (np.arange(-nArray/2., nArray/2.) - xOffset)/nBeam
        y = (np.arange(-nArray/2., nArray/2.) - yOffset)/nBeam
        [X, Y] = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        mask = np.zeros((nArray, nArray))
        mask[R <= 0.5] = 1

        wfeShift = gen_zernikes(np.arange(1, maxNollZern+1), zernCoefIn,
                                xOffset, yOffset, nBeam, nArray=nArray)
        zernCoefOut = fit_zernikes(wfeShift, mask, maxNollZern,
                                   nBeam, xOffset, yOffset)
        diffSum = np.sum(np.abs(zernCoefIn - zernCoefOut))

        self.assertTrue(isclose(diffSum, 0, abs_tol=1e-10))


class TestPropcustomInputFailure(unittest.TestCase):
    """Test suite for valid function inputs."""

    def test_gen_zernikes_inputs(self):
        """Test inputs for gen_zernikes."""
        nBeam = 100
        nArray = 110
        xOffset = 0
        yOffset = 0
        zernIndVec = np.array([1, 3, 5])
        zernCoefVec = np.array([0.5, -2.0, 3.1])

        x = (np.arange(nArray, dtype=np.float64) - nArray//2 - xOffset)
        y = (np.arange(nArray, dtype=np.float64) - nArray//2 - yOffset)
        xx, yy = np.meshgrid(x, y)

        # Make sure it runs normally first.
        gen_zernikes(zernIndVec, zernCoefVec, xOffset,
                     yOffset, nBeam, nArray=nArray)

        for zernIndVecBad in (1, np.ones((5, 10)), 'asdf'):
            with self.assertRaises(ValueError):
                gen_zernikes(zernIndVecBad, zernCoefVec, xOffset,
                             yOffset, nBeam, nArray=nArray)

        for zernCoefVecBad in (1, np.ones((5, 10)), 'asdf'):
            with self.assertRaises(ValueError):
                gen_zernikes(zernIndVec, zernCoefVecBad, xOffset,
                             yOffset, nBeam, nArray=nArray)

        zernIndVecBad = np.array([1, 3, 5])
        zernCoefVecBad = np.array([0.5, -2.0, 3.1, 1.1])
        with self.assertRaises(ValueError):
            gen_zernikes(zernIndVecBad, zernCoefVecBad, xOffset,
                         yOffset, nBeam, nArray=nArray)

        for xOffsetBad in (1j, np.ones((5, 10)), 'asdf'):
            with self.assertRaises(ValueError):
                gen_zernikes(zernIndVec, zernCoefVec, xOffsetBad,
                             yOffset, nBeam, nArray=nArray)

        for yOffsetBad in (1j, np.ones((5, 10)), 'asdf'):
            with self.assertRaises(ValueError):
                gen_zernikes(zernIndVec, zernCoefVec, xOffset,
                             yOffsetBad, nBeam, nArray=nArray)

        for nBeamBad in (-100, 1j, np.ones((5, 10)), 'asdf'):
            with self.assertRaises(ValueError):
                gen_zernikes(zernIndVec, zernCoefVec, xOffset,
                             yOffset, nBeamBad, nArray=nArray)

        for nArrayBad in (-100, 0, 1j, np.ones((5, 10)), 'asdf'):
            with self.assertRaises(ValueError):
                gen_zernikes(zernIndVec, zernCoefVec, xOffset,
                             yOffset, nBeam, nArray=nArrayBad)
        pass

    def test_fit_zernikes_inputs(self):
        """Test inputs for fit_zernikes."""
        # Generate a shifted aberration map of only Zernike modes
        nBeam = 100
        nArray = 120

        xOffset = -15.55
        yOffset = 30.1

        maxNollZern = 11
        zernCoefIn = 20*np.random.rand(maxNollZern)

        # Generate a ROI mask
        x = (np.arange(-nArray/2., nArray/2.) - xOffset)/nBeam
        y = (np.arange(-nArray/2., nArray/2.) - yOffset)/nBeam
        [X, Y] = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        mask = np.zeros((nArray, nArray))
        mask[R <= 0.5] = 1

        wfe = gen_zernikes(np.arange(1, maxNollZern+1), zernCoefIn,
                           xOffset, yOffset, nBeam, nArray=nArray)

        # Make sure it runs normally first.
        fit_zernikes(wfe, mask, maxNollZern, nBeam, xOffset, yOffset)

        for wfeBad in (5, 1j, np.ones((5,)), 2j*np.ones((5, 10)), 'asdf'):
            with self.assertRaises(ValueError):
                fit_zernikes(wfeBad, mask, maxNollZern, nBeam,
                             xOffset, yOffset)

        for maskBad in (5, 1j, np.ones((5,)), 2j*np.ones((5, 10)), 'asdf'):
            with self.assertRaises(ValueError):
                fit_zernikes(wfe, maskBad, maxNollZern, nBeam,
                             xOffset, yOffset)

        with self.assertRaises(ValueError):
            fit_zernikes(wfe, mask[2::, 2::], maxNollZern, nBeam,
                         xOffset, yOffset)

        for maxNollZernBad in (1j, np.ones((5,)), 2j*np.ones((5, 10)), 'asdf'):
            with self.assertRaises(ValueError):
                fit_zernikes(wfe, mask, maxNollZernBad, nBeam,
                             xOffset, yOffset)

        for nBeamBad in (-10, 1j, np.ones((5,)), np.ones((5, 10)), 'asdf'):
            with self.assertRaises(TypeError):
                fit_zernikes(wfe, mask, maxNollZern, nBeamBad,
                             xOffset, yOffset)

        for xOffsetBad in (1j, np.ones((5,)), np.ones((5, 10)), 'asdf'):
            with self.assertRaises(TypeError):
                fit_zernikes(wfe, mask, maxNollZern, nBeam,
                             xOffsetBad, yOffset)

        for yOffsetBad in (1j, np.ones((5,)), np.ones((5, 10)), 'asdf'):
            with self.assertRaises(TypeError):
                fit_zernikes(wfe, mask, maxNollZern, nBeam,
                             xOffset, yOffsetBad)
        pass


if __name__ == '__main__':
    unittest.main()
