"""Unit test suite for psffit.py."""

import unittest
import os
import numpy as np
from scipy import ndimage

from coralign.util import mft, shapes
from coralign.psffit.psffit import psffit

# ignore divide by zero errors
np.seterr(divide='ignore', invalid='ignore')

LOCAL_PATH = os.path.dirname(os.path.realpath(__file__))


class TestPsfFit(unittest.TestCase):
    """Unit tests for psffit."""

    def setUp(self):
        self.config_file = os.path.join(LOCAL_PATH, 'testdata',
                                        'psffit_parms.yaml')

        Nbeam = 300
        nx = 310
        ny = nx
        xOffset = 0.
        yOffset = 0.
        diamInner = 0.20*Nbeam
        diamOuter = 1.00*Nbeam
        strutAngles = np.array([0, 90, 180, 270]) + 15
        strutWidth = 0.03*Nbeam
        pupil = shapes.simple_pupil(
            nx, ny, xOffset, yOffset, diamInner, diamOuter,
            strutAngles=strutAngles, strutWidth=strutWidth)

        imageShape = (512, 512)
        ppl = 3
        Efocus = mft.do_mft(pupil, imageShape, ppl, Nbeam)
        image = (np.abs(Efocus))**2
        self.image = np.round(1e9 * image / np.max(image))

    def test_offset_calc(self):
        """Test shift returned by psffit."""
        # discrepancy tolerance
        tol = 1/6   # 1/6 pixel accuracy relative to 'truth'

        # load in reference image
        # img = (np.array(fits.getdata(self.image, 0)))[0, :, :]
        img = self.image
        img /= np.amax(img)

        # shift reference image by known amount ()'the answer')
        the_answer = np.array([80.1335, -110.1312])
        img_shifted = ndimage.shift(img, the_answer)
        img_shifted /= np.amax(img_shifted)

        # use psffit to comput offset between two images
        shift_computed, _ = psffit(img, img_shifted, self.config_file)

        # check that the computed shift of image is the same as 'the answer'
        self.assertTrue(
            np.abs(np.amax(shift_computed - the_answer)) <= tol)

    def test_amplitude_calc(self):
        """Test amplitude estimate from PSF fit."""
        # discrepancy tolerance
        tol = 0.1   # 10% accuracy on amplitude recovery

        # load in reference image
        # img = (np.array(fits.getdata(self.image, 0)))[0, :, :]
        img = self.image

        # normalize
        img /= np.amax(img)

        # shift reference image by known amount
        shift_amt = [80.1335, -110.1312]

        # scale shifted image by known amplitude
        the_answer = 0.252
        img_shifted = ndimage.shift(img * the_answer, shift_amt)

        # use psffit to comput offset between two images
        _, amplitude = psffit(img, img_shifted, self.config_file)

        # check that the computed amplitude is the same as 'the answer'
        self.assertTrue(
            np.abs(np.amax(amplitude - the_answer)) / the_answer <= tol)
        pass

    def test_offset_calc_noise(self):
        """Test shift returned by psffit, including noise."""
        # discrepancy tolerance
        tol = 1/6  # 1/6 pixel accuracy relative to 'truth'
        nfuzz = 50

        # load in reference image
        # img = (np.array(fits.getdata(self.image, 0)))[0, :, :]
        img = self.image
        img /= np.amax(img)

        # scale image to specified SNR
        snr = 20.
        img *= (snr ** 2.)

        # shift reference image by known amount ()'the answer')
        the_answer = np.array([80.5335, -110.31312])
        img_shifted = ndimage.shift(img, the_answer)

        # add noise
        rng = np.random.default_rng(55555)
        for _ in range(nfuzz):
            noise_arr = rng.standard_normal(img.shape) * snr
            img_noise = img_shifted + noise_arr

            # use psffit to comput offset between two images
            shift_computed, _ = psffit(img, img_noise, self.config_file)

            # check that the computed shift of image is same as 'the answer'
            self.assertTrue(
                np.abs(np.amax(shift_computed - the_answer)) <= tol)
            pass
        pass

    def test_amplitude_calc_noise(self):
        """Test amplitude returned by psffit, including noise."""
        # discrepancy tolerance
        tol = 0.1   # 10% accuracy on amplitude recovery
        nfuzz = 50

        # load in reference image
        # img = (np.array(fits.getdata(self.image, 0)))[0, :, :]
        img = self.image
        img /= np.amax(img)

        # scale image to specified SNR
        snr = 40.
        the_answer = snr ** 2.

        # shift reference image by known amount
        shift_amt = [80.1335, -110.1312]

        # scale image by known offset, and scale up to 'the answer'
        img_shifted = ndimage.shift(img, shift_amt)
        img_shifted *= the_answer

        # add noise
        rng = np.random.default_rng(44444)
        for _ in range(nfuzz):
            noise_arr = rng.standard_normal(img.shape) * snr
            img_noise = img_shifted + noise_arr

            # use psffit to comput offset between two images
            _, amplitude = psffit(img, img_noise, self.config_file)

            # check that the computed amplitude is the same as 'the answer'
            self.assertTrue(
                np.abs(np.amax(amplitude - the_answer)) / the_answer <= tol)
            pass
        pass


# check if this is run in main or not
if __name__ == '__main__':
    unittest.main()
