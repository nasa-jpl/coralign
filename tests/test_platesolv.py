"""Unit test suite for platesolv.py."""

import unittest
import os
import numpy as np

from coralign.platesolv.platesolv import platesolv, PlateSolvException
from coralign.util import shapes
from coralign.util.mft import do_imft, do_mft

LOCAL_PATH = os.path.dirname(os.path.realpath(__file__))


class TestPlateSolv(unittest.TestCase):
    """Unit test suite for platesolv()."""

    def setUp(self):
        self.fnParams = os.path.join(LOCAL_PATH, 'testdata',
                                     'platesolv_parms.yaml')

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
        ppl = 1.25
        Efocus = do_mft(pupil, imageShape, ppl, Nbeam)
        image = (np.abs(Efocus))**2
        self.image = np.round(1e9 * image / np.max(image))

    # test lambda/D/pixel estimate returned by platesolv
    def test_platescale_calc(self):
        """Check that platescale is recovered for sample image."""
        # discrepancy tolerance
        tol = 0.01   # 1% tolerance on lambda/d/pixel

        # reference pixel period
        # 'the answer'
        pixperiod_answer = 1.25  # reference lambda/d/pixel

        # 'the guess'
        pixperiod_guess = 1.32

        # pixperpupil value does not need to be accurate, just the same for all
        # mft and ifmt calls
        pixperpupil = np.max(self.image.shape)

        # make fake image with the 'wrong' plate scale,
        # based on pupil image of correct scaling
        pupil_image_simulated = do_imft(self.image,
                                        self.image.shape,
                                        pixperiod_answer,
                                        pixperpupil)
        answer_image = np.real(do_mft(pupil_image_simulated,
                                      self.image.shape,
                                      pixperiod_answer,
                                      pixperpupil))
        guess_image = np.real(do_mft(pupil_image_simulated,
                                     self.image.shape,
                                     pixperiod_guess,
                                     pixperpupil))

        # assuming the simulated image was generated at the wrong plate scale
        # (pixperiod_guess), solve for the correct one
        best_fit, pix_per_arr, resid_norm = platesolv(
            answer_image, guess_image, pixperiod_guess,
            yaml_param_file=self.fnParams)

        # check that the computed place scale is the same as 'the answer'
        # print(np.abs(best_fit - pixperiod_answer) / pixperiod_answer)
        self.assertTrue(np.abs(best_fit - pixperiod_answer) /
                        pixperiod_answer <= tol)
        pass

    def test_snr_scaling(self):
        """Check that platescale is recovered in the presence of noise."""
        # discrepancy tolerance
        tol = 0.05   # 5% tolerance at low SNR

        # reference pixel period
        # 'the answer'
        pixperiod_answer = 1.25  # reference lambda/d/pixel

        # 'the guess'
        pixperiod_guess = 1.32

        # pixperpupil value does not need to be accurate, just the same for all
        # mft and ifmt calls
        pixperpupil = np.max(self.image.shape)

        # snr value of 'recorded' image
        snr = 50.

        # make fake image with the 'wrong' plate scale,
        # based on pupil image of correct scaling
        pupil_image_simulated = do_imft(self.image,
                                        self.image.shape,
                                        pixperiod_answer,
                                        pixperpupil)
        answer_image = np.real(do_mft(pupil_image_simulated,
                                      self.image.shape,
                                      pixperiod_answer,
                                      pixperpupil))
        guess_image = np.real(do_mft(pupil_image_simulated,
                                     self.image.shape,
                                     pixperiod_guess,
                                     pixperpupil))

        rng = np.random.default_rng(7654321)

        # add noise to comparison image
        noise_arr = rng.standard_normal(answer_image.shape) * snr
        im_noise = (answer_image / np.amax(answer_image)) * (snr ** 2.) + \
            noise_arr

        # assuming the simulated image was generated at the wrong plate scale
        # (pixperiod_guess), solve for the correct one
        best_fit, pix_per_arr, resid_norm = platesolv(
            im_noise, guess_image, pixperiod_guess,
            yaml_param_file=self.fnParams)

        # check that the computed shift of image is the same as 'the answer'
        # print(np.abs(best_fit - pixperiod_answer) / pixperiod_answer)
        self.assertTrue(np.abs(best_fit - pixperiod_answer) /
                        pixperiod_answer <= tol)
        pass

    def test_pixperlod_platesolv(self):
        """Check pixel period is valid type for platesolv()."""
        for pixperlod in [-1.5, -1, 0, 1j, [], 'perr', (5,)]:
            with self.assertRaises(PlateSolvException):
                platesolv(self.image, self.image, pixperlod)
                pass
            pass
        pass

    def test_img_2Darray_platesolv(self):
        """Check input image type valid in platesolv()."""
        pixperlod = 1.2

        for e in [np.ones((6,)), np.ones((6, 6, 6)), [], 'fft']:
            with self.assertRaises(PlateSolvException):
                platesolv(e, e, pixperlod)
                pass
            pass
        pass

    def test_pixperlod_platesolv_input_sizes(self):
        """Check input images sizes are valid."""
        pixperlod = 1.2
        img1 = np.ones((512, 512))
        img2 = np.ones((256, 256))
        with self.assertRaises(PlateSolvException):
            platesolv(img1, img2, pixperlod)
            pass
        pass


if __name__ == '__main__':
    unittest.main()
