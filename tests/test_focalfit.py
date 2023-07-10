"""Test suite for FOCALFIT."""
import unittest
import os
import numpy as np
from astropy.io import fits

import coralign.focalfit.focalfit as ff


class TestFocalfitInputFailure(unittest.TestCase):
    """Test suite for valid function inputs."""

    def test_calc_bowtie_offsets_from_spots_inputs(self):
        """Test the inputs of calc_bowtie_offsets_from_spots."""
        xOffsetStarFromCenterPixel = 5
        yOffsetStarFromCenterPixel = 2

        localpath = os.path.dirname(os.path.abspath(__file__))
        fmImage = os.path.join(
            localpath, 'testdata',
            'spots_spc_spec_sep8.70_res2.92_xs5_ys2_xfpm1.00_yfpm-1.00.fits')
        imageSpotted = fits.getdata(fmImage)
        fnYAML = os.path.join(localpath, 'testdata',
                              'focalfit_fixed_params_spc_spec.yaml')

        # Check standard inputs do not raise anything first
        ff.calc_bowtie_offsets_from_spots(
            imageSpotted, xOffsetStarFromCenterPixel,
            yOffsetStarFromCenterPixel, fnYAML)

        with self.assertRaises(TypeError):
            ff.calc_bowtie_offsets_from_spots(
                np.zeros((10,)), xOffsetStarFromCenterPixel,
                yOffsetStarFromCenterPixel, fnYAML)
        with self.assertRaises(TypeError):
            ff.calc_bowtie_offsets_from_spots(
                imageSpotted, 1j,
                yOffsetStarFromCenterPixel, fnYAML)
        with self.assertRaises(TypeError):
            ff.calc_bowtie_offsets_from_spots(
                imageSpotted, xOffsetStarFromCenterPixel,
                np.zeros((20,)), fnYAML)

    def test_calc_offset_from_spots_inputs(self):
        """Test the inputs of calc_offset_from_spots."""
        xProbeRotDeg = 0
        xOffsetStarFromCenterPixel = 5
        yOffsetStarFromCenterPixel = 2

        localpath = os.path.dirname(os.path.abspath(__file__))
        fmImage = os.path.join(localpath, 'testdata',
            'spots_hlc_sep2.80_res2.30_xs5_ys2_xfpm-0.25_yfpm0.50.fits')
        imageSpotted = fits.getdata(fmImage)
        fnTuning = os.path.join(localpath, 'testdata',
                              'focalfit_fixed_params_hlc_nfov.yaml')
        fnTarget = os.path.join(localpath, 'testdata',
                                'focalfit_fixed_params_ratio_1.00.yaml')

        # Check standard inputs do not raise anything first
        ff.calc_offset_from_spots(imageSpotted, xProbeRotDeg,
                           xOffsetStarFromCenterPixel,
                           yOffsetStarFromCenterPixel, fnTuning, fnTarget)

        with self.assertRaises(TypeError):
            ff.calc_offset_from_spots(np.zeros((10,)), xProbeRotDeg,
                           xOffsetStarFromCenterPixel,
                           yOffsetStarFromCenterPixel, fnTuning, fnTarget)
        with self.assertRaises(TypeError):
            ff.calc_offset_from_spots(imageSpotted, [30],
                           xOffsetStarFromCenterPixel,
                           yOffsetStarFromCenterPixel, fnTuning, fnTarget)
        with self.assertRaises(TypeError):
            ff.calc_offset_from_spots(imageSpotted, xProbeRotDeg,
                           1j,
                           yOffsetStarFromCenterPixel, fnTuning, fnTarget)
        with self.assertRaises(TypeError):
            ff.calc_offset_from_spots(imageSpotted, xProbeRotDeg,
                           xOffsetStarFromCenterPixel,
                           np.zeros((20,)), fnTuning, fnTarget)


class TestFocalfit(unittest.TestCase):
    """Unit and integration tests."""

    def test_calc_offset_from_spots_hlc_fs(self):
        """Test that estimated offset of star from FS is accurate enough."""
        xProbeRotDeg = 0
        xOffsetStarFromCenterPixel = 5
        yOffsetStarFromCenterPixel = 2

        mode = 'hlc'  # used only for testdata filename
        spotSepLamD = 9.0  # lambdaC/D
        pixPerLamD = 2.3
        # spotSepPix = spotSepLamD*pixPerLamD

        fs_x_offset_vec = [-0.25, ]  # lambda0/D
        fs_y_offset_vec = [0.5, ]  # lambda0/D
        nOffsets = len(fs_x_offset_vec)

        localpath = os.path.dirname(os.path.abspath(__file__))
        fnTuning = os.path.join(localpath, 'testdata',
                              'focalfit_fixed_params_hlc_nfov_fs.yaml')
        fnTarget = os.path.join(localpath, 'testdata',
                                'focalfit_fixed_params_ratio_1.00.yaml')

        for iOffset in range(nOffsets):

            fs_x_offset = fs_x_offset_vec[iOffset]  # lambda0/D
            fs_y_offset = fs_y_offset_vec[iOffset]  # lambda0/D

            fnImage = os.path.join(localpath, 'testdata',
                ('spots_%s_sep%.2f_res%.2f_xs%d_ys%d_xfs%.2f_yfs%.2f.fits' %
                                   (mode, spotSepLamD, pixPerLamD,
                                    xOffsetStarFromCenterPixel,
                                    yOffsetStarFromCenterPixel, fs_x_offset,
                                    fs_y_offset)))
            imageSpotted = fits.getdata(fnImage)

            xOffsetStarFromMask = ff.calc_offset_from_spots(imageSpotted,
                                   xProbeRotDeg, xOffsetStarFromCenterPixel,
                                   yOffsetStarFromCenterPixel,
                                   fnTuning, fnTarget)
            yOffsetStarFromMask = ff.calc_offset_from_spots(imageSpotted,
                                   xProbeRotDeg+90, xOffsetStarFromCenterPixel,
                                   yOffsetStarFromCenterPixel,
                                   fnTuning, fnTarget)
            fs_x_offset_pix = fs_x_offset*pixPerLamD
            fs_y_offset_pix = fs_y_offset*pixPerLamD
            radialSepBefore = (fs_x_offset_pix**2 + fs_y_offset_pix**2)
            # fpm_x_offset and xOffsetStarFromMask should have opposite signs
            # because fpm_x_offset is the offset from the mask from the star.
            # Same idea for fpm_y_offset and yOffsetStarFromMask.
            radialSepAfter = ((fs_x_offset_pix+xOffsetStarFromMask)**2 +
                              (fs_y_offset_pix+yOffsetStarFromMask)**2)
            sepRatio = radialSepAfter/radialSepBefore
            self.assertTrue((sepRatio < 0.5),
                            msg='The radial offset distance needs to \
                                decrease by at least a factor of 2.')

    def test_calc_offset_from_spots_hlc_fpm(self):
        """Test that estimated offset of star from FPM is accurate enough."""
        xProbeRotDeg = 0
        xOffsetStarFromCenterPixel = 5
        yOffsetStarFromCenterPixel = 2

        mode = 'hlc'  # used only for testdata filename
        spotSepLamD = 2.8  # lambdaC/D
        pixPerLamD = 2.3
        # spotSepPix = spotSepLamD*pixPerLamD

        fpm_x_offset_vec = [-0.25, ]  # lambda0/D
        fpm_y_offset_vec = [0.5, ]  # lambda0/D
        nOffsets = len(fpm_x_offset_vec)

        localpath = os.path.dirname(os.path.abspath(__file__))
        fnTuning = os.path.join(localpath, 'testdata',
                              'focalfit_fixed_params_hlc_nfov.yaml')
        fnTarget = os.path.join(localpath, 'testdata',
                                'focalfit_fixed_params_ratio_1.00.yaml')

        for iOffset in range(nOffsets):

            fpm_x_offset = fpm_x_offset_vec[iOffset]  # lambda0/D
            fpm_y_offset = fpm_y_offset_vec[iOffset]  # lambda0/D

            fnImage = os.path.join(localpath, 'testdata',
                ('spots_%s_sep%.2f_res%.2f_xs%d_ys%d_xfpm%.2f_yfpm%.2f.fits' %
                                   (mode, spotSepLamD, pixPerLamD,
                                    xOffsetStarFromCenterPixel,
                                    yOffsetStarFromCenterPixel, fpm_x_offset,
                                    fpm_y_offset)))
            imageSpotted = fits.getdata(fnImage)

            xOffsetStarFromMask = ff.calc_offset_from_spots(imageSpotted,
                                   xProbeRotDeg,
                                   xOffsetStarFromCenterPixel,
                                   yOffsetStarFromCenterPixel,
                                   fnTuning, fnTarget)
            yOffsetStarFromMask = ff.calc_offset_from_spots(imageSpotted,
                                   xProbeRotDeg+90,
                                   xOffsetStarFromCenterPixel,
                                   yOffsetStarFromCenterPixel,
                                   fnTuning, fnTarget)
            # fpm_x_offset and xOffsetStarFromMask should have opposite signs
            # because fpm_x_offset is the offset from the mask from the star.
            # Same idea for fpm_y_offset and yOffsetStarFromMask.
            fpm_x_offset_pix = fpm_x_offset*pixPerLamD
            fpm_y_offset_pix = fpm_y_offset*pixPerLamD
            radialSepBefore = (fpm_x_offset_pix**2 + fpm_y_offset_pix**2)
            radialSepAfter = ((fpm_x_offset_pix+xOffsetStarFromMask)**2 +
                              (fpm_y_offset_pix+yOffsetStarFromMask)**2)
            sepRatio = radialSepAfter/radialSepBefore
            self.assertTrue((sepRatio < 0.5),
                            msg='The radial offset distance needs to \
                                decrease by at least a factor of 2.')

    def test_calc_bowtie_offsets_from_spots(self):
        """Test estimate accuracy of stellar offsets from bowtie mask."""
        xOffsetStarFromCenterPixel = 5
        yOffsetStarFromCenterPixel = 2

        mode = 'spc_spec'  # used only for testdata filename
        spotSepLamD = 8.7  # lambdaC/D
        pixPerLamD = 2.92
        # spotSepPix = spotSepLamD*pixPerLamD

        fpm_x_offset_vec = [1.0, ]  # lambda0/D
        fpm_y_offset_vec = [-1.0, ]  # lambda0/D
        nOffsets = len(fpm_x_offset_vec)

        localpath = os.path.dirname(os.path.abspath(__file__))
        fnYAML = os.path.join(localpath, 'testdata',
                              'focalfit_fixed_params_spc_spec.yaml')

        for iOffset in range(nOffsets):

            fpm_x_offset = fpm_x_offset_vec[iOffset]  # lambda0/D
            fpm_y_offset = fpm_y_offset_vec[iOffset]  # lambda0/D

            fnImage = os.path.join(localpath, 'testdata',
                ('spots_%s_sep%.2f_res%.2f_xs%d_ys%d_xfpm%.2f_yfpm%.2f.fits' %
                                   (mode, spotSepLamD, pixPerLamD,
                                    xOffsetStarFromCenterPixel,
                                    yOffsetStarFromCenterPixel, fpm_x_offset,
                                    fpm_y_offset)))
            imageSpotted = fits.getdata(fnImage)

            xOffsetStarFromMask, yOffsetStarFromMask =\
                ff.calc_bowtie_offsets_from_spots(
                    imageSpotted, xOffsetStarFromCenterPixel,
                    yOffsetStarFromCenterPixel, fnYAML)

            # fpm_x_offset and xOffsetStarFromMask should have opposite signs
            # because fpm_x_offset is the offset from the mask from the star.
            # Same idea for fpm_y_offset and yOffsetStarFromMask.
            fpm_x_offset_pix = fpm_x_offset*pixPerLamD
            fpm_y_offset_pix = fpm_y_offset*pixPerLamD
            radialSepBefore = (fpm_x_offset_pix**2 + fpm_y_offset_pix**2)
            radialSepAfter = ((fpm_x_offset_pix+xOffsetStarFromMask)**2 +
                              (fpm_y_offset_pix+yOffsetStarFromMask)**2)
            sepRatio = radialSepAfter/radialSepBefore
            self.assertTrue((sepRatio < 0.5),
                            msg='The radial offset distance needs to \
                                decrease by at least a factor of 2.')


if __name__ == '__main__':
    unittest.main()
