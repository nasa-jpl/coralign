% """Test suite for FOCALFIT."""
clear

LOCALPATH = fileparts(mfilename('fullpath'));
fprintf('Local directory is %s \n', LOCALPATH);

%% HLC FPM Alignment
% def test_calc_offset_from_spots_hlc_fpm(self):
%     """Test that estimated offset of star from HLC FPM is accurate enough."""
xProbeRotDeg = 0;
xOffsetStarFromCenterPixel = 5;
yOffsetStarFromCenterPixel = 2;

mode = 'hlc';  % used only for testdata filename
spotSepLamD = 2.8;  % lambdaC/D
pixPerLamD = 2.3;
% spotSepPix = spotSepLamD*pixPerLamD

fpm_x_offset_vec = [0, -0.25, 0.2];  % lambda0/D
fpm_y_offset_vec = [1.3, 0.5, -0.1];  % lambda0/D
nOffsets = length(fpm_x_offset_vec);

fnTuning = [LOCALPATH, filesep, 'testdata', filesep, 'focalfit_fixed_params_hlc_nfov.yaml'];

for iOffset = 1:nOffsets

    fpm_x_offset = fpm_x_offset_vec(iOffset);  % lambda0/D
    fpm_y_offset = fpm_y_offset_vec(iOffset);  % lambda0/D

    fnImage = [LOCALPATH, filesep, 'testdata', filesep, ... 
        sprintf('spots_%s_sep%.2f_res%.2f_xs%d_ys%d_xfpm%.2f_yfpm%.2f.fits', ... 
        mode, spotSepLamD, pixPerLamD, xOffsetStarFromCenterPixel, yOffsetStarFromCenterPixel, ...
        fpm_x_offset, fpm_y_offset)];
    imageSpotted = fitsread(fnImage);

    xOffsetStarFromMask = calc_offset_from_spots(imageSpotted, xProbeRotDeg, ...
        xOffsetStarFromCenterPixel,  yOffsetStarFromCenterPixel, fnTuning);
    yOffsetStarFromMask = calc_offset_from_spots(imageSpotted, xProbeRotDeg+90, ...
        xOffsetStarFromCenterPixel, yOffsetStarFromCenterPixel, fnTuning);
    % fpm_x_offset and xOffsetStarFromMask should have opposite signs
    % because fpm_x_offset is the offset from the mask from the star.
    % Same idea for fpm_y_offset and yOffsetStarFromMask.
    fpm_x_offset_pix = fpm_x_offset * pixPerLamD;
    fpm_y_offset_pix = fpm_y_offset * pixPerLamD;
    radialSepBefore = sqrt(fpm_x_offset_pix^2 + fpm_y_offset_pix^2);
    radialSepAfter = sqrt((fpm_x_offset_pix+xOffsetStarFromMask)^2 + ...
        (fpm_y_offset_pix+yOffsetStarFromMask)^2);
    sepRatio = radialSepAfter/radialSepBefore;
    fprintf('The offset of the star from the mask decreased by a factor of %.3f\n', 1/sepRatio)
%     self.assertTrue((sepRatio < 0.5),
%                     msg='The radial offset distance needs to \
%                         decrease by at least a factor of 2.')
end


%% SPC-WFOV FPM Alignment
% def test_calc_offset_from_spots_hlc_fpm(self):
%     """Test that estimated offset of star from HLC FPM is accurate enough."""
xProbeRotDeg = 0;
xOffsetStarFromCenterPixel = 5;
yOffsetStarFromCenterPixel = 2;

mode = 'spc_wfov';  % used only for testdata filename
spotSepLamD = 5.4;  % lambdaC/D
pixPerLamD = 3.3;
% spotSepPix = spotSepLamD*pixPerLamD

fpm_x_offset_vec = [0.3, 0.7, -0.93];  % lambda0/D
fpm_y_offset_vec = [0.0, -0.4, 0.93];  % lambda0/D
nOffsets = length(fpm_x_offset_vec);


fnTuning = [LOCALPATH, filesep, 'testdata', filesep, 'focalfit_fixed_params_spc_wfov.yaml'];

for iOffset = 1:nOffsets

    fpm_x_offset = fpm_x_offset_vec(iOffset);  % lambda0/D
    fpm_y_offset = fpm_y_offset_vec(iOffset);  % lambda0/D

    fnImage = [LOCALPATH, filesep, 'testdata', filesep, ... 
        sprintf('spots_%s_sep%.2f_res%.2f_xs%d_ys%d_xfpm%.2f_yfpm%.2f.fits', ... 
        mode, spotSepLamD, pixPerLamD, xOffsetStarFromCenterPixel, yOffsetStarFromCenterPixel, ...
        fpm_x_offset, fpm_y_offset)];
    imageSpotted = fitsread(fnImage);

    xOffsetStarFromMask = calc_offset_from_spots(imageSpotted, xProbeRotDeg, ...
        xOffsetStarFromCenterPixel,  yOffsetStarFromCenterPixel, fnTuning);
    yOffsetStarFromMask = calc_offset_from_spots(imageSpotted, xProbeRotDeg+90, ...
        xOffsetStarFromCenterPixel, yOffsetStarFromCenterPixel, fnTuning);
    % fpm_x_offset and xOffsetStarFromMask should have opposite signs
    % because fpm_x_offset is the offset from the mask from the star.
    % Same idea for fpm_y_offset and yOffsetStarFromMask.
    fpm_x_offset_pix = fpm_x_offset * pixPerLamD;
    fpm_y_offset_pix = fpm_y_offset * pixPerLamD;
    radialSepBefore = sqrt(fpm_x_offset_pix^2 + fpm_y_offset_pix^2);
    radialSepAfter = sqrt((fpm_x_offset_pix+xOffsetStarFromMask)^2 + ...
        (fpm_y_offset_pix+yOffsetStarFromMask)^2);
    sepRatio = radialSepAfter/radialSepBefore;
    fprintf('The offset of the star from the mask decreased by a factor of %.3f\n', 1/sepRatio)
%     self.assertTrue((sepRatio < 0.5),
%                     msg='The radial offset distance needs to \
%                         decrease by at least a factor of 2.')
end


