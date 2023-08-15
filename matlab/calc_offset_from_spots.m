function xOffsetStarFromMask = calc_offset_from_spots(imageSpotted, xProbeRotDeg, xOffsetStar, yOffsetStar, fnTuning)
%     """
%     Calculate the stellar offset from a focal mask.
% 
%     The offset is calculated assuming a linear or quadratic change in the ratio
%     of summed spot intensities from a pair of DM-generated satellite spots.
%     This calculation is only along one axis. To use the other axis, just add
%     90 degrees to xProbeRotDeg. The variables are named for the x-axis because
%     a rotation of zero places the spots along the horizontal axis of the array.
% 
%     Parameters
%     ----------
%     imageSpotted : numpy ndarray
%         Processed intensity image of the probes (spots). Calculated outside
%         this function from probed images as (Iplus + Iminus)/2 - Iunprobed.
%     xProbeRotDeg : float
%         How many degrees from the x-axis to rotate the regions of interest used
%         when summing the satellite spots.
%     xOffsetStar, yOffsetStar : float
%         Number of pixels in x and y that the star is offset from the center
%         pixel of the array imageSpotted.
%     fnTuning : str
%         Name of the YAML file containing the tuning parameters and other
%         data used for fitting the focal plane mask or field stop.
% 
%     Returns
%     -------
%     xOffsetStarFromMask : float
%         Estimated offset of the star from the mask along the axis defined by
%         the selected pair of satellite spots.
% 
%     Notes
%     -----
%     Variables in the YAML files are explained below:
%     spotSepPix : float
%         Expected separation of the satellite spots from the star. Used as the
%         separation for the center of the region of interest. Units of pixels.
%     roiRadiusPix : float
%         Radius of each region of interest used when summing the intensity of a
%         satellite spot. Units of pixels.
%     nSubpixels : int
%         Number of subpixels across used to make edge values of the region-of-
%         interest mask. The value of the edge pixels in the ROI is the mean of
%         all the subpixel values.
%     maxStep : float
%         Maximum allowed estimate of the star's offset from the mask. This max
%         exists because after a certain offset one spot gets completely blocked
%         and the true offset is unknown. This max value is chosen to be equal to
%         or less than the offset value at which the spot gets completely
%         blocked. Units of pixels.
%     fitCoefPow1 : float
%         Based on simulation, there is either a linear or quadratic change in
%         the summed ROI ratio versus stellar offset from the mask. fitCoefPow1
%         is the coefficient of the linear term in that polynomial fit. Must be a
%         positive scalar value to guarantee that the offset estimate has the
%         correct sign.
%     fitCoefPow2 : float
%         Based on simulation, there is either a linear or quadratic change in
%         the summed ROI ratio versus stellar offset from the mask. fitCoefPow2
%         is the coefficient of the quadratic term in that polynomial fit. Must
%         be positive to avoid divide by zero and to avoid choosing the wrong
%         answer from the quadratic formula.
%     powerOfFit : int, {1, 2}
%         Whether to perform a linear or quadratic fit. 1 for linear or 2 for
%         quadratic. If 1, then the value of fitCoefPow2 is ignored.
%     maskIsInsideSpots : int {0, 1}
%         Whether the mask doing the cutting of the spots is inside or outside
%         the spots.  An NFOV FPM has the mask on the inside (=1); an NFOV field
%         stop has the mask on the outside (=0).
%     targetRatio : float
%         Desired final ratio of summed intensities along the specified axis.
%         Unitless.
%     """
    Check.two_dim_array(imageSpotted)
    Check.real_scalar(xProbeRotDeg)
    Check.real_scalar(xOffsetStar)
    Check.real_scalar(yOffsetStar)

    xRatioTarget = 1.0;

    inp = ReadYaml(fnTuning);
    spotSepPix = inp.spotSepPix;
    roiRadiusPix = inp.roiRadiusPix;
    nSubpixels = inp.nSubpixels;
    maxStep = inp.maxStep;
    fitCoefPow1 = inp.fitCoefPow1;
    fitCoefPow2 = inp.fitCoefPow2;
    powerOfFit = inp.powerOfFit;
    maskIsInsideSpots = inp.maskIsInsideSpots;

    Check.real_positive_scalar(spotSepPix)
    Check.real_positive_scalar(roiRadiusPix)
    Check.positive_scalar_integer(nSubpixels)
    Check.real_positive_scalar(maxStep)
    Check.real_positive_scalar(fitCoefPow1)
    Check.real_positive_scalar(fitCoefPow2)
    Check.positive_scalar_integer(powerOfFit)
    if ~((powerOfFit == 1) || (powerOfFit == 2))
        error('powerOfFit must be 1 or 2')
    end
    Check.scalar_integer(maskIsInsideSpots)
    if  ~((maskIsInsideSpots == 0) || (maskIsInsideSpots == 1))
        error('maskIsInsideSpots must be 0 or 1')
    end

    [ny, nx] = size(imageSpotted);

    xProbeRotRad = pi/180*xProbeRotDeg;
    xProbePlusCoord = [sin(xProbeRotRad)*spotSepPix + yOffsetStar, ...
                       cos(xProbeRotRad)*spotSepPix + xOffsetStar];
    xProbeMinusCoord = [sin(xProbeRotRad+pi)*spotSepPix + yOffsetStar, ...
                        cos(xProbeRotRad+pi)*spotSepPix + xOffsetStar];

    xProbePlusMask = circle(nx, ny, roiRadiusPix, xProbePlusCoord(2), xProbePlusCoord(1), nSubpixels);
    xProbeMinusMask = circle(nx, ny, roiRadiusPix, xProbeMinusCoord(2), xProbeMinusCoord(1), nSubpixels);

%     figure(10); imagesc(imageSpotted); axis xy equal tight; colorbar;
%     figure(11); imagesc(xProbePlusMask); axis xy equal tight; colorbar;
%     figure(12); imagesc(xProbeMinusMask); axis xy equal tight; colorbar;
%     figure(21); imagesc(xProbePlusMask .* imageSpotted); axis xy equal tight; colorbar;
%     figure(22); imagesc(xProbeMinusMask .* imageSpotted); axis xy equal tight; colorbar;
%     drawnow;
%     pause(1);
    
    xProbePlusSum = sum(sum(xProbePlusMask .* imageSpotted));
    xProbeMinusSum = sum(sum(xProbeMinusMask .* imageSpotted));

    % Avoid divide by 0
    minVal = eps;
    if xProbeMinusSum <= 0
        xProbeMinusSum = minVal;
    end
    if xProbePlusSum <= 0
        xProbePlusSum = minVal;
    end

    % Compute offset based on ratio of summed spot intensities.
    xRatio = xProbePlusSum / xProbeMinusSum;
    if powerOfFit == 1
        xOffsetStarFromMask = calc_offset_linear(fitCoefPow1, xRatio);
        xOffsetTarget = calc_offset_linear(fitCoefPow1, xRatioTarget);

    elseif powerOfFit == 2
        xOffsetStarFromMask = calc_offset_quadratic(fitCoefPow1, fitCoefPow2, xRatio);
        xOffsetTarget = calc_offset_quadratic(fitCoefPow1, fitCoefPow2, xRatioTarget);
    end
    
    xOffsetStarFromMask = xOffsetStarFromMask - xOffsetTarget;

    xOffsetStarFromMask = bound_value(xOffsetStarFromMask, maxStep);

    % Sign of offset is inverted when spots are cut from the outside
    if  ~maskIsInsideSpots
        xOffsetStarFromMask = -1*xOffsetStarFromMask;
    end

end
