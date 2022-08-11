function offset = calc_offset_quadratic(fitCoefPow1, fitCoefPow2, roiSumRatio)
%     Solve for the offset from the quadratic equation.
% 
%     Parameters
%     ----------
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
%     roiSumRatio : float
%         Ratio of the summed energy within the regions of interest (ROI).
%         A ratio >1 means that there is more energy in the ROI along the
%         positive direction of the axis.
% 
%     Returns
%     -------
%     offset : float
%         Estimated offset in pixels.
%
    Check.real_scalar(fitCoefPow1)
    Check.real_scalar(fitCoefPow2)
    Check.real_positive_scalar(roiSumRatio)

    if roiSumRatio < 1  % Mask is to the right of star
        roiSumRatio = 1./roiSumRatio;
        sign = -1;
    else  % Mask is to the left of star
        sign = 1;
    end

    offset = sign*((-fitCoefPow1 + sqrt(fitCoefPow1*fitCoefPow1 - 4*fitCoefPow2*(1.-roiSumRatio))) / (2.*fitCoefPow2));

end
