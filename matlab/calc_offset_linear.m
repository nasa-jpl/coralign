function offset = calc_offset_linear(fitCoefPow1, roiSumRatio)
%     Solve for the offset based on a linear relationship.
% 
%     Parameters
%     ----------
%     fitCoefPow1 : float
%         Based on simulation, there is either a linear or quadratic change in
%         the summed ROI ratio versus stellar offset from the mask. fitCoefPow1
%         is the coefficient of the linear term in that polynomial fit. Must be a
%         positive scalar value to guarantee that the offset estimate has the
%         correct sign.
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
    Check.real_positive_scalar(roiSumRatio)

    if roiSumRatio < 1
        roiSumRatio = 1./roiSumRatio;
        sign = -1;
    else
        sign = 1;
    end

    offset = sign*(roiSumRatio - 1)/fitCoefPow1;

end
