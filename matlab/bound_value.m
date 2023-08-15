function outVal = bound_value(inVal, maxVal)
%     Restrict a scalar to the range [-maxVal, maxVal].
% 
%     Parameters
%     ----------
%     inVal : float
%         Input value to bound.
%     maxVal : float
%         Maximum allowed value. Must be nonnegative scalar.
% 
%     Returns
%     -------
%     outVal : float
%         Bounded value.
    Check.real_scalar(inVal)
    Check.real_nonnegative_scalar(maxVal)

    if inVal > maxVal
        outVal = maxVal;
    elseif inVal < -maxVal
        outVal = -maxVal;
    else
        outVal = inVal;
    end

end
