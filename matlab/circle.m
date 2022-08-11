function mask = circle(nx, ny, roiRadiusPix, xShear, yShear, nSubpixels)
%     """
%     Generate a circular aperture with an antialiased edge at specified offsets.
% 
%     Used as a software window for isolating a region of interest. Grayscale
%     edges are used because the detector sampling may be low enough that
%     fractional values along the edges are important.
% 
%     Parameters
%     ----------
%     nx, ny : float
%         Dimensions of the 2-D array to create.
%     roiRadiusPix : float
%         Radius of the circle in pixels.
%     xShear, yShear : float
%         Lateral offsets in pixels of the circle's center from the array's
%         center pixel.
%     nSubpixels : int
%         Each edge pixel of the circle is subdivided into a square subarray
%         nSubpixels across. The subarray is given binary values and then
%         averaged to give the edge pixel a value between 0 and 1, inclusive.
%         Must be a positive scalar integer.
% 
%     Returns
%     -------
%     mask : array_like
%         2-D array containing the circle
%     """
    Check.positive_scalar_integer(nx)
    Check.positive_scalar_integer(ny)
    Check.real_positive_scalar(roiRadiusPix)
    Check.real_scalar(xShear)
    Check.real_scalar(yShear)
    Check.positive_scalar_integer(nSubpixels)

    if mod(nx, 2) == 0
        x = linspace(-nx/2., nx/2. - 1, nx) - xShear;
    elseif mod(nx, 2) == 1
        x = linspace(-(nx-1)/2., (nx-1)/2., nx) - xShear;
    end

    if mod(ny, 2) == 0
        y = linspace(-ny/2., ny/2. - 1, ny) - yShear;
    elseif mod(ny, 2) == 1
        y = linspace(-(ny-1)/2., (ny-1)/2., ny) - yShear;
    end

    dx = x(2) - x(1);
    [X, Y] = meshgrid(x, y);
    RHO = sqrt(X.*X + Y.*Y);

    halfWindowWidth = sqrt(2.)*dx;
    mask = -1*ones(size(RHO));
    mask(abs(RHO) < roiRadiusPix - halfWindowWidth) = 1;
    mask(abs(RHO) > roiRadiusPix + halfWindowWidth) = 0;
    grayInds = find(mask == -1);
    % fprintf('Number of grayscale points = %d\n',  length(grayInds))

    dxHighRes = 1./nSubpixels;
    xUp = linspace(-(nSubpixels-1)/2., (nSubpixels-1)/2., nSubpixels)*dxHighRes;
    [Xup, Yup] = meshgrid(xUp, xUp);

    subpixelArray = zeros(nSubpixels, nSubpixels);

    % Compute the value between 0 and 1 of each edge pixel along the circle by
    % taking the mean of the binary subpixels.
    for iInterior = 1:length(grayInds)

        subpixelArray = 0*subpixelArray;

        xCenter = X(grayInds(iInterior));
        yCenter = Y(grayInds(iInterior));
        RHOHighRes = sqrt((Xup+xCenter).^2 + (Yup+yCenter).^2);
        % figure(10); imagesc(RHOHighRes); axis xy equal tight; colorbar(); drawnow;

        subpixelArray(RHOHighRes <= roiRadiusPix) = 1;
        pixelValue = sum(subpixelArray(:))/(nSubpixels*nSubpixels);
        mask(grayInds(iInterior)) = pixelValue;
    end

end
