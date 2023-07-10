# Purpose of OCCASTRO

OCCASTRO performs two calibrations using a single processed image of satellite spots as seen at ExCAM:

1. Calculates the x- and y-offsets of the occulted star from the center pixel of the image array.
2. Calculates the radial separation of those satellite spots from the star's estimated center location.

OCCASTRO differs from FOCALFIT. OCCASTRO is used for astrometry after the focal plane mask (and possibly field stop) have been aligned, whereas FOCALFIT is used to align the focal plane mask and/or field stop.


# How OCCASTRO Works

OCCASTRO works by maximizing the summed intensity of a software mask times the processed spot image. This is true for both the stellar offset and spot separation calculations. The software mask consists of filled circles distributed in the same pattern as the satellite spots. The tuning parameters for each function and each mask configuration (NFOV, Spec, and WFOV) are stored in separate YAML files. The parts that differ for the two functions are described below.

- **calc_star_location_from_spots()**: A 2-D grid search is performed by generating new software masks for a given radial separation and at different x- and y-offsets. The best (i.e., maximum) value is obtained by a 2-D quadratic fit to the grid search output, and then a new grid search is performed about that new starting guess for the star center. This loop is repeated a specified number of times.

- **calc_spot_separation()**: A line search is performed by generating new software masks for a given stellar location and different radial separations. The best (i.e., maximum) value is obtained by a 1-D quadratic fit to the line search output, and then a new line search is performed about that new starting guess for the star center. This loop is repeated a specified number of times.

The satellite spot image needs to be pre-processed to get rid of the intensity cross terms; that is, the main input to the functions is (Iplus + Iminus)/2 - Iunprobed. The satellite spots for OCCASTRO should be located such that a) they do not overlap with each other and b) they are not obscured by a focal plane mask or field stop.
