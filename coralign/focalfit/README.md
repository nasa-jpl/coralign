# Purpose of FOCALFIT

FOCALFIT computes the x- and y-offsets of the star from a focal mask using a processed set of satellite spots as seen at ExCAM. The satellite spot images need to be pre-processed to get rid of the intensity cross terms; that is, the main input to the functions is (Iplus + Iminus)/2 - Iunprobed. The baselined masks involved with FOCALFIT are:
1. HLC NFOV FS
2. HLC NFOV FPM
3. SPC-WFOV FPM
4. SPC-Spec FPM.

For cases 1-3, the x- and y-offsets are computed independently of one another with separate calls to the function `focalfit.calc_offset_from_spots`; note, however, that the spots for x and y can be in the same images since they do not overlap or interfere. For case 4, the x- and y- offsets must be computed with different sets of images using the function `focalfit.calc_bowtie_offsets_from_spots`.


# How FOCALFIT Works

The FOCALFIT algorithm works by trying to equalize the energy in the DM-generated satellite spots on either side of the star along both axes. The ratio of spot intensities vs offset was computed in simulation and fit to a line or parabola. The polynomial fit parameters are stored in YAML files and used to estimate the offset for a measured spot intensity ratio.

The estimated offsets do not need to be 100% accurate. Rather, they need to be good enough that the re-centering of mask onto the star will occur within a small number of FOCALFIT iterations. The FOCALFIT algorithms were tested for offsets of up to +/- 25 microns in x and y.

#### Sign Warning:
Note that the FSAM is one focal plane after the FPAM, and so a positive move in one plane appears as a negative move in the other plane. Because focalfit.calc_offset_from_spots() is used for mask fitting at both the FPAM and FSAM, a negative sign is included for offsets computed for the FSAM. It could be that this is backwards from reality, and that the negative sign should be applied to the FPAM offset estimates. This will need to be determined with the models of the PAMs included.
