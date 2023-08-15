# Purpose of PUPILFIT

The PUPILFIT software in this directory...
1. computes the diameter, clocking, and lateral offset of the unmasked pupil image relative to the ExCAM pixel grid.
2. computes the magnification, clocking, and lateral offset (compared to the underlying unmasked pupil's center) of each pupil mask.


#### The PUPILFIT software does not (yet) do the following:
1. Determine the new location of a shaped pupil applied to the pupil. Because of a z-axis mismatch between the flat mirror for the HLC and the reflective SPM, the pupil image can translate laterally when the SPM is moved into the beam path.

# Usage

The high-level functions are meant to be run in this order:
1. **fit_unmasked_pupil**: Run on unmasked pupil image. Return the estimates of the diameter, clocking, x-offset, and y-offset of the nominal pupil.

2. **fit_shaped_pupil_mag_clocking**: Run on image of the circular fiducials on the SPM substrate. Return the magnification and clocking of the SPMs compared to the nominal pupil.

3. **fit_lyot_stop_mag_clocking**: Run on unmasked image and image with the SPC-SPEC's Lyot stop roughly in place. Return the magnification and clocking of the Lyot stop compared to the nominal pupil. The clocking of the other Lyot stops will be determined based on microscope measurements of the masks after they are bonded to the LSAM plate.

4. **fit_pupil_mask_offsets**: Using all the previous data collected, compute the x- and y-offsets of any mask relative to the underlying pupil or mask. The self-explanatory cases are:
* 'SPCSPEC_SPM2PUPIL'
* 'SPCWFOV_SPM2PUPIL'
* 'HLCNFOV_LS2PUPIL'
* 'SPCSPEC_LS2SPM'
* 'SPCWFOV_LS2SPM'
* 'SPCSPEC_LS2PUPIL' (this last one is for calibration only when calling fit_lyot_stop_mag_clocking)
* 'SPCSPECROT_SPM2PUPIL'    _(contributed mask configuration)_
* 'SPCMSWC_SPM2PUPIL' _(contributed mask configuration)_
* 'SPCSPECROT_LS2SPM' _(contributed mask configuration)_
* 'SPCMSWC_LS2SPM' _(contributed mask configuration)_

# Techniques

I found that the best order of operations is to compute the diameter (or magnification) first, then compute the lateral offsets, and then finally the clocking angle.

Note that the magnification and clocking algorithms fail our accuracy needs when fitting shaped pupil masks, so the magnification and clocking have to be computed on other masks or fiducials instead. The lateral offset calculation works well for all masks.

### 1. Determining pupil diameter

The pupil diameter is computed by comparing the summed energy in the pupil image and in a reference pupil file, and then scaling the reference pupil's diameter by the square root of the ratio in summed energies.

Steps:
1. Need pupil amplitude and a template pupil at any resolution.
2. Sum the binary pupil amplitude in the pixels deemed to be in the pupil (by AMPTHRESH). Sum the thresholded pupil in the pixels deemed to be in the pupil (by AMPTHRESH).
3. Find the diameter by scaling the known diameter of the template pupil by this factor:
        sqrt(sumMeasured/sumTemplate)

Pros:
  - Extremely fast
  - Completely independent of pupil location and pupil clocking.
  - Very accurate (unless the measured pupil varies significantly from what is expected)

### 2. Determining lateral offsets

Steps:
1. Coarsely locate pupil to within a pixel or two using region-of-interest (ROI) summation.
  - Raster scan a region of interest (ROI) across the provided array and sum the image within the ROI. The diameter of the ROI is the pupil diameter estimated beforehand.
  - Start at a very coarse resolution that will still catch the pupil (e.g., one point every 50 pixels).
  - Take the highest summation value as the new search center, reduce the pixel step size, reduce the region of the array used for the search, re-do the survey to find the new best center.
  - Repeat until the pixel step size gets down to one, and then repeat one last time. The final estimate should be within a pixel or two of the true center if the diameter estimate was fairly accurate (within +/- 10 pixels).

2. Get a sub-pixel estimate by fitting a tip/tilt phase ramp in Fourier space.
  - Off-center crop the AMPTHRESH'ed pupil array to be centered on the pupil. Including some padding for an FFT (e.g., pad up to 1024x1024). Also pad the array-centered template pupil that has been resized to have the estimated pupil diameter.
  - FFT both the AMPTHRESH'ed pupil and the template pupil.
  - The E-fields of each FFT'ed pupil will be both be centered on the array's center pixel, but the one from measurement will have a phase ramp since it's pupil was not centered. Crop down to include only the first few and brightest Airy rings (e.g., 7 pixels across for a 512x512 array or 15 pixels across for a 1024x1024 array).
  - In a 2-D grid search, apply phase ramps in x- and y- to the FFT'ed measured pupil, subtract the FFT'ed template pupil, and square the difference. This will make a quadratic cost function with a minimum at the best centering offset of the pupil. (Make sure to understand the relationship between the magnitude of the focal plane phase ramp and the number of translated pixels in the pupil plane.)
  - Fit a 3-D paraboloid to the cost function matrix and compute the minimum of that parabola to get the best offset value.


### 3. Determining clocking

Once the pupil or mask diameter and lateral offsets are known well, clocking can be accurately computed in a straightforward manner. Clocking is computed by rotating a reference mask by several values, summing the matching pixels for each value, and then performing a quadratic fit of the squared cost function to find the best clocking value. There are two options in the code for what "matching pixels" means.

1. The default is to normalize the mask representations and call pixels matching if they are within deltaAmpMax of each other. That option works well in simulation but might not work as well with real images.
2. The non-default option is to use AMPTHRESH'ed (i.e., boolean) versions of the measured and reference pupils and call pixels matching if they have the same exact value (0 or 1). This method does not fit the non-binary mask edges as well to each other when using simulated images but could possibly work better (than option 1 above) on measured pupil images.

# Contents of This Folder

* README.md:
    This file.

* demo_fit_LS_mag_clocking.py:
    Python script that computes the magnification and clocking of the
    Lyot stop.  

* demo_fit_unmasked_pupil.py:
    Python script that computes the magnification, clocking, and
    lateral offsets for the unmasked telescope pupil.

* instructions_assorted.txt:
    Instructions on what to do for PUPILFIT sourced from the WFIRST
    Coronagraph Instrument Operations Concept Document and from
    emails from Eric Cady.

* old/:
    Folder containing some earlier versions of scripts and functions.

* pupilfit.py, pupil_open.py:
    Files containing custom functions used for the pupil fitting
    routines.

* pf_util.py, open_util.py:
    Files containing custom utility functions.

* script_gen_test_data_files.py:
    Script used to generate the FITS files used as the test data.    
    Requires the falco-python and PROPER libraries, which are not
    included in this repo.

* testdata/:
    Folder containing the YAML and FITS files used for unit and
    integration tests.

* testsuite.py:
    A simple python script used to perform the unit and integration
    tests. Uses the unittest package.

* ut_pupilfit.py, ut_pupilfit_gsw.py, ut_puilfit_open.py, ut_pf_util.py, ut_open_util.py:
    Files containing all the unit and integration tests for PUPILFIT.


# Test data generation and formatting:

Test data in the testdata folder were generated by the three demo_*.py files in this directory.

Template pupil and mask representations need to...
* be centered on the center pixel of their arrays.
* have the same expected orientation as the measured pupil. (The clocking value search can only be over a few degrees.)
* Have a high enough resolution that they don't have a lot of artifacts when generated. 1000x1000 is high enough.
