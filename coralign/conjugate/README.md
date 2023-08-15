# Purpose of CONJUGATE

The high level function to use from this module is `flatten()`, which returns the FCM and DM commands to flatten the wavefront error in a given phase retrieval.

From the document __D-105766 AAC FDD v2.0__:

822908 - Given 1) a phase retrieval, 2) current settings for the coarse FCM, DM1, and DM2, 3) registrations for both DM1 and DM2, 4) a boolean indicating whether to include or exclude wavefront components above Z4-Z11, 5) a map of nonfunctional actuators for both DM1 and DM2, and 6) a center and normalization radius for the Zernike components, the CTC GSW shall compute coarse FCM, DM1, and DM2 settings that compensate for the residual phase above piston, tip, and tilt.

NOTE:  Not all three mechanisms may be moved in any given correction.  Zernike definitions follow the Noll 1976 convention.

1050910 - The CTC GSW routine to compensate for residual phase shall accommodate the combinations of mechanism use and inclusion/exclusion of higher-spatial-frequency wavefront described in "Table of conjugation use cases" in D-105766 AAC FDD.


# Contents of This Folder

1. README.md : This file

2. conjugate.py : File containing the conjugate algorithm and directly supporting functions.

3. util/propcustom.py : File containing functions for generating and fitting Zernike modes.

4. testdata/ : Folder containing the FITS and YAML files used for unit and integration tests.

5. ut_conjugate.py : File containing all the unit and integration tests for CONJUGATE.

6. util/ut_propcustom.py : File containing all the unit and integration tests for PROPCUSTOM.

7. scripts/ : Folder containing scripts used in developing a few functions in CONJUGATE.

8. old/ : Folder containing some earlier versions of scripts and functions.
