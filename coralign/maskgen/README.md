# Purpose of MASKGEN

Convert the stored mask representations to the same size, orientation, and position as in the images from ExCAM.


# Contents of This Folder

1. README.md : This file

2. maskdesigns/ : Folder containing the as-designed mask representations for each mask configuration. The masks are stored as FITS files.

3. maskgen.py : File containing the mask fitting and generation algorithms.

4. mg_util.py : File containing support functions for mask fitting and generation.

5. materialdata/ : Folder containing the optical constants for for the HLC occulter materials (titanium, nickel, and PMGI).

6. scripts/ : Folder containing scripts to demonstrate MASKGEN functions.

7. testdata : Folder containing the FITS and YAML files used for unit and integration tests.

8. ut_maskgen.py : File containing all the unit and integration tests for maskgen.py.

9. ut_mg_util.py : File containing the unit tests for mg_util.py.

7. util/ : Folder containing utility functions and their unit tests.
* thinfilm.py : Functions to generate complex-valued HLC occulter representations from the thin film equations.
* ut_thinfilm.py
