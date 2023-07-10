# Purpose of *gainmap*

The *gainmap* module has two main functions:

- `compute_gainmap()` computes the gain of each actuator about the provided 2-D DM command map. The units of the gains depend on the units of the inputs.
- `compute_starting_commands_for_flight()` computes the DM commands at which to start EFC on orbit given an on-orbit flat map and a delta height (not to be confused with a delta surface map). The delta height map can either be computed in simulation (by differencing the DM actuator heights before and after running EFC) or from V&V data (by differencing the actuator heights computed from actuator maps of the ground flat setting and ground dark hole setting.)

This module uses two reference FITS files to compute gains at DM commands, heights at DM commands, or DM commands at heights:
1. a datacube of DM surface heights for several DM command values (the DM command is assumed to be the same for all actuators in a given slice of the datacube)
2. a vector of DM command values corresponding to each slice of the datacube.

The vector of reference DM command values must be sorted in increasing value, and no two values can be the same. Otherwise, an exception will be raised because these properties are needed for the finite difference calculations of the gains. (Note: If the reference vector has to be sorted, then the associated height datacube slices must also be sorted the same way. That would all be done manually before calling the *gainmap* module.)

Because reference data is available at only a few DM commands, **cubic spline interpolation** is used to compute heights for given DM commands, or to compute DM commands for given heights. An exception is raised if a provided DM command or height is outside the ranges in the reference data.

Gain maps are computed in two stages. First, gains are computed at DM commands intermediate to the measured DM commands via finite differencing. Then, the gain map is found by **linearly interpolating** if within the range of intermediate DM command values or by **linearly extrapolating** if outside that range. Each actuator is independent and thus has a separate call to scipy.interpolate.interp1d().
