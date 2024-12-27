# softOCT
Processing of Optical Coherence Tomography (OCT) data in Python

`softOCT` is developed to enable the processing of .tiff imagestacks containing OCT data. It includes functionality for detecting surfaces and interfaces, isolating phases in multiphase problems, and Doppler-OCT velocimetry.

## Installation
```bash
$ INSTALL COMMANDS HERE
```

## Usage
`softOCT` can be used to extract morphological and velocity data from OCT data.
These functions assume that the data is supplied in the form of .tiff image stacks or otherwise converted to numpy ndarrays of intensity / phase.

## Functions

first_nonzero

>Detect the first nonzero value of an array along a given axis.

find_substratum
>Detect the substratum of a scan via the plane of brightest intensity.

level_substratum

>Adjust the image to remove any warping in the substratum.

threshold_OCT

>Binarise an OCT scan, when no clear bimodal intensity distribution exists.

remove_outliers

>Remove salt-noise from an OCT scan without filling in gaps.

calculate_velocity

>Calculate the velocity field from Doppler-OCT phase data.

calculate_runtime_convergence

>Calculate convergence of a parameter to estimate the number of required scans.

# Examples
Several examples are included. 

`example1_oil_droplet.py` demonstrates a simple case of an OCT scan containing two clearly defined phases that can be separated using traditional thresholding methods.

`example2_biofilm.py` expands this to a two-phase problem where these thresholding methods struggle to separate the phases. Here, `threshold_OCT` is used to extract the geometry of a biofilm. Additionally, image warping is removed using `find_substratum` and `level_substratum`.

`example3_particle_detection.py` uses `threshold_OCT` to automatically detect sparse particles in B-scans.

`example4_DOCT_1D.py` demonstrates how Doppler-OCT measurements can be used to calculate velocity profiles and how the signals have to be processed to obtain accurate data.

`example5_DOCT_2D.py` shows how Doppler-OCT measurements can be expanded to 2D-velocity fields, DOCT is capable of measuring near-wall velocities, and how `calculate_runtime_convergence` can be used to check the number of samples required for converged results.


# Copyright notice
Copyright 2024 Cornelius Wittig

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.