# -*- coding: utf-8 -*-
"""
Process a 2D-Doppler-OCT measurement using the softOCT package.
"""

import numpy as np
from skimage import io
from scipy.ndimage import median_filter, gaussian_filter
import matplotlib.pyplot as plt

from softOCT.src.utilities import calculate_velocity, calculate_runtime_convergence

# =============================================================================
# Parameter declarations
# =============================================================================

# Channel parameters
theta = 9         # Tilt angle in degrees
V_dot = 2/3*1e-8    # for A-Scans volumetric flow rates [m^3/s].
H = 1.03e-3         # Height of the duct [m]
B = 3.25e-3         # Spanwise width of the duct [m]
n = 1.33            # Refractive Index - (for milk or water = 1.33)

area = H * B        # Cross-section [m^2]
U_B = V_dot/area    # Bulk velocity [m/s]

# OCT parameters
dy = 2.76e-6        # Lateral pixel size [m]
dz = 3.436e-6/n     # Axial pixel size [m]
L = 1310e-9         # Central wavelength of the OCT [m]
fs = 5.5e3          # Scan-rate of the OCT [Hz]


# =============================================================================
# Load data from file
# =============================================================================
intensityFilename = r"..\sampleFiles\ex5_2D_ModeDoppler.tif"
phaseFilename = r"..\sampleFiles\ex5_2D_ModeDoppler_phase.tif"


intensity = io.imread(intensityFilename)
phase = io.imread(phaseFilename)

"Align image coordinates with (y,z,t)."
intensity = intensity.transpose(2,1,0)
phase = phase.transpose(2,1,0)


# =============================================================================
# Trim signal to channel
# =============================================================================
intensity = intensity[130:1290,45:450,:]
phase = phase[130:1290,45:450,:]

# =============================================================================
# Calculate mean intensity field, averaging along time axis
# =============================================================================
stdIntensity = np.std(intensity,axis=(2))
meanIntensity = np.mean(intensity,axis=(2))

# =============================================================================
# Calculate velocity field
# =============================================================================

# Velocity of the particle along the flow in [mm/s]
velocity = calculate_velocity(phase, L, fs, n, theta) 

# Average velocity
meanU = np.mean(velocity,axis=(2))


# Correct magnitude of measured velocity using the known flow rate
correction = V_dot / (meanU.sum()*dy*dz)
u_corr = velocity * correction
meanU_corr = meanU * correction


# =============================================================================
# Check convergence of mean velocity at different vertical positions
# =============================================================================
y_centre = velocity.shape[0]//2
z_centre = velocity.shape[1]-velocity.shape[1]//2
z_quarter = velocity.shape[1]-velocity.shape[1]//6
z_tenth = velocity.shape[1]-velocity.shape[1]//10
z_5Percent = velocity.shape[1]-velocity.shape[1]//20

runtimeConvergence_centre = calculate_runtime_convergence(u_corr[y_centre,z_centre,:])
runtimeConvergence_quarter = calculate_runtime_convergence(u_corr[y_centre,z_quarter,:])
runtimeConvergence_tenth = calculate_runtime_convergence(u_corr[y_centre,z_tenth,:])
runtimeConvergence_5Percent = calculate_runtime_convergence(u_corr[y_centre,z_5Percent,:])
    
# =============================================================================
# Calculate shear rate
# =============================================================================
smoothMeanU_corr = gaussian_filter(meanU_corr,sigma=5)
dudy,dudz = np.gradient(smoothMeanU_corr,3)

# Switch from gradient per pixel to gradient per metre
dudy = dudy / dy
dudz = dudz / dz

shearRate = np.sqrt(dudy**2 + dudz**2)


# =============================================================================
# Plot results
# =============================================================================

# Figure style settings
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = 'stix'

MEDIUM_SIZE = 18
BIGGER_SIZE = 22

plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('axes', titlesize=BIGGER_SIZE)    # title fontsize
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels    
plt.rc('figure', figsize=(12,9))  # Default size of figure
plt.rcParams['figure.dpi'] = 100


# Show convergence of mean velocity with increasing number of samples
samples = np.arange(0,velocity.shape[-1])

plt.figure()
plt.ticklabel_format(axis='both', style='sci', scilimits=(0,3))
plt.plot(samples,runtimeConvergence_centre,'r')
plt.plot(samples,runtimeConvergence_quarter,'g')
plt.plot(samples,runtimeConvergence_tenth,'b')
plt.plot(samples,runtimeConvergence_5Percent,'y')
# 5% intervals around mean using all samples
plt.fill_between(samples,runtimeConvergence_centre[-1]*0.95, runtimeConvergence_centre[-1]*1.05, color='r',theta=0.3)
plt.fill_between(samples,runtimeConvergence_quarter[-1]*0.95, runtimeConvergence_quarter[-1]*1.05, color='g',theta=0.3)
plt.fill_between(samples,runtimeConvergence_tenth[-1]*0.95, runtimeConvergence_tenth[-1]*1.05, color='b',theta=0.3)
plt.fill_between(samples,runtimeConvergence_5Percent[-1]*0.95, runtimeConvergence_5Percent[-1]*1.05, color='y',theta=0.3)
plt.xlabel("Samples")
plt.ylabel("Mean velocity")
# plt.legend([r"50% height", r"25% height", r"10% height", r"5% height"])
plt.savefig("../figures/ex5_convergence.png")


# Show instantaneous intensity field
plt.figure()
plt.imshow(intensity[:,:,0].T,extent = [0,intensity.shape[0]*dy*1e3,0,intensity.shape[1]*dz*1e3],vmin=0,vmax=255)
plt.xlabel("y [mm]")
plt.ylabel("z [mm]")
plt.colorbar(shrink=0.33,aspect=10,label=r"Intensity")
plt.savefig("../figures/ex5_intensity.png")

# Show instantaneous velocity field
plt.figure()
plt.imshow(velocity[:,:,0].T*1e3,cmap='jet',vmin=0,vmax=8, 
           extent = [0,intensity.shape[0]*dy*1e3,0,intensity.shape[1]*dz*1e3])
plt.xlabel("y [mm]")
plt.ylabel("z [mm]")
plt.colorbar(shrink=0.33,aspect=10,label=r"U [mm/s]")
plt.savefig("../figures/ex5_velocity.png")

# Show non-outliers in instantaneous velocity field
# This represents the sparseness of accurate points in an instantaneous measurement.
vel0 = velocity[:,:,0]
outlierIndices = (vel0 <= 0.95*meanU) | (vel0 >=1.05*meanU)
vel0[outlierIndices]=0
plt.figure()
plt.imshow(vel0.T*1e3,cmap='jet',vmin=0,vmax=4, 
           extent = [0,intensity.shape[0]*dy*1e3,0,intensity.shape[1]*dz*1e3])
plt.xlabel("y [mm]")
plt.ylabel("z [mm]")
plt.colorbar(shrink=0.33,aspect=10,label=r"U [mm/s]")
plt.savefig("../figures/ex5_sparsity.png")


# Show mean velocity field
plt.figure()
plt.imshow(meanU_corr.T*1e3,cmap='jet',vmin=0,vmax=4, 
           extent = [0,intensity.shape[0]*dy*1e3,0,intensity.shape[1]*dz*1e3])
plt.xlabel("y [mm]")
plt.ylabel("z [mm]")
plt.colorbar(shrink=0.33,aspect=10,label=r"U [mm/s]")
plt.savefig("../figures/ex5_meanVelocity.png")

# Show mean shear rate field
plt.figure()
plt.imshow(shearRate.T,cmap='jet',vmin=0,vmax=5,
           extent = [0,intensity.shape[0]*dy*1e3,0,intensity.shape[1]*dz*1e3])
plt.xlabel("y [mm]")
plt.ylabel("z [mm]")
plt.colorbar(shrink=0.33,aspect=10,label=r"$\dot{\gamma}$ [1/s]")
plt.savefig("../figures/ex5_shearRate.png")


