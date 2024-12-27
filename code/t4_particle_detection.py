# -*- coding: utf-8 -*-
"""
Particle detection in ICT scans using softOCT.
"""
from skimage import io
from skimage.morphology import remove_small_objects
import matplotlib.pyplot as plt
import numpy as np

from softOCT.src.utilities import threshold_OCT


# =============================================================================
# OCT parameters
# =============================================================================

# Resolution of OCT in micron
dX = 2*2.58
dZ = 2.58 

# =============================================================================
# Load data
# =============================================================================
print("Mention 8-Bit")
scan1Filename = r"..\sampleFiles\example4_particle_detection_1.tif"
scan2Filename = r"..\sampleFiles\example4_particle_detection_2.tif"

print("Loading scans...")
intensity1 = io.imread(scan1Filename)
intensity2 = io.imread(scan2Filename)

"Align image coordinates with (x,z,t)."
intensity1 = intensity1.transpose(2,1,0)
intensity2 = intensity2.transpose(2,1,0)

"Trim field of view to channel"
intensity1 = intensity1[50:700,130:525,:]
intensity2 = intensity2[70:720,130:525,:]

# =============================================================================
# Process data
# =============================================================================
# Subtract the mean to isolate particles
# 		Optional high pass filter (2D) for sharpening
# 		Thresholding (so far, manual)
# 		Eliminate spurious small structures without modifying large particles


print("Subtracting temporal mean...")
intensity1 = intensity1 - intensity1.mean(axis=-1)[...,None] # Add None dimension at the end to enable broadcasting
intensity2 = intensity2 - intensity2.mean(axis=-1)[...,None] # Add None dimension at the end to enable broadcasting


print("Binarising scan...")
binarisedImage1, binThreshold1 = threshold_OCT(intensity1)
binarisedImage2, binThreshold2 = threshold_OCT(intensity2)



print("Removing noise...")
filteredIntensity1 = np.zeros_like(binarisedImage1)
filteredIntensity2 = np.zeros_like(binarisedImage2)
for scanID, bScan in enumerate(np.rollaxis(binarisedImage1, -1)):
    # Remove small objects from each indiidual B-scan
    filteredIntensity1[:,:,scanID] = remove_small_objects(bScan,min_size=64)
for scanID, bScan in enumerate(np.rollaxis(binarisedImage2, -1)):
    # Remove small objects from each indiidual B-scan
    filteredIntensity2[:,:,scanID] = remove_small_objects(bScan,min_size=64)


print("Calculate probability density function of particle appearance for each position")
particlePDF1 = filteredIntensity1.sum(axis=-1)/filteredIntensity1.sum()
particlePDF2 = filteredIntensity2.sum(axis=-1)/filteredIntensity2.sum()



# =============================================================================
# Visualisation
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
plt.rc('figure', figsize=(10,9))  # Default size of figure
plt.rcParams['figure.dpi'] = 100

print("Plotting results...")
# Raw intensity, frame 0
fig11 = plt.figure()
ax11 = plt.gca()
img11 = ax11.imshow(intensity1[:,:,0].T,origin="upper",extent=(0, intensity1.shape[0]*dX,0,intensity1.shape[1]*dZ))
ax11.set_aspect('equal')
plt.colorbar(img11,shrink=0.28,aspect=10,label=r"Intensity")
plt.xlabel(r"Streamwise position [$\mathrm{\mu m}$]")
plt.ylabel(r"Vertical position [$\mathrm{\mu m}$]")
plt.savefig("../figures/ex3-1-1.pdf")#, bbox_inches="tight")

fig12 = plt.figure()
ax12 = plt.gca()
img12 = ax12.imshow(intensity2[:,:,0].T,origin="upper",extent=(0, intensity2.shape[0]*dX,0,intensity2.shape[1]*dZ))
ax12.set_aspect('equal')
plt.colorbar(img12,shrink=0.28,aspect=10,label=r"Intensity")
plt.xlabel(r"Streamwise position [$\mathrm{\mu m}$]")
plt.ylabel(r"Vertical position [$\mathrm{\mu m}$]")
plt.savefig("../figures/ex3-1-2.pdf")#, bbox_inches="tight")

# Processed data, frame 0
fig21 = plt.figure()
ax21 = plt.gca()
ax21.imshow(filteredIntensity1[:,:,0].T,origin="upper",extent=(0, intensity1.shape[0]*dX,0,intensity1.shape[1]*dZ))
ax21.set_aspect('equal')
# Add space for non-existent colorbar to be equal to fig1
plt.subplots_adjust(right=0.745)  # Leave 15% of the figure width for a colorbar
plt.xlabel(r"Streamwise position [$\mathrm{\mu m}$]")
plt.ylabel(r"Vertical position [$\mathrm{\mu m}$]")
plt.savefig(r'..\figures\ex3-2-1.pdf')

fig22 = plt.figure()
ax22 = plt.gca()
ax22.imshow(filteredIntensity2[:,:,0].T,origin="upper",extent=(0, intensity2.shape[0]*dX,0,intensity2.shape[1]*dZ))
ax22.set_aspect('equal')
# Add space for non-existent colorbar to be equal to fig1
plt.subplots_adjust(right=0.745)  # Leave 15% of the figure width for a colorbar
plt.xlabel(r"Streamwise position [$\mathrm{\mu m}$]")
plt.ylabel(r"Vertical position [$\mathrm{\mu m}$]")
plt.savefig(r'..\figures\ex3-2-2.pdf')

# Probability density of particles appearing in each location
fig31 = plt.figure()
ax31 = plt.gca()
img31 = plt.imshow(particlePDF1.T,origin="upper",vmin=0,vmax=particlePDF1.max(),
                   extent=(0, intensity1.shape[0]*dX,0,intensity1.shape[1]*dZ))
ax31.set_aspect('equal')
plt.colorbar(img31,shrink=0.28,aspect=10,label=r"PDF")
plt.xlabel(r"Streamwise position [$\mathrm{\mu m}$]")
plt.ylabel(r"Vertical position [$\mathrm{\mu m}$]")
plt.savefig(r'..\figures\ex3-3-1.pdf')

fig32 = plt.figure()
ax32 = plt.gca()
img32 = plt.imshow(particlePDF2.T,origin="upper",vmin=0,vmax=particlePDF1.max(),
                   extent=(0, intensity2.shape[0]*dX,0,intensity2.shape[1]*dZ))
ax32.set_aspect('equal')
plt.colorbar(img32,shrink=0.28,aspect=10,label=r"PDF")
plt.xlabel(r"Streamwise position [$\mathrm{\mu m}$]")
plt.ylabel(r"Vertical position [$\mathrm{\mu m}$]")
plt.savefig(r'..\figures\ex3-3-2.pdf')



