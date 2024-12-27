# -*- coding: utf-8 -*-
"""
Separate two phases, here a biofilm and nutrient medium, in a large three-
dimensional OCT scan using the softOCT package.
"""
from skimage import io
from skimage.filters import rank
from skimage.morphology import disk, ball
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm

import numpy as np
from scipy.signal import medfilt2d

from softOCT.src.utilities import find_substratum, level_substratum, threshold_OCT, remove_outliers

# =============================================================================
# OCT parameters
# =============================================================================

# Resolution of OCT in micron
dX = 12
dY = 12
dZ = 2.58 

# =============================================================================
# Load data
# =============================================================================
print("Mention 8-Bit")
scanFilename = r"..\sampleFiles\example2_raw.tif"


print("Loading scan...")
image = io.imread(scanFilename)

# =============================================================================
# Process data
# =============================================================================
# Find the substratum by identifying the brightest location along the vertical 
# axis. Apply a coarse median filter to reduce the influence of noise. Then,
# shift the image such that the brightest locations are aligned. This is 
# equivalent to the function call 
# substratumPos = find_substratum(image, filter_radius=11) followed by
# levelledImage = level_substratum(image, substratumPos, substratumThickness)

print("Detecting substratum...")
substratumPos = np.argmax(image,axis=2)

filterRadius = 11 # Radius for median filter
substratumPos = medfilt2d(substratumPos,kernel_size=filterRadius) 
substratumPos = substratumPos.astype(np.uint16)

print("Levelling substratum...")
substratumThickness = 6 # Thickness of the substratum in pixels
levelledImage = np.zeros(np.shape(image),dtype=np.uint8)
zDim = np.shape(image)[2]
for x in range(np.shape(image)[0]):
    for y in range(np.shape(image)[1]):
        # Read cutoff point from substratum location
        cutoff = substratumPos[x,y]
        # Move 6 pixels to take width of substratum into account
        cutoff = cutoff + substratumThickness
        # Bound highest cutoff to zDim
        if cutoff > zDim-1:
                cutoff = zDim-1
        # Enter values into array        
        levelledImage[x,y,:zDim-cutoff] = image[x,y,cutoff:]
        
        
# Binarise the scan using the characteristic shape of the histogram if no 
# bimodal distribution is given. Otherwise, Otsu's method can be used as in 
# example 1. The signal is typically represented by a bump in the right side of
# a Gaussian noise distribution. This method finds the first inflection point to
# the right of the mode of the histogram (excluding zero values). The value 
# given as manualOffset is required to reliably match the manually determined 
# thresholds.
# The corresponding function call is 
# binarisedImage, binThreshold = threshold_OCT(levelledImage,manualOffset=3)

print("Binarising scan...")
# Calculate histogram
hist=np.histogram(levelledImage,255,range=[0,255])

# Calculate second derivative
histDiff = np.diff(hist[0],n=2,prepend=0)

# Find maximum location of histogram, excluding zero value from rotating/expanding
intPeak = np.argmax(hist[0][1:])

# Find maximum in histDiff to the right of intPeak 
manualOffset = 3
binThreshold = np.argmax(histDiff[intPeak:]) + intPeak + manualOffset

# Mask image
binarisedImage = levelledImage > binThreshold


# Remove small structures with a radius below two pixels. Convert image to ubyte
# to silence a warning.
# Removes outliers similar to the remove outliers function in FIJI. A 
# rank median filter is used to remove bright points that would disappear using
# the filter without filling in gaps.
# This is implemented as filteredImage = remove_outliers(binarisedImage, radius=2, threshold=0)
# From https://forum.image.sc/t/using-remove-outliers-filter-in-pyimagej/55460/2
print("Removing outliers...")

radius = 2
filter_threshold=0
footprint_function = disk if binarisedImage.ndim == 2 else ball
footprint = footprint_function(radius=radius)
medianFiltered = rank.median(binarisedImage.astype(np.uint8), footprint)
outliers = (
    (binarisedImage > medianFiltered + filter_threshold)
)
filteredImage = np.where(outliers, medianFiltered, binarisedImage)

# =============================================================================
# Visualisation
# =============================================================================

# Figure style settings
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = 'stix'

MEDIUM_SIZE = 24
BIGGER_SIZE = 30

plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('axes', titlesize=BIGGER_SIZE)    # title fontsize
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels    
plt.rc('figure', figsize=(12,9))  # Default size of figure
plt.rcParams['figure.dpi'] = 100


print("Plotting results...")
# Raw image
fig1 = plt.figure()
ax = plt.gca()
img1 = ax.imshow(image[360,400:550,:].T,origin="lower",extent=(0, 150*dX,0,image.shape[2]*dZ))
plt.colorbar(img1,shrink=0.23,aspect=10,label=r"Intensity")
plt.xlabel(r"$x$ [$\mathrm{\mu m}$]")
plt.ylabel(r"$z$ [$\mathrm{\mu m}$]")
plt.savefig("../figures/ex2_raw.pdf")#, bbox_inches="tight")

# Processed image
viridis = mpl.colormaps.get_cmap('viridis')  # Get the viridis colormap
cmap = ListedColormap([viridis(0), viridis(255)])
# Define the boundaries for the binary values
bounds = [-0.5, 0.5, 1.5]  # Boundaries for discrete values
norm = BoundaryNorm(bounds, cmap.N)

fig2 = plt.figure()
img2 = plt.imshow(filteredImage[360,400:550,:].T,origin="lower",
           extent=(0, 150*dX,0,image.shape[2]*dZ),cmap=cmap,norm=norm)
# Add the colorbar with ticks at 0 and 1
cbar = fig2.colorbar(img2,shrink=0.23,aspect=10,label=r"Label", ticks=[0, 1])
plt.xlabel(r"$x$ [$\mathrm{\mu m}$]")
plt.ylabel(r"$z$ [$\mathrm{\mu m}$]")
plt.savefig("../figures/ex2_processed.pdf")#, bbox_inches="tight")

MEDIUM_SIZE = 50
BIGGER_SIZE = 62

plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('axes', titlesize=BIGGER_SIZE)    # title fontsize
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels    
plt.rc('figure', figsize=(12,9))  # Default size of figure
plt.rcParams['figure.dpi'] = 100


fig3 = plt.figure()
ax3 = plt.gca()
plt.hist(levelledImage.ravel(),bins=range(256))
ax3.axvline(binThreshold,color='r')
plt.xlim([170,215])
plt.ylim([0,2e7])
plt.xlabel("Intensity value")
plt.ylabel("Number of samples")
plt.savefig("../figures/ex2_histogram.pdf", bbox_inches="tight")


# Resulting height map in an interesting field of view
biofilmThickness = np.sum(filteredImage[300:450,400:550,:],axis=-1)
fig4 = plt.figure()
ax4 = plt.gca()
img4 = ax4.imshow(biofilmThickness,
                  extent=(0,biofilmThickness.shape[0]*dX*1e-3,0,biofilmThickness.shape[1]*dY*1e-3))
# Line corresponding to slice in fig1 and fig2
ax4.axhline(90*dX*1e-3,color='r')
plt.xlabel(r"$x$ [$\mathrm{mm}$]")
plt.ylabel(r"$y$ [$\mathrm{mm}$]")
plt.colorbar(img4,shrink=0.9,label=r"Film thickness [$\mathrm{\mu m}$]")
plt.savefig("../figures/ex2_heightMap.pdf", bbox_inches="tight")




