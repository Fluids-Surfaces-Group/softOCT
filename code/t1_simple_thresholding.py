# -*- coding: utf-8 -*-
"""
Separate two clearly defined phases in an OCT scan.
"""

import numpy as np
from skimage import io
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops_table
from skimage.morphology import remove_small_objects
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm


from softOCT.src.utilities import first_nonzero

# =============================================================================
# OCT parameters
# =============================================================================

# Resolution of OCT in micron
dX = 5
dY = 5
dZ = 2.58 


# =============================================================================
# Load image and convert into binarised data
# =============================================================================


filename = r"..\sampleFiles\example1_oil_droplet_rotated_cropped.tif"

image = io.imread(filename)

# If necessary, use only one channel from RGB image
# image = image[:,:,:,0]

# Align image coordinates with groove coordinates
image = image.transpose(2,0,1)

# Trim image to droplet
image = image[160:440,:,:]

# Calculate threshold value using Otsu's method
filterThreshold = threshold_otsu(image)

# Binarise the image. 
binImage = image < filterThreshold




# =============================================================================
# Isolate drop
# =============================================================================

# Remove small objects resulting from a noisy signal, optional
cleanImage = remove_small_objects(binImage,min_size=8)

# Label regions
labelImage = label(cleanImage)

# Calculate size of connected regions
regions = regionprops_table(labelImage,properties=('label','area'))

# Identify largest object, i.e. opaque medium. Add one to account for background
largestRegion = np.argmax(regions['area']) + 1

# Isolate largest object, i.e. the drop of oil
finalImage = labelImage == largestRegion



# =============================================================================
# Calculate height map
# =============================================================================

# Find the interface by locating the first foreground element along the vertical axis
interface = finalImage.shape[2]-first_nonzero(finalImage, axis=2, invalidVal=finalImage.shape[2])

# Convert from pixels to microns
heightMap = interface * dZ


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


# Raw image
# Show results for one image slice. Note, that the histogram is calculated on the full image stack
fig1 = plt.figure()
img1 = plt.imshow(image[:,150,:].T,extent=(0,image.shape[0]*dX,0,image.shape[2]*dZ),vmin=0,vmax=255)
plt.colorbar(img1,shrink=0.3,aspect=10,label=r"Intensity")
plt.xlabel(r"$x$ [$\mathrm{\mu m}$]")
plt.ylabel(r"$z$ [$\mathrm{\mu m}$]")
plt.savefig("../figures/ex1_raw.pdf")#, bbox_inches="tight")

# Plot processed image
viridis = mpl.colormaps.get_cmap('viridis')  # Get the viridis colormap
cmap = ListedColormap([viridis(0), viridis(255)])
# Define the boundaries for the binary values
bounds = [-0.5, 0.5, 1.5]  # Boundaries for discrete values
norm = BoundaryNorm(bounds, cmap.N)

fig2 = plt.figure()
img2 = plt.imshow(finalImage[:,150,:].T,extent=(0,image.shape[0]*dX,0,image.shape[2]*dZ),
                  cmap=cmap,norm=norm)
# Add the colorbar with ticks at 0 and 1
cbar = fig2.colorbar(img2,shrink=0.3,aspect=10,label=r"Label", ticks=[0, 1])
plt.xlabel(r"$x$ [$\mathrm{\mu m}$]")
plt.ylabel(r"$z$ [$\mathrm{\mu m}$]")
plt.savefig("../figures/ex1_processed.pdf")#, bbox_inches="tight")


MEDIUM_SIZE = 50
BIGGER_SIZE = 62

plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('axes', titlesize=BIGGER_SIZE)    # title fontsize
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels    


# Histogram of intensities
fig3 = plt.figure()
ax = plt.gca()
ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
plt.hist(image.ravel(),bins=range(256))
plt.xlim([0,255])
ax.axvline(filterThreshold,color='r')
ax.set_xlabel("Intensity value")
ax.set_ylabel("Number of samples")
plt.savefig("../figures/ex1_histogram.pdf", bbox_inches="tight")

# Plot the surface
fig4 = plt.figure()
ax4 = plt.gca()
img4 = plt.imshow(heightMap.T,extent=(0,image.shape[0]*dX*1e-3,0,image.shape[1]*dY*1e-3))
ax4.axhline(150*dY*1e-3, color='r')
plt.colorbar(img4,shrink=0.95,label=r"Drop height [$\mathrm{\mu m}$]")
# Remove decimal after zero
xlabels = [item.get_text() for item in ax4.get_xticklabels()]
xlabels[0] = '0'  
ylabels = [item.get_text() for item in ax4.get_yticklabels()]
ylabels[0] = '0'  
ax4.set_xticklabels(xlabels)
ax4.set_yticklabels(ylabels)
plt.xlabel(r"$x$ [$\mathrm{mm}$]")
plt.ylabel(r"$y$ [$\mathrm{mm}$]")
plt.savefig("../figures/ex1_heightMap.pdf", bbox_inches="tight")



