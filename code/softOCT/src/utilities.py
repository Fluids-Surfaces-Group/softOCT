"""Processing of Optical Coherence Tomography (OCT) data in Python

````softOCT``` is developed to enable the processing of .tiff imagestacks 
containing OCT data. It includes functionality for detecting surfaces and 
interfaces, isolating phases in multiphase problems, and Doppler-OCT 
velocimetry.

Functions
---------
first_nonzero
    Detect the first nonzero value of an array along a given axis.
find_substratum
    Detect the substratum of a scan via the plane of brightest intensity.
level_substratum
    Adjust the image to remove any warping in the substratum.
threshold_OCT
    Binarise an OCT scan, when no clear bimodal intensity distribution exists.
remove_outliers
    Remove salt-noise from an OCT scan without filling in gaps.
calculate_velocity
    Calculate the velocity field from Doppler-OCT phase data.
calculate_runtime_convergence
    Calculate convergence of a parameter to estimate the number of required scans.
"""
import numpy as np
from skimage.filters import rank
from skimage.morphology import disk, ball
from scipy.signal import medfilt2d


def first_nonzero(arr, axis, invalidVal=-1):
    """Find the first nonzero element of an array along axis. If none are found,
    invalid-val is returned.
    from: https://stackoverflow.com/questions/47269390/how-to-find-first-non-zero-value-in-every-column-of-a-numpy-array
    
    Parameters: 
    ----------
    arr : non-empty ndarray of any shape
        Image data
    axis : int
        Axis along which the first nonzero element is to be found.
    invalidVal : scalar
        Value to be returned of no nonzero element is found.
        
    Returns
    -------
    first_nonzero : ndarray
        The index of the first nonzero elements found along axis
    """
    mask = arr!=0
    first_nonzero = np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalidVal) 
    return first_nonzero

def find_substratum(image,filterRadius=11):
    """Find substratum by detecting the maximum intensity in each A-scan
    
    Parameters: 
    ----------
    image : ndarray of shape (M,N,O)
        Image data with the substratum normal to the last axis.
    filterRadius : int
        Radius of the median filter used for filling gaps in the substratum.
    
    Returns
    -------
    smoothSub : ndarray of shape (M,N)
        Position index of the substratum along the last axis.
    """
    substratumPos = np.argmax(image,axis=2)
    
    # Use median filter to remove holes in the substratum
    smoothSub = medfilt2d(substratumPos,kernel_size=filterRadius) 
    smoothSub = smoothSub.astype(np.uint16)
    return smoothSub

def level_substratum(image, substratumPos, substratumThickness):
    """Build new image from content of image above the substratum.
    
    Parameters: 
    ----------
    image : ndarray of shape (M,N,O)
        Image data with the substratum normal to the last axis.
    substratumPos : ndarray of shape (M,N)
        Position index of the substratum along the last axis.
    substratumThickness : int
        Thickness of the substratum bloom above the location of maximum intensity.
    
    Returns
    -------
    levelledImage : ndarray
        Image with levelled substraum. The vertical extent is trimmed by the 
        height below the top of the substratum.
    """
    levelledImage = np.zeros(np.shape(image),dtype=np.uint8)
    zDim = np.shape(image)[2]
    for x in range(np.shape(image)[0]):
        for y in range(np.shape(image)[1]):
            # Read cutoff point from substratum location
            cutoff = substratumPos[x,y]
            # Move substratumThickness pixels to take width of substratum into account
            cutoff = cutoff + substratumThickness
            # Bound highest cutoff to zDim
            if cutoff > zDim-1:
                    cutoff = zDim-1
            # Enter values into array        
            levelledImage[x,y,:zDim-cutoff] = image[x,y,cutoff:]
    
    return levelledImage
 
def threshold_OCT(rawImage, manualOffset=3):
    """Thresholding of OCT signals based on the shape of a typical OCT scan. 
    The signal is typically represented by a bump in the right side of a Gaussian 
    noise distribution. This method finds the first inflection point to the right
    of the mode of the histogram (excluding zero values). The value given as 
    manualOffset is required to reliably match the manually determined thresholds.
    
    Parameters: 
    ----------
    rawImage : ndarray of integers in [0,255]
        Image data
    manualOffset : int
        Manual adjustment to the detected inflection point. This value may 
        depend on the particular setup.
    
    Returns
    -------
    """
    
    # Calculate histogram
    hist=np.histogram(rawImage,255,range=[0,255])
    
    # Calculate second derivative
    histDiff = np.diff(hist[0],n=2,prepend=0)
    
    # Find maximum location of histogram, excluding zero value from rotating/expanding
    intPeak = np.argmax(hist[0][1:])
    
    # Find maximum in histDiff to the right of intPeak 
    manualOffset = 3
    thresh = np.argmax(histDiff[intPeak:]) + intPeak + manualOffset
    
    # Mask image
    binImage = rawImage>thresh
    
    return binImage, thresh
 
def remove_outliers(image, radius=2, threshold=50):
    """Removes outliers similar to the remove outliers function in FIJI. A 
    rank median filter is used to remove bright points that would disappear using
    the filter without filling in gaps. This filter detects outliers that differ
    from the local median by more than threshold.
    
    Parameters: 
    ----------
    rawImage : 2D or 3D ndarray
        Image data with values between 0 and 255
    radius : int
        Radius of the filter
    threshold : scalar
        Points that exceed the local median by more that threshold are replaced
        with the local median.    
    
    Returns
    -------
    cleanedImage : ndarray
        The filtered image.
        
    Note
    ----
    The rank filter requires the image as an unsigned 8-bit integer. Therefore,
    the image is converted to a uint8 type. 
    """
    
    # Choose footprint function based on the dimensionality of image
    footprint_function = disk if image.ndim == 2 else ball
    footprint = footprint_function(radius=radius)
    
    # Detect outliers (salt-noise only)
    medianFiltered = rank.median(image.astype(np.uint8), footprint)
    outliers = image > medianFiltered + threshold
    
    # Remove outliers
    cleanedImage = np.where(outliers, medianFiltered, image)
    return cleanedImage

def calculate_velocity(phase, L, fs, n, alpha):
    """Calculate the velocity field from the Doppler phases "phase", the centre-
    wavelength L, effective A-scan rate fs, refractive index n, and sample 
    tilt angle alpha (in degrees).
    
    Parameters: 
    ----------
    phase : ndarray
        Phase data from Doppler-OCT measurement
    L : scalar
        Central wavelength of the OCT beam
    fs : scalar
        Effective A-scan rate
    n : scalar
        Refractive index of the medium 
    alpha : scalar
        Inclination angle between the wall-normal and the OCT-beam in degrees
    
    Returns
    -------
    velocity : ndarray
        Velocity field 
    """
    velocity = ((L * fs * phase) / (4 * np.pi* n * np.sin(np.deg2rad(alpha)))) 
    return velocity

def calculate_runtime_convergence(parameter):
    """Calculate the convergence of the mean of a 1D-parameter with increasing 
    number of samples.
    
    Parameters: 
    ----------
    parameter : list or 1D-ndarray
        Measurement parameter that is supposed to converge
    
    Returns
    -------
    runtimeConvergence : ndarray
        Convergence behaviour of parameter
    """
    runtimeConvergence = np.zeros_like(parameter)
    for t in range(len(parameter)):
        runtimeConvergence[t] = np.mean(parameter[:t+1])
    return runtimeConvergence
