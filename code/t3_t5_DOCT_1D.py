# -*- coding: utf-8 -*-
"""
Process a Doppler-OCT measurement using A-scans.
"""

import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from softOCT.src.utilities import calculate_velocity


def calculate_boussinesq_solution(H,B,V_dot,M,mu,y,z):
    """
    Calculate Boussinesq's analytical solution for a rectangular duct flow.
    H: Channel height
    B: Channel width
    V_dot: Flow rate
    M: Number of terms to be considered in infinite sums
    mu: Density
    y: Horizontal position to be examined
    z: Vertical positions to be examined
    """
    
    y = np.array(y)
    z = np.array(z)
    
    G_sum = 0
    for m in range(1,M+1):
        beta_m  = (2*m-1) * np.pi / H
        G_sum += 1/(2*m-1)**5 * (np.cosh(beta_m*B)-1) / np.sinh(beta_m*B)
    G = V_dot * 1 / (H**3 * B / (12*mu) - 16 * H**4 / (np.pi**5*mu) * G_sum)
    
    if (y.ndim>0) and (z.ndim>0):
        "2D-evaluation"
        z,y = np.meshgrid(z,y)
        u_sum = 0
        for m in range(1,M+1):
            beta_m = (2*m-1)*np.pi/H
            u_sum += 1/(2*m-1)**3 * (np.sinh(beta_m*y) + np.sinh(beta_m*(B-y))) / np.sinh(beta_m*B) * np.sin(beta_m*z)
        u = G/(2*mu) * z * (H-z) - 4*G*H**2/(mu*np.pi**3) * u_sum
    else:
        "1D velocity profile"
        u_sum = 0
        for m in range(1,M+1):
            beta_m = (2*m-1)*np.pi/H
            u_sum += 1/(2*m-1)**3 * (np.sinh(beta_m*y) + np.sinh(beta_m*(B-y))) / np.sinh(beta_m*B) * np.sin(beta_m*z)
        u = G/(2*mu) * z * (H-z) - 4*G*H**2/(mu*np.pi**3) * u_sum
    return u


# =============================================================================
# Parameter declarations
# =============================================================================

# Channel parameters
theta = 5.9         # Tilt angle in degrees
V_dot = 2/3*1e-8    # for A-Scans volumetric flow rates [m^3/s].
H = 1.01e-3         # Height of the duct [m]
B = 3.25e-3         # Spanwise width of the duct [m]
n = 1.33            # Refractive Index - (for milk or water = 1.33)

area = H * B        # Cross-section [m^2]
U_B = V_dot/area    # Bulk velocity [m/s]

# OCT parameters
dx = 20e-6          # Lateral pixel size [m]
dz = 3.436e-6/n     # Axial pixel size [m]
L = 1310e-9         # Central wavelength of the OCT [m]
fs = 5.5e3          # Scan-rate of the OCT [Hz]



# =============================================================================
# Load data from file
# =============================================================================
intensityFilename = r"..\sampleFiles\ex4_1D_ModeDoppler.tiff"
phaseFilename = r"..\sampleFiles\ex4_1D_ModeDoppler_phs.tiff"


intensity = io.imread(intensityFilename)
phase = io.imread(phaseFilename)

"Align image coordinates with (z,x,t)."
intensity = intensity.transpose(1,2,0)
phase = phase.transpose(1,2,0)


# =============================================================================
# Calculate mean intensity field
# =============================================================================
meanIntensity = np.mean(intensity,axis=(1,2))

# Intensity for edge detection
binnedMeanIntensity = np.mean(intensity,axis=2)
stdIntensity = np.std(binnedMeanIntensity,axis=1)

# =============================================================================
# Calculate velocity field
# =============================================================================

# Velocity of the particle along the flow in [m/s]
# The equivalent function call is 
# velocity = calculate_velocity(L,fs,phase,n,theta)
velocity = ((L * fs * phase) / (4 * np.pi * n * np.sin(np.deg2rad(theta)))) 

# Nondimensionalise velocity and average along all time steps
u_nondim = velocity / U_B
meanU_nondim = np.mean(u_nondim,axis=(1,2))



# =============================================================================
# Show influence of averaging on the variance of the intensity
# =============================================================================
# Reshape intensity field into an Intensity[z,t] array
intResh = np.reshape(intensity,(velocity.shape[0],velocity.shape[1]*velocity.shape[2]))

# Calculate the standard deviation of the intensity depending on the size of 
# the downsampling window
stdConvergence = np.zeros((intResh.shape[0],1000))
for t in range(1000):
    segment_length = t  +1  
    tmpMean =  [np.mean(intResh[:,i:i+segment_length],axis=1) for i in range(0, intResh.shape[1], segment_length)]
    tmpMean = np.array(tmpMean).T
    stdConvergence[:,t] = np.std(tmpMean,axis=-1)

# Uncomment to see a 2D representation of this convergence
# plt.figure()
# plt.imshow(stdConvergence[:,:1000],aspect=7)

# =============================================================================
# Plot results
# =============================================================================

# Create vector of channel height coordinates
nZ = phase.shape[0] # number of pixels in axial direction
ceilingOffset = 0.34 # Vertical offset to place origin at channel ceiling
heights = np.linspace(0,-dz*nZ, num=nZ)*1e3 + ceilingOffset # Height coordinates in mm


# Create background from mean intensity at each height
plotIntensity = np.tile(np.mean(intensity,axis=(1,2))[:,None],(1,intensity.shape[1]))


# Calculate analytical solution of velocity field
M = 10 # Number of terms of infinite sum that will be considered
mu = 1e-3 # Viscosity of water in mPas

z = np.linspace(0,H,395)  # Vertical coordinates where the velocity field will be evaluated
y = B/3 # Cross-stream position of the scan

u_analytical = calculate_boussinesq_solution(H,B,V_dot,M,mu,y,z) # Analytical solution of fully developed laminar flow in a rectangular duct


# Correct magnitude of measured velocity
correction = (u_analytical.sum()/U_B) / meanU_nondim[np.where((heights>=-H*1e3) & (heights<=0))].sum() 
u_corr = meanU_nondim * correction


# Visualisation of 50 consecutive intensity measurements
# Fluctuations at the interfaces are visible
plt.figure()
plt.imshow(intensity[:,:,0],cmap='gray',extent=[0,49,heights[-1],heights[0]],aspect='auto',interpolation=None,origin="upper")
plt.ylabel(r"Vertical position [$mm$]")
plt.xlabel(r"Scan number")
plt.savefig("../figures/ex4_raw.png", bbox_inches="tight")


# Figure setup
MEDIUM_SIZE = 24
BIGGER_SIZE = 30
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('axes', titlesize=BIGGER_SIZE)    # title fontsize
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels    

plt.rc('figure', figsize=(4,9))  # Default size of figure

# Mean intensity of an A-scan. The horizontal lines highlight the interfaces and 
# example points in each phase.
pointOfInterest = np.array([130, 523, 50, 325, 600])*(-dz)*1e3 + ceilingOffset
plt.figure()
plt.imshow(plotIntensity,cmap='gray',origin="upper", extent=[0,1,heights[-1],heights[0]])
plt.ylabel(r"Vertical position [$\mathrm{\mu m}$]")
plt.axhline(y=pointOfInterest[0],color='r', linewidth='3')
plt.axhline(y=pointOfInterest[1],color='b', linewidth='3')
plt.axhline(y=pointOfInterest[2],color='g',linestyle=':', linewidth='3')
plt.axhline(y=pointOfInterest[3],color='y',linestyle=':', linewidth='3')
plt.axhline(y=pointOfInterest[4],color='magenta',linestyle=':', linewidth='3')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.savefig("../figures/ex4_meanIntensity.png", bbox_inches="tight")


# Figure setup
MEDIUM_SIZE = 32
BIGGER_SIZE = 40
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('axes', titlesize=BIGGER_SIZE)    # title fontsize
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels  
plt.rc('figure', figsize=(18,9))  # Default size of figure

# Standard deviation of intensity signal at different location, depending on 
# averaging
plt.figure()

# Wall positions
plt.semilogy(stdConvergence[130,:],'r', linewidth='2', label="Top wall")
plt.semilogy(stdConvergence[523,:],'b', linewidth='2', label="Bottom wall")

# Add a dummy plot to create a blank space in the legend
plt.plot([], [], ' ', label=" ")  

# Non-interface positions
plt.semilogy(stdConvergence[50,:],'g:', linewidth='2', label="Upper solid")
plt.semilogy(stdConvergence[350,:],'y:', linewidth='2', label="Fluid")
plt.semilogy(stdConvergence[600,:],'magenta',linestyle=':', linewidth='2', label="Lower solid")
plt.xlabel("Bin size")
plt.ylabel(r"$\sigma_{\langle I \rangle}$")
plt.xlim([0,1000])
plt.ylim([0.1, 10])
plt.legend(ncol=2,loc="lower center",bbox_to_anchor=(0.5, -0.5))
plt.savefig("../figures/ex4_edges.png", bbox_inches="tight")



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

# Compare velocity measurements with theoretical velocity profile
plt.figure()
plt.imshow(plotIntensity,cmap='gray',extent=[-1.1,2,heights[-1],heights[0]],aspect='auto',interpolation=None,origin="upper")
plt.plot(meanIntensity/np.max(meanIntensity),heights,'ro',markersize='2', mfc='none')
plt.plot(stdIntensity/np.max(stdIntensity),heights,'g', marker='o',markersize='2', mfc='none', linestyle='')
plt.plot(u_analytical/U_B,(z-H)*1e3,'-',color='orange', linewidth='6') # Shift profile to align with duct
plt.plot(u_corr,heights,'bo',markersize='2', mfc='none')
plt.legend([r"$\langle I \rangle / \langle I \rangle_\mathrm{max} $", 
            r"$\sigma_I / \sigma_\mathrm{I,max}$", 
            r"$U_\mathrm{analytical}/U_B$",r"$U_\mathrm{exp}/U_B$"],
            prop={"family":"Times New Roman"}, markerscale=4)
plt.ylabel(r"Vertical position [$\mathrm{mm}$]")
plt.savefig("../figures/ex4_results.png", bbox_inches="tight")


