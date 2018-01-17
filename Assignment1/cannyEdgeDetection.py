"""
Performs canny edge detection which has four stages
1. Gaussian Smoothing
2. Gradient magnitude and direction calculation using Sobel Operator
3. Non max supression
4. Hysterersis Thresholding

All the stages are perfomed in the same order and output images are saved in the directory specified

Sample execution command 
-------------------------
python cannyEdgeDetection.py input_file_path gaussian_window_size sobel_kernel_size low_threshold high_threshold output_files_path

Arguments
---------
input_file_path        : Path to take input image from
gaussian_window_size   : Window size used for gaussian smoothing
sobel_kernel_size      : Window size used for sobel operator 
low_threshold          : Min threshold used for hysteresis thresholding
high_threshold         : Max threshold used for hysteresisThresholding
output_files_path      : Path where all the images will be saved

"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.misc import imsave
import sys


def noiseReductionUsingGaussian(image, Kx = 5, Ky = 5) :
    """
    Performs Gaussian Blurring
  
    Parameters
    ----------
    image : (M,N) ndarray
        Input image
    Kx : Int
        Kernel size in X direction
    Ky : Int
        Kernel size in Y direction
   
    Returns
    -------
    image : (M, N) ndarray
        Image after smoothing.
    """
    blur = cv2.GaussianBlur(image,(Kx,Ky),0)
    return blur

def readImage(path) :
	image = cv2.imread(path,0)
	return image


def round_angle(angle) :
    """
    Converts angle in radians to degrees and rounds off to nearest direction angle
    
    Parameters 
    ----------
    angle : Radians
        Angle to be converted
    
    Returns
    --------
    angle : Degrees
        One of the direction angle
    
    """
    #Converting angle from radians to degrees
    angle = np.rad2deg(angle) % 180
   # print(angle)
    
    #Rounding to horizontal direction
    if (0 <= angle < 22.5) or (157.5 <= angle < 180):
        angle = 0
        
    #Rounding to diagonal direction
    elif(22.5 <= angle < 67.5):
        angle = 45
        
    #Rounding to vertical direction
    elif(67.5 <= angle < 112.5):
        angle = 90
    
    #Rounding to diagonal direction
    else :
        angle = 135
        
    return angle


def intensity_gradient(image, Kx = 3, Ky = 3) :
    """
    Calculates the gradient and its direction for entire image using Sobel Operator
    
    Parameters
    ----------
    image : (M,N) ndarray
        Input image
    Kx : Int
        Kernel size in X direction
    Ky : Int
        Kernel size in Y direction
    
    Returns
    -------
    (Gradient, Direction, Ix, Iy) : Tuple
    
    """
    #Finding Gradient using sobel operator  
    #Applying Kernels to the image
    Ix = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=Kx)
    Iy = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=Ky)
    

    #Calculating the gradient magnitude and direction
    G = np.hypot(Ix, Iy)
    D = np.arctan2(Iy, Ix);
    
    #Calculating the Directions by rounding off
    M, N = D.shape
    R_D = np.zeros((M,N), dtype = np.int32)
    
    #print(D.shape)
    #print(D[0][0])
    for i in range(M) :
        for j in range(N) :
            R_D[i][j] = round_angle(D[i,j])
    
    return (G,R_D,Ix,Iy)


def non_max_supression(gradient, direction) :
    """
    Performs Non max Supression which removes any unwanted pixels which may not constitute the edge 
    
    Parameters
    -----------
    gradient : [M,N] ndarray
        Contains the gradient magnitude at each and every pixel co-ordinate
    direction : [M,N] ndarray
        Contains the direction information of the gradient
        
    Returns
    -------
    S : [M,N] array
    gradient array but this has pixels which constitute the edge, others will be marked as O
    
    """
    M, N = gradient.shape
    S = np.zeros((M,N), dtype = np.int32)
    #Todo : Dealing in a better way with boundary points
    for i in range(M):
        for j in range(N):
            if(direction[i][j] == 0) :
                if((j!= 0 and j!= N-1) and (gradient[i, j] >= gradient[i, j - 1]) 
                	and (gradient[i, j] >= gradient[i, j + 1])):
                    S[i,j] = gradient[i,j]
                    
            elif(direction[i][j] == 90) :
                if ((i!=0 and i!= M-1) and (gradient[i, j] >= gradient[i - 1, j]) 
                	and (gradient[i, j] >= gradient[i + 1, j])):
                    S[i,j] = gradient[i,j]
            
            elif(direction[i][j] == 135) :
                if ((i!=0 and i!=M-1 and j!=0 and j!= N-1 ) and (gradient[i, j] >= gradient[i - 1, j + 1]) 
                	and (gradient[i, j] >= gradient[i + 1, j - 1])):
                    S[i,j] = gradient[i,j]
            
            elif(direction[i][j] == 45) :
                if ((i!=0 and i!=M-1 and j!=0 and j!= N-1 ) and gradient[i, j] >= gradient[i - 1, j - 1]) and 
                (gradient[i, j] >= gradient[i + 1, j + 1]):
                    S[i,j] = gradient[i,j]
    
    return S
    

def hysteresisThresholding(image, low, high) :
    """
    This function decides which are all edges are really edges and which are not by means of thresholding
    based on two values
    
    Parameters
    -----------
    image : (M, N) ndarray
        Input image 
    low : Int
        Minimum value of threshold
    high : Int
        Maximum value of threshold
    
    Returns
    --------
    thresholded : (M,N) boolean ndarray 
        Binary edges with edges marked as true
    """
    low = np.clip(low, a_min=None, a_max=high)  # ensure low always below high
    mask_low = image > low
    mask_high = image > high
    # print(mask_high*1)
    # Connected components of mask_low
    labels_low, num_labels = ndimage.label(mask_low)

    # Check which connected components contain pixels from mask_high
    sums = ndimage.sum(mask_high, labels_low, np.arange(num_labels + 1))
    connected_to_high = sums > 0
    thresholded = connected_to_high[labels_low]
    return thresholded


def hsvColor(edgeImage, gradient, direction) :
    """
    Assigns colors the edges based on the direction and the intensity value of 
    the color depends on the magnitude of the gradient at the point
    
    Parameters
    -----------
    edgeImage : [M, N] ndarray
        Binary Image with edges assigned value of 255
    gradient : [M,N] ndarray
        Contains the gradient magnitude at each and every pixel co-ordinate
    direction : [M,N] ndarray
        Contains the direction information of the gradient
        
    Returns
    --------
    hsvColoredImage : [M, N] ndarray
        Image colored based on the gradient direction and magnitude
        
    """
    M,N = edgeImage.shape
    hsv_image = np.zeros((M,N,3), dtype = np.uint8) 
    max_gradient = np.max(gradient)
    min_gradient = np.min(gradient)
    for i in range(M) :
        for j in range (N) :
            if(edgeImage[i][j]) :
                v = int(255*((gradient[i][j] - min_gradient)/(max_gradient - min_gradient)))
                if(direction[i][j] == 0) :
                    hsv_image[i][j] = [0,255,v]
                elif(direction[i][j] == 45) :
                     hsv_image[i][j] = [45,255,v]
                elif(direction[i][j] == 90) :
                     hsv_image[i][j] = [90,255,v]
                else :
                     hsv_image[i][j] = [135,255,v]
                        
    return cv2.cvtColor(hsv_image,cv2.COLOR_HSV2RGB)

if __name__ == '__main__' :
	l = list(sys.argv)
	output = l[6]
	image = readImage(l[1])
	gaussianKernel = int(l[2])
	smoothImage = noiseReductionUsingGaussian(image,gaussianKernel)
	imsave(output + "smoothned_image.png", smoothImage)
	sobel = int(l[3])
	gradient, direction, Ix, Iy = intensity_gradient(smoothImage, sobel,sobel)
	imsave(output + "gradient.png", gradient)
	supressed_image = non_max_supression(gradient, direction)
	imsave(output + "supressed_image.png", supressed_image)
	low = int(l[4])
	high = int(l[5])
	thresholded = hysteresisThresholding(supressed_image, low, high)
	imsave(output + "thresholded.png",thresholded)
	hsv_color = hsvColor(thresholded, gradient, direction)
	print(hsv_color)
	imsave(output + "hsv.png", hsv_color)
