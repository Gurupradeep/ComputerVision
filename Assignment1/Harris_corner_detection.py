"""
Performs Harris corner detection.

Sample execution command 
-------------------------
python Harris_corner_detection.py input_file_path window_size sobel_kernel_size k threshold_fraction output_files_path

Arguments
---------
input_file_path        : Path to take input image from
window_size            : It is the size of neighbourhood considered for corner detection
sobel_kernel_size      : Window size used for sobel operator 
k                      : Harris detector free parameter in the equation.
threshold_fraction     : It is used to decide the threshold while calculating the corner. Threshold will be equal to
                        threshold_fraction*maximum_R_value.
output_files_path      : Path where all the images will be saved

"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.misc import imsave
import sys


def readImage(path) :
    image = cv2.imread(path,0)
    return image

def cornerColorImage(image, corner_array, threshold,window_size) :
    """
    Marks the corners in given Image
    
    parameters
    ----------
    image : (N,M) ndarray
        Input image
    corner_arry : (N,3) ndarray
        Has co-ordinates and their corresponding R values
    threshold : Float or Int
        Threshold which is used for classifying whether it's corner or not
    window_size : Int
        It is the size of neighbourhood considered for corner detection
    
    Returns
    -------
    image : (N,M) ndarray
        Image with corners marked with Red Color
    """
    #Creating a copy of the image to mark the corner pixels
    newImg = image.copy()
    color_img = cv2.cvtColor(newImg, cv2.COLOR_GRAY2RGB)
    offset = window_size/2
    height, width = image.shape
    
    count = 0;
    for i in range(offset, height - offset) :
        for j in range(offset, width - offset) :
            
            if( corner_array[count][2] > threshold ) :
                color_img.itemset((i, j, 0), 255)
                color_img.itemset((i, j, 1), 0)
                color_img.itemset((i, j, 2), 0)
            count = count + 1
    return color_img

def findCorners(image, window_size, ksize, k) :
    
    """
    Finds the corners in the image using window of all ones.

    Parameters
    ----------
    image : (N,M) ndarray
        Input image
    window_size : Integer
        It is the size of neighbourhood considered for corner detection
    ksize : Integer 
        Aperture parameter of Sobel derivative used.
    k : Float
        Harris detector free parameter in the equation.
        
    Returns
    -------
    
    corner_list : List
        List containing the R values of all points
    
    """  
        
    #Calculating the gradient
    dx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize =  ksize)
    dy = cv2.Sobel(image,cv2.CV_64F,0,1,ksize = ksize)
    
    #List which will have co ordinates of all the corners
    cornerList = []
    
    Ixx = dx**2
    Iyy = dy**2
    Ixy = dx*dy
    
    height, width = image.shape
    
    offset = window_size/2
    
    for i in range(offset, height - offset) :
        for j in range(offset, width - offset) :
            
    #Here the window used is a matrix with all ones, so directly sum can be taken 
            Ix_window = Ixx[j-offset:j+offset+1, i-offset:i+offset+1]
            Iy_window = Iyy[j-offset:j+offset+1, i-offset:i+offset+1]
            Ixy_window = Ixy[j-offset:j+offset+1, i-offset:i+offset+1] 
            
            Sxx = np.sum(Ix_window)
            Syy = np.sum(Iy_window)
            Sxy = np.sum(Ixy_window)
            
    # Find determinant and trace, use to get corner response
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            r = det - k*(trace**2)
            cornerList.append([i,j,r])
    
    return cornerList



def gaussian_kernel_2D(ksize, s):
    offset = int(ksize / 2)
    res = np.zeros((ksize, ksize))
    for i in range(-offset, offset + 1):
        for j in range(-offset, offset + 1):
            res[i + offset][j + offset] = np.exp(-(i ** 2 + j ** 2) / (2 * s ** 2)) / (2 * np.pi * s ** 2)
    res = res / np.sum(res)
    #print(res)
    #print(np.sum(res))
    return res


def findCornersGaussianWindow(image, window_size, ksize, k, std) :
    """
    Finds the corners in the image using Gaussian window.

    Parameters
    ----------
    image : (N,M) ndarray
        Input image
    window_size : Integer
        It is the size of neighbourhood considered for corner detection
    ksize : Integer 
        Aperture parameter of Sobel derivative used.
    k : Float
        Harris detector free parameter in the equation.
    std : Int
        Standard deviation used by Gaussian Kernel
    
    Returns
    -------
    
    corner_list : List
        List containing the R values of all points
    
    """  
    
    #Calculating the gradient
    dx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize = ksize)
    dy = cv2.Sobel(image,cv2.CV_64F,0,1,ksize = ksize)
    
    #List which will have co ordinates of all the corners
    cornerList = []
    
    Ixx = dx**2
    Iyy = dy**2
    Ixy = dx*dy
    
    height, width = image.shape
    
    offset = window_size/2
    
   
    #Calling function to get 2D gaussian kernel of required size
    Gaussian_weights = gaussian_kernel_2D(window_size,std)
            
    for i in range(offset, height - offset) :
        for j in range(offset, width - offset) :
            
    #Here the window used is a matrix with all ones, so directly sum can be taken 
            Ix_window = Ixx[i-offset:i+offset+1, j-offset:j+offset+1]
            Iy_window = Iyy[i-offset:i+offset+1, j-offset:j+offset+1]
            Ixy_window = Ixy[i-offset:i+offset+1, j-offset:j+offset+1] 
            
           # print(Ix_window.shape)
           # print()
            Ix_weighted = Ix_window * Gaussian_weights
            Iy_weighted = Iy_window * Gaussian_weights
            Ixy_weighted = Ixy_window * Gaussian_weights
            
            Sxx = np.sum(Ix_weighted)
            Syy = np.sum(Iy_weighted)
            Sxy = np.sum(Ixy_weighted)
            
    # Find determinant and trace, use to get corner response
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            r = det - k*(trace**2)
           
            cornerList.append([i,j,r])
            
    return cornerList;

if __name__ == '__main__' :
    l = list(sys.argv)
    image = readImage(l[1])
    window_size = int(l[2])
    kernel_size = int(l[3])
    k = float(l[4])
    threshold_fraction = float(l[5])
    output = l[6]

    corners = findCornersGaussianWindow(image, window_size, kernel_size, k, 1)
    corners = np.array(corners)
    max_value = np.max(corners[:,2])
   # print(maxi)
   # print(threshold_fraction)
   # print(k)
    corner_image = cornerColorImage(image,corners,threshold_fraction*max_value,window_size)

    imsave(output + "corner_image.png", corner_image)