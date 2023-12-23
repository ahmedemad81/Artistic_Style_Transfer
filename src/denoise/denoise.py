import skimage.io as io
import numpy as np
import cv2
from skimage.restoration import *
from commonfunctions import *
import numpy as np

# img: RGB image to apply the filter t.
# abs_derivative: | d/dx (img(x)) |
# J[n] = I[n] + a^d (J[n-1] - I[n])
#the recursive filter is ran on 1D signal twice per iteration (left to right and right to left)
def recursive_filter(img, abs_derivative, sigma_h):
    smoothed_img = np.copy(img)
    a = np.exp(-1 * np.sqrt(2) / sigma_h) #feedback coefficient
    var = np.power(a, abs_derivative) #variance (used to calculate pixel value at each iteration)
    _, height, channels = img.shape
    for i in range(1, height): #the for loop is ran on the rows of the image
        for j in range(channels):
            smoothed_img[:, i, j] = img[:, i, j] + np.multiply(var[:, i], (smoothed_img[:, i - 1, j] - img[:, i, j]))
    for i in range(height - 2, 0, -1):
        for j in range(channels):
            smoothed_img[:, i, j] = img[:, i, j] + np.multiply(var[:, i + 1], (smoothed_img[:, i + 1, j] - img[:, i, j]))
    return smoothed_img


# img: the image to denoise.
# sigma_r: controls variance over the signal's range, according to the paper, the value 0.77 usually works well.
# sigma_s: controls variance over the signal's spatial domain, according to the paper, the value 40 usually works well.
# denoises img through applying the domain transform then a RecursiveFilter. Returns denoised image.
# iters: number of iterations to run the filter.
def denoise(img, sigma_r=0.77, sigma_s=40,iters = 3):
    # using difference to calculate the image derivative (since the image is discrete)
    dI_dx = np.abs(np.diff(img,axis=1))  
    dI_dy = np.abs(np.diff(img,axis=0))  
    x_der = np.zeros_like(img[:,:,0])
    y_der = np.zeros_like(img[:,:,0])
    x_der[:, 1:] = np.sum(dI_dx, axis=2) #starting from 1 as there's no difference for the first row/column
    y_der[1:, :] = np.sum(dI_dy, axis=2)
    # horizontal and vertical derivatives
    dhdx = (1 + sigma_s / sigma_r * x_der) #calculating the horizontal derivative according to the paper
    dvdy = np.transpose((1 + sigma_s / sigma_r * y_der)) ##calculating the vertical derivative according to the paper, this is transposed as the recursive filter is ran on the rows not the columns
    const = sigma_s * np.sqrt(3) / np.sqrt(4**iters - 1) #this const will be used in calculating sigma_h
    smoothed = np.copy(img)
    for i in range(iters):  
        sigma_h = const * 2**(iters -i-1) #-1 is added as in the paper, the iterations start from 1 not 0
        smoothed = recursive_filter(smoothed, dhdx, sigma_h) #running the filter horizontally
        smoothed = np.transpose(smoothed, axes=(1, 0, 2))  #transposing the image to run the filter vertically
        smoothed = recursive_filter(smoothed, dvdy, sigma_h) #running the filter vertically
        smoothed = np.transpose(smoothed, axes=(1, 0, 2))  
    return smoothed    

