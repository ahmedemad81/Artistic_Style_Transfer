import cv2
import numpy as np
from skimage.feature import canny
from scipy import ndimage
from skimage.exposure import equalize_hist
from skimage.color import rgb2gray

def watershed_segmentation(img , closing_iterations=2 , dilation_iterations=3):
    """ 
    Watershed segmentation algorithm
    Args:
        img : BGR image
        closing_iterations : number of iterations for closing
        dilation_iterations : number of iterations for dilation
    Returns:
        segmented_img : segmented image
    """
    
    # Convert BGR to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert image to grayscale
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    
    # Apply histogram equalization to enhance the image.
    gray = cv2.equalizeHist(gray)

    # Ensure the image is of type CV_8U (8-bit unsigned integer) to apply threshold correctly
    gray = np.uint8(gray)

    # Apply Otsu's thresholding for optimal threshold value
    _ , thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Noise removal with morphological closing
    filter = np.ones((2, 2))
    closing = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, filter, iterations=closing_iterations)

    # Apply dilation to fill in holes in the image
    dilated = cv2.dilate(closing, filter, iterations=dilation_iterations)
    
    return dilated

def canny_segmentation(img , sigma=2.5, filter_size=3 , closing_iterations=2 , dilation_iterations=4):
    """
    Segments the image using Canny edge detector
    Args:
        img : BGR image
        sigma : Standard deviation for Gaussian kernel
        filter_size : size of the filter for closing
        closing_iterations : number of iterations for closing
        dilation_iterations : number of iterations for dilation
    Returns:
        segmented_img : segmented image
    """
    
    # Convert image to grayscale
    img_gray = rgb2gray(img)
    
    #apply histogram equalization to enhance the image.
    img_gray = equalize_hist(img_gray)

    # Apply Canny detector
    edges = canny(img_gray, sigma=sigma)
    
    # edges int8
    edges = edges.astype(np.uint8)
    
    # Remove Noise using closing
    filter = np.ones((filter_size, filter_size))
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, filter, iterations=closing_iterations)
    
    # Dilate the edges
    dilated = cv2.dilate(closing, filter, dilation_iterations=dilation_iterations)
    
    # Fill holes
    filled = ndimage.binary_fill_holes(dilated)

    return filled

def kmeans_segmentation(img, k=2):
    """
    Segments the image using K-means clustering
    Args:
        img : BGR image
        k : number of clusters
    Returns:
        segmented_img : segmented image
    """
    
    # Convert image to 1D array of float32
    pixels = img.reshape((-1, 3)).astype(np.float32)

    # Define criteria and apply kmeans()
    # cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER = stop the algorithm iteration if specified accuracy, max_iter, or both criteria are met.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.5)
    _ , labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8-bit values
    centers = np.uint8(centers)

    # Map the labels to the centers
    segmented_image = centers[labels.flatten()]

    # Reshape the segmented image back to the original shape
    segmented_image = segmented_image.reshape(img.shape)
    
    # Binary Threshold inverted
    segmented_image= cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    _ , segmented_image = cv2.threshold(segmented_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return segmented_image