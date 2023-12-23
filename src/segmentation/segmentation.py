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

def otsu_segmentation_binary_mask(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Otsu's thresholding
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Noise removal
    kernel = np.ones((2, 2), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Sure background area
    sure_fg = cv2.dilate(closing, kernel, iterations=3)

    # Finding sure foreground area using distance transform
    # Distance transform calculates the distance of each pixel from the nearest zero pixel (background pixel)
    # DIST_L2: DistanceType ( Euclidean Distance)
    # 3: mask size
    dist_transform = cv2.distanceTransform(sure_fg, cv2.DIST_L2, 3)

    # Threshold the distance transform image to obtain only sure foreground
    _ , sure_bg = cv2.threshold(dist_transform, 0.08 * dist_transform.max(), 255, 0)

    # Finding unknown region (Region which we are not sure of containing foreground or background)
    sure_bg = np.uint8(sure_bg)
    unknown = cv2.subtract(sure_fg, sure_bg)

    # Marker labeling (Label the regions in the image)
    _ , markers = cv2.connectedComponents(sure_bg)

    # Add one to all labels so that sure background is not 0, but 1 (Assuming the background is labeled 0)
    markers = markers + 1

    # Now, mark the region of unknown with zero (Unknown region is marked 0)
    markers[unknown == 255] = 0

    # Watershed segmentation (Apply watershed segmentation)
    markers = cv2.watershed(img, markers)

    # Create a binary mask based on watershed result
    binary_mask = np.zeros_like(gray) 
    binary_mask[markers == 1] = 255  # Assuming the sure background is labeled as 1

    # Invert the binary mask
    inverted_mask = cv2.bitwise_not(binary_mask)

    return inverted_mask