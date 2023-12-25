import cv2
import numpy as np 

def color_transfer_histogram(content_img, style_img):
    """ Transfers the color distribution from style image to content image using histogram matching.
    Args:
        content_img : BGR content image
        style_img : BGR style image
    Returns:
        transferred_img : BGR content image with color distribution of style image 
    """
    # Convert to uint8
    content_img = content_img.astype(np.uint8)
    style_img = style_img.astype(np.uint8)
    
    # Convert BGR to RGB
    content_img = cv2.cvtColor(content_img, cv2.COLOR_BGR2RGB)
    style_img = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)

    # Flatten images to 1-D array so that we can calculate histograms
    content_hist, _ = np.histogram(content_img.reshape(-1, 3), bins=256, range=(0, 256))
    style_hist, _ = np.histogram(style_img.reshape(-1, 3), bins=256, range=(0, 256))

    # Calculate cumulative distribution function (CDF) and normalize it
    content_cdf = content_hist.cumsum() / content_hist.sum()
    style_cdf = style_hist.cumsum() / style_hist.sum()

    # Calculate transfer_table from style_cdf to content_cdf
    # It interpolates values of style_cdf so that it matches the values of content_cdf
    transfer_table = np.interp(content_cdf, style_cdf, range(256))

    # Apply transfer_table to content image
    # For example, if content_img[0, 0] is 100, then transferred_img[0, 0] will be transfer_table[100]
    # Convert to uint8 as cv2.cvtColor expects a uint8 array as input
    transferred_img_flat = transfer_table[content_img.reshape(-1)].astype(np.uint8)

    # Reshape transferred_img to its original shape
    transferred_img = transferred_img_flat.reshape(content_img.shape)

    # Convert back to BGR
    transferred_img = cv2.cvtColor(transferred_img, cv2.COLOR_RGB2BGR)
    
    # Clip values to be between 0 and 255
    transferred_img = np.clip(transferred_img, 0, 255)
    

    return transferred_img


def img_stats(img):
    """ 
    Calculates mean and standard deviation of an image
    
    Args:
        img: image
    Returns:
        mean: mean of image
        std: standard deviation of image
    """
    # compute the means and standard deviations for each axis separately in lαβ space.
    mean = np.mean(img, axis=(0, 1))
    std = np.std(img, axis=(0, 1))
    return mean, std

def color_transfer_lab(content_img, style_img):
    """
    Convert images to LAB color space and transfer color distribution from style image to content image.
    Transfers the color distribution from style image to content image using mean and standard deviation.
    transferred_img = (content_img - content_mean) / content_std * style_std + style_mean
    
    Args:
        content_img : BGR content image
        style_img : BGR style image
    Returns:
        transferred_img : BGR content image with color distribution of style image 
    """
    
    # Why LAB color space?
    # LAB color space tends to decorrelate color information, which means that the color channels are less correlated with each other. 
    # This decorrelation simplifies the process of treating color channels separately during color transfer.
    
    content_img = cv2.cvtColor(content_img, cv2.COLOR_BGR2LAB)
    style_img = cv2.cvtColor(style_img, cv2.COLOR_BGR2LAB)
    
    # Calculate mean and standard deviation of content image and style image
    content_mean, content_std = img_stats(content_img)
    style_mean, style_std = img_stats(style_img)
    
    # Transfer color distribution from style image to content image by the following formula:
    transferred_img = ((content_img - content_mean) / (content_std)) * style_std + style_mean

    # Clip values to be between 0 and 255
    # Some pixels may have invalid values, such as negative values or values greater than 255.
    transferred_img = np.clip(transferred_img, 0, 255)
    
    # Convert back to BGR
    transferred_img = cv2.cvtColor(transferred_img.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    return transferred_img

    
def color_transfer_mean_std(content_img, style_img):
    """ 
    Transfers the color distribution from style image to content image using mean and standard deviation.
    transferred_img = (content_img - content_mean) / content_std * style_std + style_mean
    
    Args : 
        content_img : BGR content image
        style_img : BGR style image
    Returns :
        transferred_img : BGR content image with color distribution of style image 
    """
    
    # Is it necessary to convert to LAB color space?
    # No, it is not necessary to convert to LAB color space.
    # You can also transfer color distribution in RGB color space.
    
    # Calculate mean and standard deviation of content image and style image
    content_mean, content_std = img_stats(content_img)
    style_mean, style_std = img_stats(style_img)
    transferred_img = ((content_img - content_mean) / (content_std)) * style_std + style_mean
    
    # Clip values to be between 0 and 255
    transferred_img = np.clip(transferred_img, 0, 255)
    
    # Convert to uint8
    transferred_img = transferred_img.astype(np.uint8)
    
    return transferred_img