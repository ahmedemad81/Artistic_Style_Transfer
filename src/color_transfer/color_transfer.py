import cv2
import numpy as np 

def color_transfer_histogram(content_img, style_img):
    """
        Transfer the color of the style image to the content image using histogram matching.
        Args:
            content_img: RGB image to transfer its color.
            style_img: RGB image to transfer its color.
        Returns:
            transferred: RGB image represents the content image with the style image color.
    
    """
    content_img = (content_img*255.0)
    style_img = (style_img*255.0)
    
    # Loop over the image channels
    for i in range(content_img.shape[2]):
        content_channel = content_img[:, :, i].flatten()
        style_channel = style_img[:, :, i].flatten()
        
        content_channel = content_channel.astype(np.uint8)
        style_channel = style_channel.astype(np.uint8)

        # Calculate histograms
        content_hist = np.bincount(content_channel, minlength=256)
        style_hist = np.bincount(style_channel, minlength=256)

        # Calculate cumulative histograms
        content_cumhist = np.cumsum(content_hist)
        style_cumhist = np.cumsum(style_hist)
        
        # Normalize cumulative histograms
        content_cumhist = content_cumhist / content_cumhist[-1]
        style_cumhist = style_cumhist / style_cumhist[-1]

        # Calculate transfer function
        # The transfer function is calculated by finding the closest pixel values between the cumulative histograms of the content and style images.
        matched = np.interp(content_cumhist, style_cumhist, np.arange(256))
        content_img[:, :, i] = matched[content_channel].reshape(content_img[:, :, i].shape).astype(np.uint8)
        

    content_img = np.clip(content_img, 0, 255)
    
    return (content_img/255.0).astype(np.float32)


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

    # Convert images to LAB color space
    content_img = cv2.cvtColor(content_img.astype(np.float32), cv2.COLOR_BGR2LAB)
    style_img = cv2.cvtColor(style_img.astype(np.float32), cv2.COLOR_BGR2LAB)
    
    # Calculate mean and standard deviation of content image and style image
    content_mean, content_std = img_stats(content_img)
    style_mean, style_std = img_stats(style_img)
    
    # Transfer color distribution from style image to content image by the following formula:
    transferred_img = (content_img - content_mean) # Subtract mean of content image
    transferred_img = transferred_img * style_std # Multiply by standard deviation of style image
    transferred_img = transferred_img / content_std # Divide by standard deviation of content image
    transferred_img = transferred_img + style_mean # Add mean of style image

    # Convert back to BGR
    transferred_img = cv2.cvtColor(transferred_img.astype(np.float32), cv2.COLOR_LAB2BGR)
    
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
    
    # Convert to uint8
    transferred_img = transferred_img.astype(np.float32)
    
    return transferred_img

def color_transfer(content_img, style_img, type):
    """
    Transfers the color distribution from style image to content image.
    
    Args:
        content_img: BGR content image
        style_img: BGR style image
        type: type of color transfer to be performed
    Returns:
        transferred_img: BGR content image with color distribution of style image 
    """
    if type == 'histogram':
        return color_transfer_histogram(content_img, style_img)
    elif type == 'lab':
        return color_transfer_lab(content_img, style_img)
    elif type == 'mean':
        return color_transfer_mean_std(content_img, style_img)