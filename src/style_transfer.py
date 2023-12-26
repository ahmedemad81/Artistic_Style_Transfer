import numpy as np
from numpy import pad
import cv2
import time
from sklearn.feature_extraction.image import extract_patches_2d
from skimage.util import random_noise
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import skimage.io as io
import matplotlib.pyplot as plt
from color_transfer.color_transfer import color_transfer_histogram, color_transfer_lab, color_transfer_mean_std
from segmentation.segmentation import kmeans_segmentation, watershed_segmentation, canny_segmentation, otsu_segmentation_binary_mask
from utils.irls import IRLS
from denoise.denoise import denoise
from timeit import default_timer


LMAX = 3
IMG_SIZE = 400
PATCH_SIZES = np.array([33, 21, 13])
SAMPLING_GAPS = np.array([28, 18, 9])
IALG = 3
IRLS_it = 8
IRLS_r = 0.8
PADDING_MODE = 'edge'
content_weight = 0.8

# def mat2gray(image):
#     # Ensure that the image is of type float
#     image = image.astype(float)

#     # Normalize the image to the range [0, 1]
#     min_val = np.min(image)
#     max_val = np.max(image)

#     if min_val == max_val:
#         return np.zeros_like(image)
    
#     normalized_image = (image - min_val) / (max_val - min_val)
    
#     return normalized_image

# Show the figures / plots inside the notebook
def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show() 
    
def build_gaussian_pyramid(img, L):
    img_arr = []
    img_arr.append(img)  # D_L (img) = img
    for i in range(L - 1):
        img_arr.append(cv2.pyrDown(img_arr[-1].astype(np.float32)).astype(np.float32))
    return img_arr

def style_transfer(content, style, segmentation_mask, sigma_r=0.77, sigma_s=40):
    """
    Performs style transfer between content and style images.
    Args:
        content_img: Content image
        style_img: Style image
        patch_sizes: List of patch sizes to consider
        iterations: Number of iterations
        sub_sampling: Sub sampling gap between patches
        r: Robust statistic value
        noise_std: Standard deviation of noise to be added to content image
        denoise_flag: Flag to indicate whether to denoise content image or not
        color_transfer_type: Type of color transfer to be performed
    Returns:
        X: Estimated X after style transfer
    """
    start_time = default_timer() # initialize the timer to measure performance
    ## Build gaussian pyramid
    print("Building Pyramids ...")
    content_layers = []
    style_layers = []
    seg_layers = []
    # content_layers.append(content)
    # style_layers.append(style)
    # seg_layers.append(segmentation_mask)
    # for iter in range(LMAX-1, 0, -1):
    #     content_layers.append(cv2.pyrDown(content_layers[-1].astype(np.float32)).astype(np.float32))
    #     style_layers.append(cv2.pyrDown(style_layers[-1].astype(np.float32)).astype(np.float32))
    #     seg_layers.append(cv2.pyrDown(seg_layers[-1].astype(np.float32)).astype(np.float32))    
    content_layers = build_gaussian_pyramid(content, LMAX)
    style_layers = build_gaussian_pyramid(style, LMAX)
    seg_layers = build_gaussian_pyramid(segmentation_mask, LMAX)
    
    
    content_layers.reverse()      
    style_layers.reverse()  
    seg_layers.reverse()  
    

    # Initialize X with the content + strong noise.
    X = random_noise(content_layers[0], mode='gaussian', var=0.5)
    # Set up Content Fusion constants.
    fus_const1 = []
    fus_const2 = []
    for i in range(LMAX):
        sx, sy = seg_layers[i].shape
        curr_segm = seg_layers[i].reshape(sx, sy, 1)
        fus_const1.append(curr_segm * content_layers[i])
        fus_const2.append(1.0 / (curr_segm + 1))
        
    # Starting the style transfer
        
    print('Starting Style Transfer..')
    for L in range(LMAX):
        print('Scale ', L)
        # Add some extra noise to the output
        X = random_noise(X, mode='gaussian', var=0.2)

        for n in range(PATCH_SIZES.size):
            p_size = PATCH_SIZES[n]
            print('Patch Size', p_size)
            
            # pad content, style, segmentation mask and X for correct style mapping
            original_size = style_layers[L].shape[0]
            # num_patches = int((original_size - p_size) / SAMPLING_GAPS[n] + 1)	
            # pad_size = p_size - (original_size  - num_patches * SAMPLING_GAPS[n])
            pad_arr = ((0, p_size), (0, p_size), (0, 0))
            # pad all inputs
            current_style = pad(style_layers[L], pad_arr, mode='edge')
            current_seg = pad(seg_layers[L].reshape(seg_layers[L].shape[0], seg_layers[L].shape[1], 1), pad_arr, mode='edge')
            current_content = pad(content_layers[L], pad_arr, mode='edge')
            X = pad(X, pad_arr, mode='edge')
            const1 = pad(fus_const1[L], pad_arr, mode=PADDING_MODE)
            const2 = pad(fus_const2[L], pad_arr, mode=PADDING_MODE)
            
            # Extracting Style patches and computing nearest neighbors
            style_patches = extract_patches_2d(current_style, patch_size=(p_size, p_size))
            # Sub Sampling
            style_patches = style_patches[::SAMPLING_GAPS[n], :, :]
            style_features = style_patches.reshape(-1, p_size * p_size * 3)
            projection_matrix = 0
            # for small patch sizes perform PCA
            if p_size <= 21:
                n_comp = min(25,style_features.shape[0])
                proj_style_features = PCA (n_components=n_comp, svd_solver='auto').fit_transform(style_features)
                neighbors = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(proj_style_features )
            else:
                neighbors = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(style_features)
            
            # new_style_patches = PCA(n_components=0.95, svd_solver='full').fit_transform(style_patches)
            # neighbors = NearestNeighbors(n_neighbors=1, p=2, n_jobs=njobs).fit(new_style_patches)
            # new_style_patches = new_style_patches.reshape((-1, p_size, p_size, 3))
            
            for iter in range(IALG):
                print("Iteration",iter," ...")
                # Steps 1 & 2: Patch-Extraction and and Robust Patch Aggregation
                X_patches = extract_patches_2d(X, patch_size=(p_size, p_size))
                # Sub Sampling
                X_patches = X_patches[::SAMPLING_GAPS[n], :, :]
                X = IRLS(X, X_patches, style_patches, neighbors, IRLS_it, SAMPLING_GAPS[n], IRLS_r, p_size)
                
                # Step 3: Content Fusion
                X = (1.0 / (content_weight * current_seg + 1)) * (X + (content_weight * current_seg * current_content))
                
                # Step 4: Color Transfer
                X = color_transfer_histogram(X, current_style)
                
                # Step 5: Denoising
                X = denoise(X, sigma_r, sigma_s ,3 )
                
            # Original size
            X = X[:original_size, :original_size, :]
            X = X.astype(np.float32) / 255.0
            show_images([X])
            
            
            
        # Upsample X to the next level of the pyramid
        if (L != LMAX-1):    
            X = cv2.resize(X.astype(np.float32), (content_layers[L+1].shape[0], content_layers[L+1].shape[1])).astype(np.float32)
        
            
    print("Stylization Done!")
    print("Stylization time = ", default_timer()-start_time, " Seconds")
    
    
    
    return X


# Read content and style images
def main(segmentation_mode = 'watershed' , color_transfer_mode = 'histogram' , denoise_flag = True):
    content_img = io.imread('input/content/eagles.jpg')
    style_img = io.imread('input/style/van_gogh.jpg')
    
    # Segmentation Modes (Kmeans is Default)
    if segmentation_mode == 'watershed':
        segm_mask = watershed_segmentation(content_img)
    elif segmentation_mode == 'canny':
        segm_mask = canny_segmentation(content_img)
    elif segmentation_mode == 'otsu':
        segm_mask = otsu_segmentation_binary_mask(content_img)
    else:
        segm_mask = kmeans_segmentation(content_img)
        
        
    content_img = (cv2.resize(content_img, (IMG_SIZE, IMG_SIZE)))
    style = (cv2.resize(style_img, (IMG_SIZE, IMG_SIZE)))
    segm_mask = (cv2.resize(segm_mask, (IMG_SIZE, IMG_SIZE)))
    original_content = content_img.copy()
    
    # Color Transfer Modes (Histogram is Default)
    if color_transfer_mode == 'lab':
        content_img = color_transfer_lab(content_img, style)
    elif color_transfer_mode == 'mean_std':
        content_img = color_transfer_mean_std(content_img, style)
    else:
        content_img = color_transfer_histogram(content_img, style)
    
    show_images([original_content, segm_mask, style ,content_img])
    
    # Style Transfer
    start = time.time()
    X = style_transfer(content_img, style, segm_mask)
    end = time.time()
    print("Style Transfer took ", end - start, " seconds!")
    # Finished. Just show the images
    
    show_images([X])
    
if __name__ == "__main__":
    main()
