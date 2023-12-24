import numpy as np
from numpy import pad
import cv2
import time
from sklearn.feature_extraction.image import extract_patches_2d
from skimage.util import random_noise
from sklearn.neighbors import NearestNeighbors
from utils.irls import IRLS
from sklearn.decomposition import PCA
from color_transfer.color_transfer import color_transfer_histogram, color_transfer_lab,color_transfer_mean_std
from denoise.denoise import denoise
import skimage.io as io
import matplotlib.pyplot as plt
from segmentation.segmentation import watershed_segmentation, canny_segmentation , kmeans_segmentation , otsu_segmentation_binary_mask
from utils.pca import *

LMAX = 3
IMG_SIZE = 400
PATCH_SIZES = np.array([21, 13, 9, 5])
SAMPLING_GAPS = np.array([28, 18, 8, 5, 3])
IALG = 1
IRLS_it = 1
IRLS_r = 0.8
PADDING_MODE = 'edge'

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

def solve_irls(X, X_patches_raw, p_index, style_patches, neighbors , projection_matrix):
    p_size = PATCH_SIZES[p_index]
    sampling_gap = SAMPLING_GAPS[p_index]
    current_size = X.shape[0]
    # Extracting Patches
    X_patches = X_patches_raw.reshape(-1, p_size * p_size * 3)
    npatches = X_patches.shape[0]
    # PCA Projection
    if p_size <= 33:
        X_patches = project(X_patches, projection_matrix)  # Projecting X to same dimention as style patches
    # Computing Nearest Neighbors
    distances, indices = neighbors.kneighbors(X_patches)
    distances += 0.0001
    # Computing Weights
    weights = np.power(distances, IRLS_r - 2)
    # Patch Accumulation
    R = np.zeros((current_size, current_size, 3), dtype=np.float32)
    Rp = extract_patches_2d(R, patch_size=(p_size, p_size),max_patches=0.95 )
    X[:] = 0
    t = 0
    for t1 in range(X_patches_raw.shape[1]):
        for t2 in range(X_patches_raw.shape[2]):
            nearest_neighbor = style_patches[indices[t, 0]]
            X_patches_raw = X_patches_raw.astype(np.float64)
            weights = weights.astype(np.float64)
            X_patches_raw += nearest_neighbor * weights[t]
            Rp += 1 * weights[t]
            t = t + 1
    R = R.astype(np.float32)
    X = X.astype(np.float32)
    R += 0.0001  # to avoid dividing by zero.
    X /= R

def style_transfer(content, style, segmentation_mask, sigma_r=0.17, sigma_s=15):
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
    content_arr = build_gaussian_pyramid(content, LMAX)
    style_arr = build_gaussian_pyramid(style, LMAX)
    segm_arr = build_gaussian_pyramid(segmentation_mask, LMAX)
    
    # Initialize X with the content + strong noise.
    X = random_noise(content_arr[LMAX - 1], mode='gaussian', var=50)
    
    # Fusion of content 
    fusion_content =[]
    fusion_style = []
    
    # Loop over all levels of the pyramid
    for i in range(LMAX):
        sx, sy = segm_arr[i].shape
        curr_segm = segm_arr[i].reshape(sx, sy, 1)
        fusion_content.append(curr_segm * content_arr[i])
        fusion_style.append(1.0 / (curr_segm + 1))
        
    print('Starting Style Transfer..')
    for L in range(LMAX - 1, -1, -1):  # over scale L
        print('Scale ', L)
        current_size = style_arr[L].shape[0]
        style_L_sx, style_L_sy, _ = style_arr[L].shape
        X = random_noise(X, mode='gaussian', var=20 / 250.0)
        for n in range(PATCH_SIZES.size):
            p_size = PATCH_SIZES[n]
            print('Patch Size', p_size)
            npatchx = int((style_L_sx - p_size) / SAMPLING_GAPS[n] + 1)	
            # Pad the image to avoid boundary artifacts
            padding = p_size - (style_L_sx - npatchx * SAMPLING_GAPS[n])
            padding_arr = ((0, padding), (0, padding), (0, 0))
            current_style = pad(style_arr[L], padding_arr, mode=PADDING_MODE)
            X = pad(X, padding_arr, mode=PADDING_MODE)
            const1 = pad(fusion_content[L], padding_arr, mode=PADDING_MODE)
            const2 = pad(fusion_style[L], padding_arr, mode=PADDING_MODE)
            style_patches = extract_patches_2d(current_style, patch_size=(p_size, p_size),max_patches=0.95 )
            
            # Preparing for NN
            style_patches = style_patches.reshape(-1, p_size * p_size * 3)
            njobs = 1
            if (L == 0) or (L == 1 and p_size <= 13):
                njobs = -1
                
            projection_matrix = 0
            # for small patch sizes perform PCA
            if p_size <= 33:
                new_style_patches, projection_matrix = pca(style_patches)
                neighbors = NearestNeighbors(n_neighbors=1, p=2, n_jobs=njobs).fit(new_style_patches)
            else:
                neighbors = NearestNeighbors(n_neighbors=1, p=2, n_jobs=njobs).fit(style_patches)
            style_patches = style_patches.reshape((-1, p_size, p_size, 3))
            
            for k in range(IALG):
                # Steps 1 & 2: Patch-Extraction and and Robust Patch Aggregation
                X_patches_raw = extract_patches_2d(X, patch_size=(p_size, p_size),max_patches=0.95 )
                for i in range(IRLS_it):
                    solve_irls(X, X_patches_raw, n, style_patches, neighbors,projection_matrix)
                # Step 3: Content Fusion
                X = const2 * (X + const1)
                # Step 4: Color Transfer
                X = color_transfer_histogram(X, style)
                # Step 5: Denoising
                X = denoise(X, sigma_r, sigma_s)
            # Discard the padding
            X = X[0:style_L_sx, 0:style_L_sy, :]
        # Upsample X to the next level of the pyramid
        if L > 0:
            sizex, sizey, _ = content_arr[L - 1].shape
            X = cv2.resize(X, (sizex, sizey))
    return X

# Read content and style images
def main(segmentation_mode = 'kmeans' , color_transfer_mode = 'histogram' , denoise_flag = True):
    content_img = io.imread('./input/content/eagles.jpg')
    style_img = io.imread('./input/style/van_gogh.jpg')
    
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
