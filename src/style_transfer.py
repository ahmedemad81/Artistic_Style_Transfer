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
from color_transfer.color_transfer import color_transfer
from segmentation.segmentation import kmeans_segmentation, watershed_segmentation, canny_segmentation, otsu_segmentation_binary_mask
from utils.irls import IRLS
from denoise.denoise import denoise
from timeit import default_timer
from skimage.restoration import denoise_bilateral
from patchify import patchify
from utils.pca import pca



LMAX = 3
IMG_SIZE = 400
PATCH_SIZES = np.array([40 , 30 , 20 ,10])
SAMPLING_GAPS = np.array([20,15,10 ,5])
IALG = 5
IRLS_it = 3
IRLS_r = 0.8
PADDING_MODE = 'edge'
content_weight = 0.6

def patch_matching (flatten_Xp, patch_size, subsampling_gap, flatten_style_patches, nn_model, xp_shape):
    z = []
    sc = 0
    for Xpatch in flatten_Xp:
        flatten_Xpp = [Xpatch]

        if patch_size >= 21 :
            unflattened_Xp=Xpatch.reshape(patch_size, patch_size, 3)
            if(sc%xp_shape[0]):
                unflattened_Xp[:,:patch_size-subsampling_gap,:]=z[sc-1][:,-(patch_size-subsampling_gap):,:]
            if(sc>=xp_shape[1]):
                unflattened_Xp[:patch_size-subsampling_gap,:,:]=z[sc-xp_shape[1]][-(patch_size-subsampling_gap):,:,:]
            flatten_Xpp = unflattened_Xp.reshape(-1, patch_size * patch_size * 3)
            sc+=1

        flatten_neighbour_patch = flatten_style_patches[nn_model.kneighbors(flatten_Xpp)[1][0][0]]
        z.append(flatten_neighbour_patch.reshape(patch_size, patch_size, 3))
    return z

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

def style_transfer(content, style, segmentation_mask, color_transfer_mode = "histogram" , sigma_r=0.7, sigma_s=5):
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
    # Reverse the order of the layers so that the first layer is the smallest one 
    print("Building Pyramids ...")
    content_layers = []
    style_layers = []
    seg_layers = []
    content_layers = build_gaussian_pyramid(content, LMAX)
    style_layers = build_gaussian_pyramid(style, LMAX)
    seg_layers = build_gaussian_pyramid(segmentation_mask, LMAX)
    content_layers.reverse()      
    style_layers.reverse()  
    seg_layers.reverse()  
    
    # Intialize X with the content
    X = np.copy(content_layers[0])

    # Starting the style transfer
    print('Starting Style Transfer..')
    for L in range(LMAX):
        print('Scale ', L)
        # Add some extra noise to the output
        X = random_noise(X, mode='gaussian', var=0.01)
        

        for n in range(PATCH_SIZES.size):
            p_size = PATCH_SIZES[n]
            s_size = SAMPLING_GAPS[n]
            print('Patch Size', p_size)
            
            # pad content, style, segmentation mask and X for correct style mapping
            original_size = style_layers[L].shape[0]
            # Pad all inputs from right and bottom with p_size
            current_style = pad(style_layers[L], ((0, p_size), (0, p_size), (0, 0)), mode=PADDING_MODE)
            current_seg = pad(seg_layers[L].reshape(seg_layers[L].shape[0], seg_layers[L].shape[1], 1), ((0, p_size), (0, p_size), (0, 0)), mode=PADDING_MODE)
            current_content = pad(content_layers[L], ((0, p_size), (0, p_size), (0, 0)), mode=PADDING_MODE)
            X = pad(X, ((0, p_size), (0, p_size), (0, 0)), mode=PADDING_MODE)

            # Extracting Style patches and computing nearest neighbors
            style_patches = patchify(current_style, (p_size, p_size, 3), step=s_size)
            style_features = style_patches.reshape(-1, p_size * p_size * 3)
            projection_matrix = []
            # for small patch sizes perform PCA
            if p_size <= 21:
                proj_style_features, projection_matrix = pca(style_features)
                neighbors = NearestNeighbors(n_neighbors=1).fit(proj_style_features)
            else:
                neighbors = NearestNeighbors(n_neighbors=1).fit(style_features)

            for iter in range(IALG):
                print("Iteration",iter," ...")
                # Extracting X patches
                z=[]
                X_patches = patchify(X, (p_size, p_size, 3), step=s_size)
                X_patches_flatten = X_patches.reshape((-1, p_size * p_size * 3))
                
                if (p_size <= 21):
                    X_patches_flatten = X_patches_flatten - np.mean(X_patches_flatten, axis=0)
                    X_patches_flatten = np.matmul(X_patches_flatten, projection_matrix.T)
                    
                # Patch Matching
                z = patch_matching(X_patches_flatten, p_size, s_size, style_features, neighbors, X_patches.shape)
                #robust patch matching
                IRLS(X,z,IRLS_r,IRLS_it,(p_size,p_size,3),s_size)
                # Color Transfer
                X = color_transfer(X, current_style, color_transfer_mode)
                # Content Fusion
                X = (1.0 / (content_weight * current_seg + 1)) * (X + (content_weight * current_seg * current_content))
                # Denoising
                X = denoise(X, sigma_r, sigma_s)
                
                
            # Original size
            X = X[:original_size, :original_size, :]
            
        # Upsample X to the next level of the pyramid
        if (L != LMAX-1):    
            X = cv2.resize(X.astype(np.float32), (content_layers[L+1].shape[0], content_layers[L+1].shape[1])).astype(np.float32)
        
            
    print("Stylization Done!")
    print("Stylization time = ", default_timer()-start_time, " Seconds")
    
    
    
    return X


# Read content and style images
def main(segmentation_mode = 'watershed' , color_transfer_mode = 'histogram'):
    content_img = io.imread('input/content/eagles.jpg').astype(np.float32)/255.0
    style_img = io.imread('input/style/van_gogh.jpg').astype(np.float32)/255.0
    # Segmentation Modes (Kmeans is Default)
    if segmentation_mode == 'watershed':
        segm_mask = watershed_segmentation((content_img*255).astype(np.uint8))
    elif segmentation_mode == 'canny':
        segm_mask = canny_segmentation((content_img*255).astype(np.uint8), sigma=6, filter_size=3 , closing_iterations=2 , dilation_iterations=4)
    elif segmentation_mode == 'otsu':
        segm_mask = otsu_segmentation_binary_mask((content_img*255).astype(np.uint8))
    else:
        segm_mask = kmeans_segmentation((content_img*255).astype(np.uint8))
        
        
    content_img = (cv2.resize(content_img, (IMG_SIZE, IMG_SIZE)))
    style = (cv2.resize(style_img, (IMG_SIZE, IMG_SIZE)))
    segm_mask = (cv2.resize(segm_mask, (IMG_SIZE, IMG_SIZE)))
    original_content = content_img.copy()
    
    # Color Transfer Modes (Histogram is Default)
    content_img = color_transfer(content_img, style, color_transfer_mode)
    
    
    show_images([original_content, segm_mask, style ,content_img])
    
    # Style Transfer
    start = time.time()
    X = style_transfer(content_img, style, segm_mask , color_transfer_mode)
    end = time.time()
    print("Style Transfer took ", end - start, " seconds!")
    # Finished. Just show the images
    
    show_images([X])
    
if __name__ == "__main__":
    main()
