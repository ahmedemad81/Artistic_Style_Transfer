import numpy as np
from numpy import pad
import cv2
import time
from skimage.util import random_noise
from sklearn.neighbors import NearestNeighbors
import skimage.io as io
import matplotlib.pyplot as plt
from color_transfer.color_transfer import color_transfer
from segmentation.segmentation import kmeans_segmentation, watershed_segmentation, canny_segmentation, otsu_segmentation_binary_mask
from utils.irls import IRLS
from denoise.denoise import denoise
from timeit import default_timer
from patchify import patchify
from utils.pca import pca

# Original Parameters
LMAX = 3 
IMG_SIZE = 400
PATCH_SIZES = np.array([33 ,21 , 13 , 9])
SAMPLING_GAPS = np.array([28 , 18 , 8 , 5])
IALG = 3
IRLS_it = 3
IRLS_r = 0.8
PADDING_MODE = 'edge'
content_weight = 0.8


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
    """
    Build a Gaussian pyramid of a given image.
    Args:
        img: Input image
        L: Number of levels in the pyramid
    Returns:
        img_arr: Gaussian pyramid of the input image
    """
    img_arr = []
    img_arr.append(img)  # D_L (img) = img
    for i in range(L - 1):
        img_arr.append(cv2.pyrDown(img_arr[-1].astype(np.float32)).astype(np.float32))
    return img_arr

def style_transfer(content, style, segmentation_mask, color_transfer_mode = "histogram" , sigma_r=0.05, sigma_s=10 , LMAX = 3 , PATCH_SIZES = PATCH_SIZES , SAMPLING_GAPS = SAMPLING_GAPS , IALG = IALG , IRLS_it = IRLS_it , IRLS_r = IRLS_r):
    """
    Performs style transfer between content and style images.
    Args:
        content: Content image
        style: Style image
        segmentation_mask: Segmentation mask of the content image
        color_transfer_mode: Color transfer mode
        sigma_r: Range sigma for bilateral filter
        sigma_s: Spatial sigma for bilateral filter
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
        if (L == 0):
            X = random_noise(X, mode='gaussian', var=50/255.0)
        else:
            X = random_noise(X, mode='gaussian', var=0.01)
        

        for n in range(PATCH_SIZES.size):
            p_size = PATCH_SIZES[n]
            s_size = SAMPLING_GAPS[n]
            print('Patch Size', p_size)
            
            original_size = style_layers[L].shape[0]
            # Pad all inputs from right and bottom with p_size
            pad_size = p_size
            current_style = pad(style_layers[L], ((0, pad_size), (0, pad_size), (0, 0)), mode=PADDING_MODE)
            current_seg = pad(seg_layers[L].reshape(seg_layers[L].shape[0], seg_layers[L].shape[1], 1), ((0, pad_size), (0, pad_size), (0, 0)), mode=PADDING_MODE)
            current_content = pad(content_layers[L], ((0, pad_size), (0, pad_size), (0, 0)), mode=PADDING_MODE)
            X = pad(X, ((0, pad_size), (0, pad_size), (0, 0)), mode=PADDING_MODE)

            # Extracting Style patches and computing nearest neighbors
            style_patches = patchify(current_style, (p_size, p_size, 3), step=s_size)
            style_features = style_patches.reshape(-1, p_size * p_size * 3)
            projection_matrix = []
            # PCA projection
            if p_size <= 21:
                proj_style_features, projection_matrix = pca(style_features)
                neighbors = NearestNeighbors(n_neighbors=1).fit(proj_style_features)
            else:
                neighbors = NearestNeighbors(n_neighbors=1).fit(style_features)

            # Iterations of the algorithm
            for iter in range(IALG):
                print("Iteration",iter," ...")
                # Extracting X patches
                matched_patches=[]
                X_patches = patchify(X, (p_size, p_size, 3), step=s_size)
                X_patches_flatten = X_patches.reshape((-1, p_size * p_size * 3))
                
                if (p_size <= 21):
                    X_patches_flatten = X_patches_flatten - np.mean(X_patches_flatten, axis=0)
                    X_patches_flatten = np.matmul(X_patches_flatten, projection_matrix.T)
                    
                # Patch Matching
                for Xpatch in X_patches_flatten:
                    flatten_Xpp = [Xpatch] # Initialize a list with the current patch
                    # Find the nearest neighbor of the current patch
                    flatten_neighbour_patch = neighbors.kneighbors(flatten_Xpp, return_distance=False)
                    chosen_patch = style_features[flatten_neighbour_patch[0][0]] 
                    # Reshape the chosen patch to be 3D patch
                    # Append the chosen patch to the matched patches list
                    matched_patches.append(chosen_patch.reshape(p_size, p_size, 3))

                # IRLS Optimization
                X = IRLS(X, matched_patches, IRLS_it , s_size , IRLS_r , p_size)
                # Color Transfer 
                X = color_transfer(X, current_style, color_transfer_mode)
                # Content Fusion as in the paper
                X = (1.0 / (content_weight * current_seg + 1)) * (X + (content_weight * current_seg * current_content))
                # Denoising
                X = denoise(X, sigma_r, sigma_s)
                
            # Original size
            X = X[:original_size, :original_size, :]
            
        # Upsample X to the next level of the pyramid
        if (L != LMAX - 1):
            X = cv2.pyrUp(X.astype(np.float32), dstsize=(content_layers[L + 1].shape[0], content_layers[L + 1].shape[1])).astype(np.float32)
        
            
    print("Stylization Done!")
    print("Stylization time = ", default_timer()-start_time, " Seconds")
    
    
    
    return X


# Read content and style images
def main( content_path, style_path , sigma_r , sigma_s , canny_sigma , canny_filter_size , closing_iterations , dilation_iterations , kmean_k  ,segmentation_mode = 'watershed' , color_transfer_mode = 'histogram' , LMAX = LMAX , PATCH_SIZES = PATCH_SIZES , SAMPLING_GAPS = SAMPLING_GAPS , IALG = IALG , IRLS_it = IRLS_it , IRLS_r = IRLS_r ):
    content_img = io.imread(content_path).astype(np.float32)/255.0
    style_img = io.imread(style_path).astype(np.float32)/255.0
    # Segmentation Modes (Kmeans is Default)
    if segmentation_mode == 'watershed':
        segm_mask = watershed_segmentation((content_img*255).astype(np.uint8) , closing_iterations, dilation_iterations)
    elif segmentation_mode == 'canny':
        segm_mask = canny_segmentation((content_img*255).astype(np.uint8), canny_sigma, canny_filter_size , closing_iterations , dilation_iterations)
    elif segmentation_mode == 'otsu':
        segm_mask = otsu_segmentation_binary_mask((content_img*255).astype(np.uint8))
    else:
        segm_mask = kmeans_segmentation((content_img*255).astype(np.uint8) , kmean_k)
        
        
    content_img = (cv2.resize(content_img, (IMG_SIZE, IMG_SIZE)))
    style = (cv2.resize(style_img, (IMG_SIZE, IMG_SIZE)))
    segm_mask = (cv2.resize(segm_mask, (IMG_SIZE, IMG_SIZE)))
    
    # Color Transfer Modes (Histogram is Default)
    content_img = color_transfer(content_img, style, color_transfer_mode)
    
    # show_images ([original_content , content_img , style , segm_mask] , ["Original Content" , "Color Transfered Content" , "Style" , "Segmentation Mask"])
    # Style Transfer
    start = time.time()
    X = style_transfer(content_img, style, segm_mask , color_transfer_mode , sigma_r , sigma_s , LMAX , PATCH_SIZES , SAMPLING_GAPS , IALG , IRLS_it , IRLS_r)
    end = time.time()
    time_taken = end - start
    print("Style Transfer took ", time_taken, " seconds!")
    # Finished. Just show the images
    
    return X , time_taken
    
if __name__ == "__main__":
    stylized_img , time_taken = main ( content_path = './input/content/eagles.jpg' , style_path = './input/style/derschrei.jpg' , sigma_r = 0.05 , sigma_s = 10 , canny_sigma = 0.5 , canny_filter_size = 3 , closing_iterations = 3 , dilation_iterations = 3 , kmean_k = 2 , segmentation_mode = 'watershed' , color_transfer_mode = 'histogram' , LMAX = LMAX , PATCH_SIZES = PATCH_SIZES , SAMPLING_GAPS = SAMPLING_GAPS , IALG = IALG , IRLS_it = IRLS_it , IRLS_r = IRLS_r)
    show_images([stylized_img] , ["Stylized Image"])
    print("Time taken = ", time_taken , " seconds")
