import numpy as np
from patchify import patchify

def IRLS(X, matched_patches, iterations , sub_sampling , r , p_size):
    """
    Performs IRLS robust optimization between content and style patches.
    Args:
        X: Content image
        matched_patches: Matched patches from style image
        iterations: Number of iterations
        sub_sampling: Sub sampling factor
        r: r parameter
        p_size: Patch size
    Returns:
        X: Estimated X after IRLS optimization
    """
    X_patches = patchify(X, (p_size, p_size, 3), sub_sampling)
    patch_num = X_patches.shape[0]
    num_of_patches = patch_num * patch_num
    W = np.ones((num_of_patches, 1))  # Initialize weights with ones
    
    for _ in range(iterations):
        for x in range (patch_num):
            for y in range (patch_num):
                min_dist_patch =  matched_patches[x * patch_num + y] - X_patches[x, y, 0, :, :, :] # Compute the difference between the patch and the matched patch
                norm = np.linalg.norm(min_dist_patch)# Compute the norm of the difference
                W[x * patch_num + y] = np.power(norm, r - 2) # Compute the weight of the patch
                # Update the patch using the weighted difference between the patch and the matched patch
                X_patches[x, y, 0, :, :, :] +=  (min_dist_patch) * W[x * patch_num + y]
                
    return X

