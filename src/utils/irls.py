import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.decomposition import PCA

def IRLS(X, X_patches, style_patches, neighbors, iterations , sub_sampling , r , p_size ):
    """
    Performs IRLS robust optimization between content and style patches.
    Args:
        X: Content before optimization 
        X_patches: Content patches (2D Patches)
        style_patches: Style patches (2D Patches)
        neighbors: Number of neighbors to consider
        patch_size: Size of patch
        iterations: Number of iterations
        sub_sampling: Sub sampling gap between patches
        r: Robust statistic value
    Returns:
        X: Estimated X after IRLS optimization
    """
    for i in range(iterations):
        print('Iteration ', i)
        current_size = X.shape[0]
        # Extracting X patches
        X_patches_raw = extract_patches_2d(X, patch_size=(p_size, p_size))
        # Sub Sampling
        X_patches = X_patches_raw[::sub_sampling, :, :]
        X_patches = X_patches.reshape((-1, X_patches.shape[1] * X_patches.shape[2] * 3))
        # PCA Projection
        if p_size <= 21:
            n_comp = min(25,style_patches.shape[0])
            X_patches = PCA (n_components=n_comp, svd_solver='auto').fit_transform(X_patches)
            
        style_patches_iter = style_patches # Style patches for current iteration
        style_patches_iter = style_patches_iter.reshape((-1, style_patches.shape[1] *  style_patches.shape[2] * 3))
        
        # X_patches = PCA(n_components=0.95, svd_solver='full').fit_transform(X_patches)
        # X_patches = X_patches.reshape((-1, p_size, p_size, 3))
        # Computing Nearest Neighbors
        distances, indices = neighbors.kneighbors(X_patches)
        distances += 0.0001
        # Computing Weights
        weights = np.power(distances, r - 2 )
        # Patch Accumulation
        R = np.zeros((current_size, current_size, 3), dtype=np.float32)
        Rp = extract_patches_2d(R, patch_size=(p_size, p_size))
        # Sub Sampling
        Rp = Rp[::sub_sampling, :, :]
        X[:] = 0
        t = 0
        for j in range(0, current_size - p_size + 1, sub_sampling):
            for k in range(0, current_size - p_size + 1, sub_sampling):
                nearest_neighbor_patch = style_patches_iter[indices[t]]
                nearest_neighbor_patch = nearest_neighbor_patch.reshape((p_size, p_size, 3))
                X = X.astype(np.float32)
                weights = weights.astype(np.float32)
                X[j:j + p_size, k:k + p_size, :] += nearest_neighbor_patch * weights[t]
                Rp[j:j + p_size, k:k + p_size, :] += weights[t]
                t += 1
        
        R = R.astype(np.float32)
        X = X.astype(np.float32)
        
                
        R += 0.0001  # to avoid dividing by zero.
        X /= R
        
    return X
