import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

def IRLS(X, X_patches, style_patches, neighbors, patch_size, iterations , sub_sampling , r):
    """
    Performs IRLS robust optimization between content and style patches.
    Args:
        X: Content before optimization
        X_patches: Content patches
        style_patches: Style patches
        neighbors: Number of neighbors to consider
        patch_size: Size of patch
        iterations: Number of iterations
        sub_sampling: Sub sampling gap between patches
        r: Robust statistic value
    Returns:
        X: Estimated X after IRLS optimization
    """
    
    for i in range(iterations):
        # Reshape the patches to 2D array 
        x_features = X_patches.reshape(-1, X_patches.shape[3] * X_patches.shape[3] * 3) 
        
        # # PCA projection to reduce the dimensionality of the feature vectors
        # # if patch_size <= 21: (Can be experimented with)
        # # Choose the number of components to keep such that 95% of the variance is retained
        # threshold = 0.95 # As mentioned in the paper
        # # When n_components is between 0 and 1, select the number of components such that the amount of variance 
        # # that needs to be explained is greater than the percentage specified by n_components 
        # # svd_solver is full to avoid the error
        # pca = PCA(n_components=threshold, svd_solver='full')
        # x_features = pca.fit_transform(x_features)
        
        # Reshape the style patches to 2D array
        style_patches_iter = style_patches.reshape(-1, style_patches.shape[3] * style_patches.shape[3] * 3)
        
        # KNN
        distances, indices = neighbors.kneighbors(x_features)
        distances += 0.0001 # To avoid division by zero (Can be experimented with) (distance +=0.00001)
        W = np.power(distances, r-2) # Robust statistic to reduce the effect of outliers
        
        # A is the matrix of weights
        A = np.zeros((X.shape[0], X.shape[1], 3) , dtype=np.float64)
        A_patches = extract_patches_2d(A, patch_size= [patch_size, patch_size])
        
        # Reset X (Why ?)
        X.fill(0)

        # Initialize cumulative index 
        cumulative_index = 0

        # Loop over all patches in X
        for patch_index_1 in range(X_patches.shape[0]):
            for patch_index_2 in range(X_patches.shape[1]):

                # Find the nearest neighbor for the current patch
                nearest_style_patch_flat = style_patches_iter[indices[cumulative_index, 0]].flatten()

                # Reshape nearest_style_patch_flat into an array compatible with X_patches
                nearest_style_patch = nearest_style_patch_flat.reshape(patch_size, patch_size, 3)

                # Update the current patch based on the nearest neighbor and weights
                X_patches[patch_index_1, patch_index_2] += nearest_style_patch * W[cumulative_index]

                # Update cumulative weights
                A_patches[patch_index_1, patch_index_2] += W[cumulative_index]

                # Increment cumulative index
                cumulative_index += 1



        # Add a small constant to cum_mat to prevent division by zero
        A += 0.0001

        # Divide X by cum_mat
        X /= A
        
            
    return X
