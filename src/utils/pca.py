import numpy as np

def pca (patches):
    """
    Performs PCA on the given patches.
    Args:  
        patches: Patches to perform PCA on
    Returns:
        patches: Transformed patches
        ep: Projection matrix
    """
    # Get Mean
    mean = np.mean(patches, axis=0)
    # Center Data
    patches = patches - mean
    # Get Covariance Matrix 
    cov = np.cov(patches.T)
    # Get Eigen Values and Eigen Vectors 
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov)
    # Sort Eigen Values and Eigen Vectors in descending order
    eig_val_vec_pairs = [(eig_val_cov[i], eig_vec_cov[:, i]) for i in range(len(eig_val_cov))]
    eig_val_vec_pairs.sort(key=lambda x: x[0], reverse=True)

    # Update eig_val_cov and eig_vec_cov with sorted eigen values and eigen vectors
    for i, (eig_val, eig_vec) in enumerate(eig_val_vec_pairs):
        eig_val_cov[i] = eig_val
        eig_vec_cov[:, i] = eig_vec

    # Normlizing eigen values
    eig_val_cov = eig_val_cov / np.sum(eig_val_cov)
    # Get Projection Matrix Size (k x d) where k is the number of eigen vectors to keep and d is the dimension of the data
    k = 0
    summation = 0
    for i in range(0, eig_val_cov.shape[0]):
        if summation >= 0.95: # 95% of the variance
            break
        k = k + 1
        summation = summation + eig_val_cov[i]
    ep = np.zeros((k, patches.shape[1]))
    # Get Projection Matrix  
    for i in range(k):
        ep[i] = eig_vec_cov[:, i].real
    # Project Data onto the projection matrix
    patches = np.matmul(ep, patches.T).T
    return (patches, ep)