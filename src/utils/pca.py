
import collections
import numpy as np

def pca (patches):
    # Get Mean
    mean = np.mean(patches, axis=0)
    # Center Data
    patches = patches - mean
    # Get Covariance Matrix
    cov = np.cov(patches.T)
    # Get Eigen Values and Eigen Vectors
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov)
    # Sort Eigen Values and Eigen Vectors
    eig_val_vec = {}
    for i in range(eig_val_cov.shape[0]):
        eig_val_vec[eig_val_cov[i]] = eig_vec_cov[:, i]
    # ordering eigen values in decending order
    eig_val_vec = dict(collections.OrderedDict(sorted(eig_val_vec.items(), reverse=True)))
    i = 0
    for key in eig_val_vec:
        eig_val_cov[i] = key
        eig_vec_cov[:, i] = eig_val_vec[key]
        i = i + 1
    # Normlizing eigen values
    eig_val_cov = eig_val_cov / np.sum(eig_val_cov)
    # Get Projection Matrix
    k = 0
    summation = 0
    for i in range(0, eig_val_cov.shape[0]):
        if summation >= 0.95: # 95% of the variance
            break
        k = k + 1
        summation = summation + eig_val_cov[i]
    ep = np.zeros((k, patches.shape[1]))
    for i in range(k):
        ep[i] = eig_vec_cov[:, i].real
    # Project Data
    patches = np.matmul(ep, patches.T).T
    return (patches, ep)