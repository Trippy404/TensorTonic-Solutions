import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """
    # Write code here
    x=np.asarray(X,dtype=float)
    cent=x-np.mean(x,axis=0)
    trans=cent.T
    cov=(trans @ cent)/(x.shape[0]-1)

    return cov

    