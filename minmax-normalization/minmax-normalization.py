import numpy as np

def minmax_scale(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns a floating-point NumPy array matching the shape of X.
    """
    x=np.asarray(X,dtype=float)
    x_min=np.min(x,axis=axis,keepdims=True)
    x_max=np.max(x,axis=axis,keepdims=True)
    data_r=x_max-x_min
    safe_range=np.where(data_r > eps ,data_r,1.0)
    return (x-x_min)/safe_range