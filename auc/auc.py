import numpy as np

def auc(fpr: list, tpr: list) -> float:
    """
    Returns the area as a float.
    """
    # Write code here
    fpr=np.asarray(fpr,dtype=float)
    tpr=np.asarray(tpr,dtype=float)
    width=np.diff(fpr)
    height= 0.5*(tpr[:-1]+tpr[1:])
    return float(np.sum(height*width))