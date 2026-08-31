import numpy as np

def r2_score(y_true: list, y_pred: list) -> float:
    """
    Returns the coefficient of determination as a Python float.
    """
    # Write code here
    y_pred=np.asarray(y_pred,dtype=float)
    y_true=np.asarray(y_true,dtype=float)

    rss=np.sum((y_true-y_pred)**2)
    tss=np.sum((y_true-np.mean(y_true))**2)
    if tss == 0:
        if rss ==0:
            return 1.0
        return 0.0
    fract=rss/tss

    r_sq=1-fract
    return float(r_sq)
    