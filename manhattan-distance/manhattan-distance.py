import numpy as np

def manhattan_distance(x: list, y: list) -> float:
    """
    Returns the Manhattan distance as a Python float.
    """
    x=np.asarray(x)
    y=np.asarray(y)

    l1=float(np.abs(x-y).sum())
    
    return l1
    