import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x=np.array(x)
    p=np.array(p)



    if len(x) != len(p):
        raise ValueError("x and p must have the same length")

        # Check probabilities are non-negative
    if np.any(p < 0):
        raise ValueError("Probabilities must be non-negative")

     # Check probabilities sum to 1
    if not np.allclose(np.sum(p), 1):
        raise ValueError("Probabilities must sum to 1")

    

    return np.sum(x*p)
    
