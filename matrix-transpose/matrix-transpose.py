import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    a=np.array(A)
    row,col=a.shape
    res=np.zeros((col,row))
    
    for i in range(row):
        for j in range(col):
            res[j][i] = a[i][j]

    return res
