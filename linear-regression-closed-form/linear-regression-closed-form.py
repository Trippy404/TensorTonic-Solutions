import numpy as np

def linear_regression_closed_form(X, y):

    X=np.array(X)
    y=np.array(y)

    x_t=X.T
    y_mul=np.dot(x_t,y)
    w_1=np.linalg.inv(np.dot(x_t,X))
    w=np.dot(w_1,y_mul)

    return w
    