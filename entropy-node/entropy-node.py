import numpy as np

def entropy_node(y):
    y=np.array(y)
    if len(y) == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)

    prob = counts /np.sum(counts)
    prob = prob[prob > 0]

    entropy_node = - np.sum(prob*np.log2(prob))

    return float(entropy_node)
    