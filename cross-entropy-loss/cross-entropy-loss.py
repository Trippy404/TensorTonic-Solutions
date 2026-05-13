import numpy as np

def cross_entropy_loss(y_true, y_pred) -> float:
    """
    Compute average multi-class cross-entropy loss.

    y_true: shape (N,)  -> integer class labels
    y_pred: shape (N, C) -> predicted probabilities (each row sums to 1)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Number of samples
    N = len(y_true)

    # Clip probabilities to avoid log(0)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # Select predicted probability of correct class
    correct_probs = y_pred[np.arange(N), y_true]

    # Compute loss
    loss = -np.mean(np.log(correct_probs))

    return float(loss)
    
    