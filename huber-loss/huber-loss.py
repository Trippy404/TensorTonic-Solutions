import numpy as np

def huber_loss(y_true: list, y_pred: list, delta: float = 1.0) -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    y_pred=np.asarray(y_pred,dtype=float)
    y_true=np.asarray(y_true,dtype=float)

    abs_err=np.abs(y_true-y_pred)

    err=y_true-y_pred

    hub_loss=np.where(
        abs_err <= delta,
        0.5*(err)**2,
        delta * (abs_err - 0.5 * delta)
    )

    return float(np.mean(hub_loss))