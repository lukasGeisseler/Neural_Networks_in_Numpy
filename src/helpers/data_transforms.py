import numpy as np

def normalize(
    X: np.ndarray[np.ndarray[np.ndarray[np.float32]]], x_min: float, x_max: float
) -> np.ndarray[np.ndarray[np.ndarray[np.float32]]]:
    """norm normalizes the values of the input arrays between x_min and x_max.

    Args:
        X (np.ndarray): image data as np.ndarrays
        x_min (float): min value of the transformed input array
        x_max (float): max value of the transformed input array

    Returns:
        np.ndarray: transformed input array
    """
    nom = (X - X.min(axis=0)) * (x_max - x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom == 0] = 1
    return x_min + nom / denom