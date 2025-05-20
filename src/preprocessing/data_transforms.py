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


def standard_scale(X: np.ndarray, verbose: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
    """
    Calculates mean and std of the training data, and then scales the training data

    Args:
        X (np.ndarray): The training data, where rows are samples
                              and columns are features.

    Returns:
        if verbose is False:
            - x_scaled (np.ndarray): The scaled training data.
        if verbose is True:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - x_scaled (np.ndarray): The scaled training data.
                - mean (np.ndarray): The mean of each feature in X.
                - std (np.ndarray): The standard deviation of each feature in X.
    """
    if X.ndim == 1: # Handle case where X_train might be a single feature vector
        X = X.reshape(-1, 1)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # Add a small epsilon to std to prevent division by zero if a feature has zero variance
    epsilon = 1e-8
    x_scaled = (X - mean) / (std + epsilon)

    return  (x_scaled, mean, std) if verbose else x_scaled
