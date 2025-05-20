"""This file contains all the activations from the notebooks stroed additionally for later imports"""

import numpy as np

def step_function(x: np.ndarray, threshold:float = 0.0) -> np.ndarray:
    """step_function applies the step activation function to the input array.

    Args:
        x (np.ndarray): input array
        threshold (float, optional): threshold value for the step function. Defaults to 0.0.

    Returns:
        np.ndarray: transformed input array, where values greater than the threshold are set to 1, and others to 0.
    """
    return np.where(x > threshold, 1, 0)
