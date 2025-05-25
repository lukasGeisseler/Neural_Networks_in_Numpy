import numpy as np

def compute_weighted_sum(X_transformed: np.ndarray, weights: np.ndarray, bias_size: int = 1) -> np.ndarray:
    """computes the weighted sum of the input array and the weights + bias.
    Args:
        X_transformed (np.ndarray): transformed input array
        weights (np.ndarray): weights of the perceptron
        bias_size (int, optional): size of the bias. Defaults to 1.
    Returns:
        np.ndarray: weighted sum of the input array and the weights + bias
    """
    #return np.matmul(X_transformed, weights[:-bias_size]) + weights[-bias_size]
    return X_transformed @ weights[:-bias_size] + weights[-bias_size]