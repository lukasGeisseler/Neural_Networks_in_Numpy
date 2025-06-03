import numpy as np

def initialize_randn_weights(
    n_inputs: int,
    n_classes: int,
    bias_size: int = 1,
    mean: float = 0,
    std: float = 1,
    seed: int | None = None,
) -> np.ndarray:
    """
    Initialize weights with random values from a normal distribution (mean = 0, variance = 1).

    Args:
        n_inputs (int): Number of input features.
        n_classes (int): Number of output neurons if n_classes >= 2, otherwise it is a binary classification problem.
        bias_size (int): Size of the bias term. Default is 1.
        mean (float): mu or center of the normal distribution. Default is 0.
        std (float): Standard deviation of the normal distribution. Default is 1.
        seed (int | None): Random seed for reproducibility. Default is None.
    Returns:
        np.ndarray: Randomly initialized weights of shape (n_inputs + bias_size, n_outputs).
    """

    if n_classes <= 1:
        raise ValueError(
            "n_classes must be greater than 1 for a classification problem."
        )

    rng = np.random.default_rng(seed) if seed else np.random.default_rng()
    return (
        rng.normal(loc=mean, scale=std, size=(n_inputs + bias_size))
        if n_classes == 2
        else rng.normal(
            loc=mean, scale=std, size=(n_classes, n_inputs + bias_size)
        )
    )