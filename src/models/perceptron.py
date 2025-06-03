# Train the perceptron

from typing import Tuple
import numpy as np

from src.preprocessing.data_transforms import shuffle_data, normalize
from src.components.activations import step_function
from src.components.blocks import compute_weighted_sum


def predict(
    transformed_image: np.ndarray,
    weights: np.ndarray,
    bias_size: int = 1,
    backward=False,
    idx2label: dict = None
) -> int:
    """
    Perform a forward pass through the perceptron.

    Args:
        transformed_image (np.ndarray): The input image after transformation.
        weights (np.ndarray): The weights of the perceptron.
        bias_size (int): The size of the bias term.
        backward (bool): If True, return idx otherwise return the label.

    Returns:
        int: The prediction made by the perceptron.
    """
    # apply random conections from the input to the assosciated weights

    # binary classifier case
    if len(weights.shape) == 1:
        # Calculate the weighted sum of inputs and weights
        weighted_sum = compute_weighted_sum(transformed_image, weights, bias_size)

        # Apply the activation function
        preds = step_function(weighted_sum, 0)
        return preds if backward else idx2label.get(preds.item())

    # Multiclass case
    else:
        preds = np.zeros(weights.shape[0])

        # Calculate the weighted sum of inputs and weights for each class
        for i in range(weights.shape[0]):
            weighted_sum = compute_weighted_sum(
                transformed_image, weights[i], bias_size
            )

            # Apply the activation function add the result to the predictions
            preds[i] = step_function(weighted_sum, 0)

        # Return the index of the class with the highest prediction
        return np.argmax(preds)


def backward_pass(
    transformed_image: np.ndarray,
    weights: np.ndarray,
    label: int,
    learning_rate: float = 0.1,
    bias_size: int = 1,
    label2idx: dict = None,
    idx2label: dict = None,
) -> Tuple[np.ndarray, float]:
    """
    Perform a backward pass through the neural network.

    Args:
        transformed_image (np.ndarray): The input image after transformation.
        weights (np.ndarray): The weights of the neural network.
        label (int): The true label of the image.
        learning_rate (float, optional): The learning rate for weight updates.
        bias_size (int, optional): amount of additional weights for the bias term. Defaults to 1.

    Returns:
        Tuple[np.ndarray, float]: Updated weights and the error.
    """

    # Forward pass
    prediction = predict(transformed_image, weights, bias_size, backward=True, idx2label=idx2label)

    # Binary classifier case
    if len(weights.shape) == 1:
        # Calculate the error
        error = label2idx.get(label) - prediction

        # Update weights and bias
        weights[:-bias_size] += learning_rate * error * transformed_image
        weights[-bias_size] += learning_rate * error

        loss = abs(error)

    # Multiclass case
    else:
        loss = 0
        if prediction != label:
            # Calculate the error
            loss = 1

            # update the weights
            # first decrease the weights for the misclassified class as the error is 1, we don't need to multiply with the error
            weights[prediction][:-bias_size] -= learning_rate * transformed_image
            weights[prediction][-bias_size] -= learning_rate
            # then increase the weights for the correct class
            weights[label][:-bias_size] += learning_rate * transformed_image
            weights[label][-bias_size] += learning_rate

    return weights, loss


def train_perceptron(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 1,
    learning_rate: float = 0.1,
    bias_size: int = 1,
) -> np.ndarray:
    """trains the perceptron using the training data.

    Args:
        X_train (np.ndarray): training data
        y_train (np.ndarray): training labels
        epochs (int, optional): number of epochs. Defaults to 100.
        learning_rate (float, optional): learning rate. Defaults to 0.1.

    Returns:
        np.ndarray: weights
    """

    idx2label = dict(enumerate(np.unique(y_train)))
    label2idx = {v: k for k, v in idx2label.items()}

    # initialize weights
    w_len = X_train[0].flatten().shape[0]
    n_classes = np.unique(y_train).shape[0]
    X_train = normalize(X_train, 0, 1).round(decimals=0)
    # if we have a binary classifier, we need only one array of weights else we need n_classes of weightarrays
    weights = (
        np.zeros(w_len + bias_size)
        if n_classes == 2
        else np.zeros((n_classes, w_len + bias_size))
    )
    # if you want to use random connections for the weight input, you can uncomment the following line
    # random_weight = step_function(np.diag(np.random.rand(w_len)), 0) # omitted random connections

    # initialize metrics
    train_metrics = {"loss": np.zeros(epochs), "accuracy": np.zeros(epochs)}

    # iterate over the epochs
    for epoch in range(epochs):
        # iterate over the training data

        # shuffle the data
        X_train, y_train = shuffle_data(X_train, y_train)

        epoch_loss = 0
        for image, label in zip(X_train, y_train):
            # transform the image to a binary vector
            transformed_image = image.flatten()
            # transformed_image = np.matmul(random_weight, transformed_image) # omitted random connections
            weights, loss = backward_pass(
                transformed_image,
                label=label,
                weights=weights,
                learning_rate=learning_rate,
                bias_size=bias_size,
                label2idx=label2idx,
                idx2label=idx2label,
            )
            epoch_loss += loss
        # as we use only binary values of 0 and 1, the loss is the number of misclassified images
        # we can calculate the accuracy as well

        train_metrics["loss"][epoch] = epoch_loss
        train_metrics["accuracy"][epoch] = (len(y_train) - epoch_loss) / len(y_train)
        
        if epoch % (epochs // 10 if epochs >= 10 else 1) == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch}: Loss: {epoch_loss:.4f}, Accuracy: {train_metrics['accuracy'][epoch]:.4f}"
            )
    return weights, train_metrics