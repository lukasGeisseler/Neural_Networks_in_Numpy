import os
import sys
import logging
from enum import Enum
import numpy as np

# Add the project root directory to sys.path to allow for absolute imports
# Assuming this script is in src/models/, the project root is two levels up.
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, project_root)

from src.utils.load_data import mnist
from src.preprocessing.data_transforms import normalize, shuffle_data, one_hot_encoding, reshape_input
from src.components.initializers import initialize_randn_weights
from src.components.activations import step_function
from src.components.blocks import compute_weighted_sum
from src.utils.plot import plot_metrics

logging.basicConfig(
    format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)

class Activations(Enum):
    STEP_FUNCTION = step_function
    # SIGMOID = sigmoid
    # RELU = relu


def zero_initializer(
    n_inputs: int, n_classes: int, bias_size: int = 1, mean: float = 0, std: float = 1
):
    return np.zeros_like(
        initialize_randn_weights(n_inputs, n_classes, bias_size, mean=mean, std=std)
    )


class Initializers(Enum):
    RANDOM_NORMAL = initialize_randn_weights
    ZERO = zero_initializer

print(getattr(Initializers, "RANDOM_NORMAL"))

class Perceptron:
    def __init__(
        self, activation: str = "step_function", initializer: str = "random_normal"
    ):
        """Initializes a Perceptron instance with the specified activation and weight initializer.

        Sets up the perceptron's configuration, including activation function, weight initializer, and internal state variables.

        Args:
            activation (str): The activation function to use. Defaults to "step_function".
            initializer (str): The weight initializer to use. Defaults to "random_normal".
        """
        self.bias_size = 1
        self.activation = getattr(Activations, activation.upper())  # get(activation.upper())
        self.initializer = getattr(Initializers, initializer.upper())
          # get(initializer.upper())
        self.learning_rate = None
        self.epochs = None
        self.weights = None
        self.x = None
        self.y = None
        self.num_labels = None
        self.num_images = None
        self.num_features = None
        self.metrics = None

    def predict(self, X: np.ndarray, preprocess=True, **kwargs) -> np.ndarray:
        """
        Predict the labels for the input data using the trained weights.

        Args:
            X (np.ndarray): The input data to predict.
            preprocess (bool): Whether to preprocess the input data. Defaults to True.

        Keyword Args:
            threshold (float): The threshold for the step function. Defaults to 0.0.

        Returns:
            np.ndarray: The predicted labels.
        """
        if preprocess:
            X = reshape_input(normalize(X, 0, 1).round(decimals=0))
        weighted_sum = compute_weighted_sum(X, self.weights.T, self.bias_size)
        return np.argmax(
            self.activation(weighted_sum, threshold=kwargs.get("threshold", 0.0))
        )

    def backward_pass(self, x, y, **kwargs) -> int:
        """
        Perform a backward pass through the perceptron to update weights.

        Args:
            x (np.ndarray): The input data for a single example.
            y (np.ndarray): The true one-hot encoded label of the example.

        Keyword Args:
            kwargs: Additional arguments, such as thresholds for activation functions.

        Returns:
            int: Returns 1 if the weights were updated due to a misclassification, otherwise 0.
        """

        pred = self.predict(x, kwargs=kwargs, preprocess=False)
        if pred != np.argmax(y):
            error_signal = y - one_hot_encoding(pred, self.num_labels)
            # print(error_signal)
            ## same as decrease the weights for the misclassified class and increase for the correct
            ## here we have a 1 for the correct class and -1 for the incorrect class
            update = (self.learning_rate * error_signal).reshape(-1, 1) * np.stack(
                [np.hstack((x, np.ones(self.bias_size)))] * self.num_labels
            )
            ## therefore we should add the update to the weights
            self.weights += update
            return 1
        return 0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        """
        Train the perceptron with the given training data.

        Args:
            X_train (np.ndarray): The input data to train. 
            y_train (np.ndarray): The true labels of the input data.

        Keyword Args:
            epochs (int): The number of epochs to train. Defaults to 10.
            learning_rate (float): The learning rate for updating the weights. Defaults to 0.1.
            bias_size (int): The size of the bias term. Defaults to 1.
            plot_metrics (bool): Whether to plot the metrics at the end of training. Defaults to False.

        Returns:
            self: The trained perceptron.
        """
        self.num_labels = np.unique(y_train).shape[0]
        self.x = reshape_input(normalize(X_train, 0, 1).round(
            decimals=0
        ))
        print(self.x.shape)
        self.num_images, self.num_features = self.x.shape
        self.y = one_hot_encoding(y_train, num_labels=len(np.unique(y_train)))
        self.weights = self.initializer(
            self.num_features, self.num_labels + 1, self.bias_size, 0.0, 0.01
        )[:-1]  # hack to omit binary weight altohg there are only 2 classes present.
        # self.weights = np.zeros_like(self.weights)
        self.epochs = kwargs.get("epochs", 10)
        self.learning_rate = kwargs.get("learning_rate", 0.1)
        self.bias_size = kwargs.get("bias_size", 1)
        self.metrics = {
            "loss": np.zeros(self.epochs),
            "accuracy": np.zeros(self.epochs),
        }

        logging.info("Starting training for %d epochs", self.epochs)
        for epoch in range(self.epochs):
            x, y = shuffle_data(self.x, self.y.T)

            epoch_loss = sum(
                self.backward_pass(image, label, kwargs=kwargs)
                for image, label in zip(x, y)
            )
            self.metrics["loss"][epoch] = epoch_loss
            self.metrics["accuracy"][epoch] = (
                self.num_images - epoch_loss
            ) / self.num_images

            if (
                epoch % (self.epochs // 10 if self.epochs >= 10 else 1) == 0
                or epoch == self.epochs - 1
            ):
                logging.info(
                    "Epoch %d: Loss: %.4f, Accuracy: %.4f",
                    epoch,
                    epoch_loss,
                    self.metrics["accuracy"][epoch],
                )
        if kwargs.get("plot_metrics"):
            plot_metrics(self.metrics)

    def eval(self, X_test: np.ndarray, y_test: np.ndarray, **kwargs):   
        """
        Evaluate the model on the given test data.

        Args:
            X_test (np.ndarray): The input data to evaluate. 
            y_test (np.ndarray): The true labels of the input data.

        Keyword Args:
            kwargs: Additional arguments, such as thresholds for activation functions.

        Returns:
            float: The test accuracy of the model.
        """
        preds = [self.predict(image, kwargs=kwargs) for image in X_test]
        self.metrics["test_accuracy"] = np.mean(preds == y_test)
        return self.metrics["test_accuracy"]


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = mnist()
    print(X_train.shape)
    perceptron = Perceptron(activation="step_function", initializer="zero")
    perceptron.fit(X_train, y_train, epochs=10, learning_rate=0.001, threshold=0.5)
    logging.info("Test accuracy: %.4f", perceptron.eval(X_test, y_test, threshold=0.5, max_dim=1))
