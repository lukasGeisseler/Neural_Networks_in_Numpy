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
from src.preprocessing.data_transforms import normalize, shuffle_data, one_hot_encoding
from src.components.initializers import initialize_randn_weights
from src.components.activations import step_function
from src.components.blocks import compute_weighted_sum
from src.utils.plot import plot_metrics

logging.basicConfig(
    format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)

class ExtendedEnum(Enum):

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

class Activations(ExtendedEnum):
    STEP_FUNCTION = step_function
    # SIGMOID = sigmoid
    # RELU = relu


def zero_initializer(
    n_inputs: int, n_classes: int, bias_size: int = 1, mean: float = 0, std: float = 1
):
    return np.zeros_like(
        initialize_randn_weights(n_inputs, n_classes, bias_size, mean=mean, std=std)
    )


class Initializers(ExtendedEnum):
    RANDOM_NORMAL = initialize_randn_weights
    ZERO = zero_initializer

print(getattr(Initializers, "RANDOM_NORMAL"))

class Perceptron:
    def __init__(
        self, activation: str = "step_function", initializer: str = "random_normal"
    ):
        self.bias_size = 1
        self.activation = Activations.STEP_FUNCTION  # get(activation.upper())
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

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        weighted_sum = compute_weighted_sum(X, self.weights.T, self.bias_size)
        return np.argmax(
            self.activation(weighted_sum, threshold=kwargs.get("threshold", 0.0))
        )

    def backward_pass(self, x, y, **kwargs) -> int:
        pred = self.predict(x, kwargs=kwargs)
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
        self.num_labels = np.unique(y_train).shape[0]
        self.x = normalize(X_train.reshape(X_train.shape[0], -1), 0, 1).round(
            decimals=0
        )
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
        x = normalize(X_test.reshape(X_test.shape[0], -1), 0, 1).round(decimals=0)
        preds = [self.predict(image, kwargs=kwargs) for image in x]
        self.metrics["test_accuracy"] = np.mean(preds == y_test)
        return self.metrics["test_accuracy"]


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = mnist()
    perceptron = Perceptron(activation="step_function", initializer="zero")
    perceptron.fit(X_train, y_train, epochs=10, learning_rate=0.001, threshold=0.5)
    logging.info("Test accuracy: %.4f", perceptron.eval(X_test, y_test, threshold=0.5))
