# download the mnist dataset

import gzip
import os
from urllib.request import urlretrieve

import numpy as np


def mnist(path: str = None) -> tuple[np.ndarray]:
    """mnist downloads the mnist dataset from http://yann.lecun.com/exdb/mnist/ and saves it as .gz files to ./data/mnist if no other path is specified.
    It returns the train and test dataset and their labels as np.ndarrays.

    Args:
        path (str, optional): path where the mnist data is saved. Defaults to ./data/mnist.

    Returns:
        tuple[np.ndarray,...]: train_images, train_labels, test_images, test_labels
    """

    url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    if path is None:
        path = os.path.abspath("./data/mnist")
    else:
        path = os.path.abspath(path)

    os.makedirs(path, exist_ok=True)

    for file in files:
        if file not in os.listdir(path):
            urlretrieve(url + file, os.path.join(path, file))
            print(f"Downloaded {file} to {path}")

    def images(path):
        with gzip.open(path) as f:
            pixels = np.frombuffer(f.read(), "B", offset=16)
        return pixels.reshape(-1, 28, 28).astype(np.float32)  # / 255

    def labels(path):
        with gzip.open(path) as f:
            integer_labels = np.frombuffer(f.read(), "B", offset=8)

        return integer_labels

    train_images = images(os.path.join(path, files[0]))
    train_labels = labels(os.path.join(path, files[1]))
    test_images = images(os.path.join(path, files[2]))
    test_labels = labels(os.path.join(path, files[3]))

    print(
        f"""Dataset MNIST
    Number of datapoints     
    Train: {train_images.shape[0]:6}
    Test: {test_images.shape[0]:7}
    Source: {url}
"""
    )

    return train_images, train_labels, test_images, test_labels