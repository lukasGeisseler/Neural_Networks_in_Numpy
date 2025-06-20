# Module contains functions to plot the data

from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np

# plot images
def plot_examples(
    images: Iterable,
    labels: Iterable,
    predictions: Iterable | None = None,
    num_row: int = 2,
    num_col: int = 5,
    fig_title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """plot_examples plots several images in a grid.

    Args:
        images (Iterable): image data as Iterable of np.ndarrays
        labels (Iterable): labels of the images as Iterables
        num_row (int, optional): number of rows in the grid
        num_col (int, optional): number of columns in the grid
        title (str, optional): title of the plot. Defaults to None.
        vmin (float, optional): minimum value for the color scale. If none min(images) is used. Defaults to None.
        vmax (float, optional): maximum value for the color scale. If none max(images) is used. Defaults to None.
    """
    if vmin is None:
        vmin = np.min(images)
    if vmax is None:
        vmax = np.max(images)
        
    if len(images) < num_row * num_col:
        raise ValueError(
            f"Not enough images to fill the grid. {len(images)} < {num_row * num_col}"
        )
    if len(labels) < num_row * num_col:
        raise ValueError(
            f"Not enough labels to fill the grid. {len(labels)} < {num_row * num_col}"
        )
    if len(images) != len(labels):
        raise ValueError(
            f"Number of images and labels do not match. {len(images)} != {len(labels)}"
        )
    if num_row < 1 or num_col < 1:
        raise ValueError(
            f"Number of rows and columns must be greater than 0. {num_row} {num_col}"
        )
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
    fig.suptitle(fig_title, fontsize=16)
    for i in range(num_row * num_col):
        ax = axes if num_row == 1 and num_col == 1 else axes.flat[i]
        ax.imshow(images[i], cmap="gray", vmin=vmin, vmax=vmax)
        if predictions is not None:
            ax.set_title(f"Label: {labels[i]}\nPred: {predictions[i]}")
        else:
            ax.set_title(f"Label: {labels[i]}")
        ax.tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
        )
    plt.tight_layout()
    plt.show()


def plot_layers(layers: Dict[str, np.ndarray]) -> None:
    """
    plot_layers plots the given layers as images in a row.

    Args:
        layers (dict): dictionary with the layers to plot. The key is the name of the layer and the value is the 2D numpy array of the layer.
    """
    if len(layers) > 1:
        _, ax = plt.subplots(1, len(layers), figsize=[15, 5])
        for i, (name, layer) in enumerate(layers.items()):
            ax[i].imshow(layer, cmap="binary")
            ax[i].set(title=f"{name}\n{layer.shape}")
            ax[i].tick_params(
                left=False,
                right=False,
                labelleft=False,
                labelbottom=False,
                bottom=False,
            )
            ax[i].spines[["right", "top", "bottom", "left"]].set_visible(False)
        plt.show()
    else:
        name = list(layers.keys())[0]
        layer = layers.get(name)
        plt.imshow(layer, cmap="binary")
        plt.title(f"{name}\n{layer.shape}")


def plot_metrics(metrics: Dict[str, np.ndarray]) -> None:
    """
    plot_loss_accuracy plots the loss and accuracy of the model during training.

    Args:
        metrics (dict): dictionary with the metrics to plot.
    """
    fig, ax1 = plt.subplots()

    fig.suptitle("Loss and Accuracy")
    color = "tab:blue"
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("loss", color=color)
    ax1.plot(metrics["loss"], color=color, linewidth=0.5)
    ax1.tick_params(axis="y", labelcolor=color)
    if metrics.get("val_loss"):
        ax1.plot(
            metrics["val_loss"], color=color, linewidth=0.5, linestyle="--"
        )
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:orange"
    ax2.set_ylabel("accuracy", color=color)  # we already handled the x-label with ax1
    ax2.plot(metrics["accuracy"], color=color, linewidth=0.5)
    if metrics.get("val_accuracy"):
        ax2.plot(metrics["val_accuracy"], color=color, linewidth=0.5, linestyle="--")
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.legend(labels=["train_loss", "val_loss"])
    ax2.legend(labels=["train_acc", "val_acc"])
    plt.show()
    plt.show()
