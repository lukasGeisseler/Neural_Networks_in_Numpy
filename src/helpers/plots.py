# Module contains functions to plot the data

from typing import Iterable
import matplotlib.pyplot as plt

# plot images
def plot_examples(images:Iterable, labels:Iterable, predictions:Iterable | None=None, num_row:int=2, num_col:int=5) -> None:
    """plot_examples plots several images in a grid.

    Args:
        images (Iterable): image data as Iterable of np.ndarrays
        labels (Iterable): labels of the images as Iterables
        num_row (int, optional): number of rows in the grid
        num_col (int, optional): number of columns in the grid
    """
    
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
    for i in range(num_row * num_col):
        ax = axes[i // num_col, i % num_col]
        ax.imshow(images[i], cmap="gray")
        if predictions is not None:
            ax.set_title(
                f"Label: {labels[i]}\nPred: {predictions[i]}"
            )
        else:
            ax.set_title(f"Label: {labels[i]}")
        ax.tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
        )
    plt.tight_layout()
    plt.show()
