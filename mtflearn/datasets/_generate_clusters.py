import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def generate_two_blobs(
        variable_center,
        fixed_center=[0, 0],
        n_samples=500,
        variable_proportion=0.5,
        cluster_std=0.5,
        random_state=None,
        plot=False
):
    """
    Generate a dataset with two Gaussian blobs, optionally imbalanced in size.

    Parameters:
    - variable_center (list or array-like): Coordinates of the second blob center.
    - fixed_center (list or array-like): Coordinates of the first blob center (default is [0, 0]).
    - n_samples (int): Total number of points to generate.
    - variable_proportion (float): Fraction of samples to assign to the variable center (between 0 and 1).
    - cluster_std (float): Standard deviation of the blobs.
    - random_state (int or None): Seed for reproducibility.
    - plot (bool): Whether to display a scatter plot of the generated blobs.

    Returns:
    - X (ndarray): Array of shape (n_samples, 2) with feature vectors.
    - y (ndarray): Array of shape (n_samples,) with cluster labels (0 or 1).
    """
    # Compute number of samples per cluster
    n_var = int(n_samples * variable_proportion)
    n_fix = n_samples - n_var
    centers = [fixed_center, variable_center]

    X, y = make_blobs(
        n_samples=[n_fix, n_var],
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state
    )

    # Sort labels in y and reorder X accordingly
    sorted_indices = np.argsort(y)
    X = X[sorted_indices]
    y = y[sorted_indices]

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='o', edgecolor='k')
        ax.scatter(*zip(*centers), c='red', marker='x', s=200, label='Centers')
        ax.legend()
        ax.axis('equal')

    return X, y

