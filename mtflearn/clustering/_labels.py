import numpy as np

def labels2matrix(labels):
    """
    Convert a 1D array of class labels into an n x n matrix where entry (i, j) is 1
    if labels[i] == labels[j], and 0 otherwise.

    Parameters:
    labels (np.ndarray): 1D numpy array of class labels.

    Returns:
    np.ndarray: n x n matrix with entries 1 if labels are the same, 0 otherwise.
    """
    labels = np.asarray(labels)
    return (labels[:, None] == labels[None, :]).astype(int)