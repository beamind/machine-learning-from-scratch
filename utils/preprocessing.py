import numpy as np


def standardize(x: np.ndarray, axis=0):
    """
    Standardize `x` by removing the mean and scaling to unit variance along `axis`.
    """
    mean = x.mean(axis=axis, keepdims=True)
    result = x - mean
    delta = np.sqrt(np.square(result).mean(axis=axis, keepdims=True))
    result /= delta
    return result
