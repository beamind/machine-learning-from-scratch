import math
import numpy as np


def toy_data_for_linear_regression(weight, n, m, variance=0.001, seed=0):
    np.random.seed(seed)
    x = np.random.randn(n, m - 1)
    x = np.concatenate([np.ones((n, 1)), x], axis=1)
    y = np.dot(x, weight) + np.random.randn(n) * math.sqrt(variance)
    return x, y
