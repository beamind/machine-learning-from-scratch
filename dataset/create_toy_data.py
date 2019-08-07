import math
import numpy as np


def toy_data_for_linear_regression(weight, bias, n, variance=0.001, seed=0):
    np.random.seed(seed)
    x = np.random.randn(n, len(weight))
    y = np.dot(x, weight) + bias + np.random.randn(n) * math.sqrt(variance)
    return x, y
