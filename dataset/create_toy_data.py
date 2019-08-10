import math
import numpy as np


def toy_data_for_linear_regression(weight, bias, n, variance=0.001, seed=0):
    np.random.seed(seed)
    x = np.random.randn(n, len(weight))
    y = np.dot(x, weight) + bias + np.random.randn(n) * math.sqrt(variance)
    return x, y


def toy_data_for_logistic_regression(weight, bias, n, variance=0.1, seed=0):
    np.random.seed(seed)
    x = np.random.randn(n, len(weight))
    offset = np.zeros(len(weight))
    offset[0] = -bias / weight[0]
    x += offset
    logit = np.dot(x, weight) + bias + np.random.randn(n) * math.sqrt(variance)
    y = np.where(logit > 0, 1, 0)
    return x, y
